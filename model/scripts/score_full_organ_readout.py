"""Replay all E25 saved readers before fixed full-support paired assessment."""
import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_full_organ_readout import (
    DEST, KINDS, PARAMETERS, PROTOCOL, READERS, ROOT, features, paired_data,
    require_model, verify_full_grid, verify_raw, write_json,
)
from whole_body_jepa.multisource_intervals import sha256

ARMS = (*KINDS, 'raw', 'native')
CONTROLS = ('body', 'shuffled', 'expanded', 'raw', 'native')


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.variable.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame[['patient', 'component', 'fold', 'variable']].isna().any().any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not set(frame.fold) <= set(range(5))
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired full-support predictions')
    patients = frame[list(ARMS)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    components = patients.groupby(links.reindex(patients.index)).mean()
    comparisons = {}
    for i, control in enumerate(CONTROLS):
        stats = dict(patient=paired_summary(patients.organs-patients[control], seed+i, repetitions),
                     component=paired_summary(components.organs-components[control], seed+100+i, repetitions))
        passed = len(patients) >= 30 and len(components) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*5) for s in stats.values())
        comparisons['organs_minus_'+control] = dict(paired=stats, screen_pass=bool(passed))
    return dict(patients=len(patients), components=len(components), query_occurrences=len(frame),
        folds=sorted(map(int, frame.fold.unique())), comparisons=comparisons,
        normalized_mae=dict(patient=patients.mean().to_dict(), component=components.mean().to_dict()))


def summarize(frame, variables, repetitions=5000):
    if len(variables) != 53 or len(set(variables)) != 53 or not set(frame.variable) <= set(variables):
        raise ValueError('Complete canonical inventory required')
    targets = {v: assess(frame[frame.variable == v], 20261003+i*1000, repetitions)
               if (frame.variable == v).any() else dict(status='unsupported') for i, v in enumerate(variables)}
    scored = {v: r for v, r in targets.items() if 'normalized_mae' in r}
    if not scored:
        raise ValueError('No scored targets')
    screens = {'organs_minus_'+c: [v for v, r in scored.items()
        if r['comparisons']['organs_minus_'+c]['screen_pass']] for c in CONTROLS}
    incremental = [v for v in scored if all(v in screens['organs_minus_'+c] for c in CONTROLS[:3])]
    return targets, dict(variables=len(scored), query_occurrences=len(frame), patients=int(frame.patient.nunique()),
        variable_folds=len(frame[['variable', 'fold']].drop_duplicates()), screens=screens,
        joint_incremental=incremental,
        joint_all_controls=[v for v in incremental if all(v in screens['organs_minus_'+c] for c in CONTROLS[3:])],
        normalized_macro={level: {arm: float(np.mean([r['normalized_mae'][level][arm] for r in scored.values()]))
                                 for arm in ARMS} for level in ('patient', 'component')})


def verify_frame(frame, ids, query, variable, fold, components):
    expected = pd.DataFrame(dict(patient=ids, component=[components[p] for p in ids],
        variable=variable, fold=fold, truth=query['truth'], native=query['prediction']))
    if len(frame) != len(ids) or frame.patient.duplicated().any():
        raise ValueError('Changed saved prediction support')
    pd.testing.assert_frame_equal(frame[expected.columns].reset_index(drop=True), expected)
    if not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all():
        raise ValueError('Nonfinite saved predictions')


def main():
    output = Path('artifacts/native_reconstruction_full_organ_readout_assessment_20261003.json')
    if output.exists():
        raise ValueError('Completed E25 assessment already exists')
    grid, variables, shape, bindings = verify_full_grid()
    checked, frames, diagnostics = {}, [], []
    roles_all = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any() or links.isna().any().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    for p in (Path(__file__), Path('scripts/train_full_organ_readout.py'),
              Path('scripts/score_additional_reconstruction.py'), PROTOCOL):
        bindings[str(p)] = sha256(p)
    for fold in range(5):
        rp = Path(f'artifacts/native_reconstruction_full_organ_readers_fold{fold}_20261003.json')
        if not rp.exists():
            raise ValueError('All five complete reader folds required')
    with threadpool_limits(limits=4):
        for fold in range(5):
            rp = Path(f'artifacts/native_reconstruction_full_organ_readers_fold{fold}_20261003.json')
            report = json.loads(rp.read_text()); contract = report['contract']
            if (report['complete'] is not True or contract['fold'] != fold or contract['variables'] != variables
                    or contract['organ_shape'] != list(shape) or contract['parameters'] != PARAMETERS
                    or contract['projection_seed'] != 20261003+fold or set(report['targets']) != set(variables)):
                raise ValueError('Changed full reader contract')
            verify_bound_files(contract['bindings'], checked); bindings.update(contract['bindings'])
            bindings[str(rp)] = sha256(rp)
            dest = READERS/f'fold{fold}'; cp = dest/'contract.json'
            if json.loads(cp.read_text()) != contract:
                raise ValueError('Saved reader contract mismatch')
            raw_dir = ROOT/f'nonlinear_probe_20261001/fold{fold}_probes'
            roles, refs, organs = roles_all[fold], {}, {}
            projection = np.random.default_rng(20261003+fold).normal(size=(64, int(np.prod(shape))))/8
            try:
                for role in ('fit', 'selection'):
                    refs[role] = sqlite3.connect((ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
                    organs[role] = sqlite3.connect((DEST/f'fold{fold}_{role}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
                for c, variable in enumerate(variables):
                    meta = report['targets'][variable]
                    if variable not in grid[fold, 'fit']['support']:
                        if meta != dict(status='unsupported', fit_patients=0, selection_patients=0):
                            raise ValueError('Unreported unsupported target')
                        continue
                    if meta['status'] != 'fitted' or meta['fold'] != fold or meta['target'] != variable:
                        raise ValueError('Missing supported reader')
                    mp, pp, jp = dest/f'{variable}.joblib', dest/f'{variable}.parquet', dest/f'{variable}.json'
                    verify_bound_files({str(cp): meta['contract_sha256'], str(mp): meta['model_sha256'],
                                        str(pp): meta['prediction_sha256']}, checked)
                    if json.loads(jp.read_text()) != meta:
                        raise ValueError('Changed reader metadata')
                    bindings.update({str(p): sha256(p) for p in (cp, mp, pp, jp)})
                    fids, f = paired_data(refs['fit'], organs['fit'], c, shape, roles, 'fit')
                    qids, q = paired_data(refs['selection'], organs['selection'], c, shape, roles, 'selection')
                    if (len(fids), len(qids)) != (meta['fit_patients'], meta['selection_patients']):
                        raise ValueError('Changed full reader support')
                    old = json.loads((raw_dir/f'{variable}.json').read_text())
                    source_report = json.loads(Path(f'artifacts/native_reconstruction_nonlinear_training_fold{fold}_20261001.json').read_text())
                    if old != source_report['targets'][variable] or old['model_sha256'] != meta['raw_model_sha256']:
                        raise ValueError('Changed raw metadata')
                    verify_bound_files({str(raw_dir/'contract.json'): old['contract_sha256'],
                        str(raw_dir/f'{variable}.joblib'): old['model_sha256'],
                        str(raw_dir/f'{variable}.parquet'): old['prediction_sha256']}, checked)
                    raw, rp_values, weights = verify_raw(joblib.load(raw_dir/f'{variable}.joblib'),
                        pd.read_parquet(raw_dir/f'{variable}.parquet'), old, fold, variable, fids, qids, f, q, roles, components)
                    if hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest() != meta['weights_sha256']:
                        raise ValueError('Changed fitting weights')
                    frame = pd.read_parquet(pp); verify_frame(frame, qids, q, variable, fold, components)
                    np.testing.assert_allclose(frame.raw, rp_values, atol=1e-12, rtol=1e-12)
                    bundle = joblib.load(mp)
                    if bundle['target'] != variable or bundle['parameters'] != PARAMETERS or set(bundle['models']) != set(KINDS):
                        raise ValueError('Wrong saved model bundle')
                    np.testing.assert_array_equal(bundle['projection'], projection)
                    fit_stats = {}
                    for kind in KINDS:
                        model = bundle['models'][kind]
                        x, qx = features(f, fids, kind, projection), features(q, qids, kind, projection)
                        require_model(model, x.shape[1])
                        np.testing.assert_allclose(model.predict(qx), frame[kind], atol=1e-12, rtol=1e-12)
                        errors = abs(model.predict(x)-f['truth'])
                        fit_stats[kind] = dict(fit_mae=float(np.mean(errors)), weighted_fit_mae=float(np.average(errors, weights=weights)),
                            mean_leaves=float(np.mean([t.get_n_leaf_nodes() for group in model._predictors for t in group])))
                    errors = abs(raw.predict(f['raw'])-f['truth'])
                    fit_stats['raw'] = dict(fit_mae=float(np.mean(errors)), weighted_fit_mae=float(np.average(errors, weights=weights)),
                        mean_leaves=float(np.mean([t.get_n_leaf_nodes() for group in raw._predictors for t in group])))
                    fit_stats['native'] = dict(fit_mae=float(np.mean(abs(f['prediction']-f['truth']))))
                    for kind, stats in fit_stats.items():
                        if set(stats) != set(meta['fit_diagnostics'][kind]):
                            raise ValueError('Changed fit diagnostics')
                        for name, value in stats.items():
                            np.testing.assert_allclose(value, meta['fit_diagnostics'][kind][name], atol=1e-12, rtol=1e-12)
                    diagnostics.append(dict(variable=variable, fold=fold, fit=fit_stats))
                    frames.append(frame)
                print('Replayed full-support readers', fold, flush=True)
            finally:
                for db in (*refs.values(), *organs.values()):
                    db.close()
    data = pd.concat(frames, ignore_index=True)
    if len(data) != sum(grid[f, 'selection']['expected'] for f in range(5)) or len(diagnostics) != 47*5:
        raise ValueError('Incomplete full selection grid')
    targets, summary = summarize(data, variables)
    write_json(output, dict(summary=summary, targets=targets, fit_diagnostics=diagnostics, bindings=bindings,
        saved_predictions_reproduced=True, fit_predictions_reproduced=True,
        independent_statistical_verification_complete=False, runtime_modified=False, promotion_allowed=False,
        limitation='Reused development, masked-current readout comparison. Not future forecasting, a new lockbox, or a release gate.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
