"""Compare saved full and bounded raw readers on identical masked queries."""
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.cache_nonlinear_probe import decode_features
from scripts.probe_organ_compartments import DEST as SMALL, ROOT, write_json
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256

ARMS = ('full_raw', 'small_raw', 'native')
CONTRASTS = (('full_raw', 'small_raw'), ('full_raw', 'native'), ('small_raw', 'native'))
OUTPUT = Path('artifacts/native_reconstruction_reader_support_20261003.json')
PRIVATE = ROOT / 'reader_support_20261003'


def verify_subset(data, rows, db, target, roles, role):
    ids = data['patient'][rows].tolist()
    if len(ids) != len(set(ids)) or any(roles.get(p) != role for p in ids):
        raise ValueError('Duplicate or incorrect patient role')
    for i, p in zip(rows, ids):
        found = db.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?',
                           (p, int(target))).fetchone()
        if found is None:
            raise ValueError('Bounded context missing from full cache')
        row = decode_features(*found)
        np.testing.assert_array_equal(row['raw'], data['raw'][i])
        np.testing.assert_array_equal(row['truth'], data['truth'][i])
        np.testing.assert_allclose(row['body'], data['body'][i], atol=1e-6, rtol=0)
        np.testing.assert_allclose(row['prediction'], data['native'][i], atol=1e-7, rtol=0)


def require_roles(fit_ids, query_ids, small_fit_ids, roles):
    if (len(fit_ids) != len(set(fit_ids)) or len(query_ids) != len(set(query_ids))
            or set(fit_ids) & set(query_ids) or not set(small_fit_ids) <= set(fit_ids)
            or any(roles.get(p) != 'fit' for p in fit_ids)
            or any(roles.get(p) != 'selection' for p in query_ids)):
        raise ValueError('Invalid full/bounded support or role leakage')


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.variable.nunique() != 1
            or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid support comparison pairs')
    patient = frame[list(ARMS)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    component = patient.groupby(links.reindex(patient.index)).mean()
    comparisons = {}
    for i, (a, b) in enumerate(CONTRASTS):
        stats = dict(patient=paired_summary(patient[a]-patient[b], seed+i, repetitions),
                     component=paired_summary(component[a]-component[b], seed+100+i, repetitions))
        passed = len(patient) >= 30 and len(component) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*3) for s in stats.values())
        comparisons[a+'_minus_'+b] = dict(paired=stats, screen_pass=bool(passed))
    return dict(patients=len(patient), components=len(component), query_occurrences=len(frame),
        folds=sorted(map(int, frame.fold.unique())),
        normalized_mae=dict(patient=patient.mean().to_dict(), component=component.mean().to_dict()),
        comparisons=comparisons)


def summarize(frame, variables, repetitions=5000):
    targets = {v: assess(frame[frame.variable == v], 20261003+i*1000, repetitions)
               if (frame.variable == v).any() else dict(status='insufficient_support')
               for i, v in enumerate(variables)}
    scored = {v: r for v, r in targets.items() if 'normalized_mae' in r}
    summary = dict(variables=len(scored), patients=int(frame.patient.nunique()),
        query_occurrences=len(frame), variable_folds=len(frame[['variable', 'fold']].drop_duplicates()),
        normalized_macro={l: {a: float(np.mean([r['normalized_mae'][l][a] for r in scored.values()]))
                             for a in ARMS} for l in ('patient', 'component')},
        screens={a+'_minus_'+b: [v for v, r in scored.items()
            if r['comparisons'][a+'_minus_'+b]['screen_pass']] for a, b in CONTRASTS})
    return targets, summary


def main():
    if OUTPUT.exists() or PRIVATE.exists():
        raise ValueError('Existing support audit; do not overwrite evidence')
    checked, bindings, frames, support = {}, {}, [], []

    def bind(path, expected=None):
        path = Path(path)
        if expected is not None:
            verify_bound_files({str(path): expected}, checked)
        digest = checked.get(path.resolve()) or sha256(path)
        checked[path.resolve()] = digest
        bindings[str(path)] = digest
        return path

    for p in (Path(__file__), Path('RECONSTRUCTION_READER_SUPPORT_PROTOCOL_20261003.md'),
              Path('scripts/train_nonlinear_probe.py'), Path('scripts/probe_organ_compartments.py'),
              Path('scripts/cache_nonlinear_probe.py'), Path('scripts/score_additional_reconstruction.py')):
        bind(p)
    roles_all = torch.load(bind(ROOT/'partitions_20260922.pt'), weights_only=False)['folds']
    links = pd.read_parquet(bind(ROOT/'provenance.parquet'))[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous component provenance')
    components = dict(links.itertuples(index=False, name=None))
    full_root = ROOT/'nonlinear_probe_20261001'
    with threadpool_limits(limits=4):
        for fold in range(5):
            roles = roles_all[fold]
            sr = json.loads(bind(f'artifacts/native_reconstruction_organ_compartments_fold{fold}_20261003.json').read_text())
            fr = json.loads(bind(f'artifacts/native_reconstruction_nonlinear_training_fold{fold}_20261001.json').read_text())
            variables = sr['contract']['variables']
            if len(variables) != 53 or sr['contract']['parameters'] != PARAMETERS or fr['parameters'] != PARAMETERS:
                raise ValueError('Changed schema or model parameters')
            small_dir, full_dir = SMALL/f'fold{fold}', full_root/f'fold{fold}_probes'
            scp, fcp = bind(small_dir/'contract.json'), bind(full_dir/'contract.json')
            if json.loads(scp.read_text()) != sr['contract']: raise ValueError('Changed small contract')
            for group in (sr['contract']['bindings'], fr['bindings'],
                          sr['contract']['encoded_contract']['bindings'],
                          sr['contract']['encoded_contract']['parent']['bindings'],
                          json.loads(fcp.read_text())['bindings']):
                verify_bound_files(group, checked)
            data, databases = {}, {}
            try:
                for role in ('fit', 'selection'):
                    data[role] = joblib.load(bind(small_dir/f'{role}.joblib', sr['extraction'][role]['sha256']))
                    cr = json.loads(bind(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_{role}_20261001.json').read_text())
                    verify_bound_files(cr['bindings'], checked)
                    dbpath = bind(full_root/f'fold{fold}_{role}.sqlite', cr['private_sha256'])
                    databases[role] = sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro', uri=True)
                f, q = data['fit'], data['selection']
                for c, v in enumerate(variables):
                    sm = sr['targets'][v]
                    if sm['status'] != 'fitted': continue
                    fm, qm = np.flatnonzero(f['target'] == c), np.flatnonzero(q['target'] == c)
                    if sm['support'] != dict(fit=len(fm), selection=len(qm)): raise ValueError('Changed small support')
                    full_meta = json.loads(bind(full_dir/f'{v}.json').read_text())
                    if full_meta != fr['targets'][v]: raise ValueError('Changed full training metadata')
                    bind(fcp, full_meta['contract_sha256']); bind(scp, sm['contract_sha256'])
                    fit_ids = [p for (p,) in databases['fit'].execute('SELECT patient FROM features WHERE target=?', (c,))]
                    query_ids = [p for (p,) in databases['selection'].execute('SELECT patient FROM features WHERE target=?', (c,))]
                    require_roles(fit_ids, query_ids, f['patient'][fm], roles)
                    if len(fit_ids) != full_meta['fit_patients'] or len(query_ids) != full_meta['selection_patients']:
                        raise ValueError('Incomplete full support')
                    for role, rows in (('fit', fm), ('selection', qm)):
                        verify_subset(data[role], rows, databases[role], c, roles, role)
                    small = joblib.load(bind(small_dir/f'{v}.joblib', sm['model_sha256']))['models']['raw']['hgb']
                    full_bundle = joblib.load(bind(full_dir/f'{v}.joblib', full_meta['model_sha256']))
                    full = full_bundle['models']['raw_probe']
                    if full_bundle['target'] != v or full_bundle['parameters'] != PARAMETERS:
                        raise ValueError('Wrong full model')
                    for model in (small, full):
                        if model.n_features_in_ != 265 or model.n_iter_ != 100 or any(
                                model.get_params()[k] != val for k, val in PARAMETERS.items()):
                            raise ValueError('Unequal reader specification')
                    old = pd.read_parquet(bind(small_dir/f'{v}.parquet', sm['prediction_sha256']))
                    full_saved = pd.read_parquet(bind(full_dir/f'{v}.parquet', full_meta['prediction_sha256']))
                    if (full_saved.patient.duplicated().any() or set(full_saved.patient) != set(query_ids)
                            or not (full_saved.target == v).all() or not (full_saved.fold == fold).all()):
                        raise ValueError('Changed full selection grid')
                    aligned = full_saved.set_index('patient').loc[q['patient'][qm]]
                    frame = pd.DataFrame(dict(patient=q['patient'][qm], component=q['component'][qm],
                        variable=v, fold=fold, truth=q['truth'][qm], native=q['native'][qm]))
                    pd.testing.assert_frame_equal(old[frame.columns].reset_index(drop=True), frame)
                    if not np.array_equal(frame.component, frame.patient.map(components)):
                        raise ValueError('Changed component')
                    np.testing.assert_array_equal(aligned.truth.to_numpy(), frame.truth.to_numpy())
                    np.testing.assert_allclose(aligned.native, frame.native, atol=1e-7, rtol=0)
                    frame['small_raw'] = small.predict(q['raw'][qm])
                    frame['full_raw'] = full.predict(q['raw'][qm])
                    np.testing.assert_allclose(frame.small_raw, old.hgb_raw, atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(frame.full_raw, aligned.raw_probe, atol=1e-12, rtol=1e-12)
                    frames.append(frame)
                    support.append(dict(variable=v, fold=fold, small_fit=len(fm), full_fit=len(fit_ids),
                        selection=len(qm), selection_mae={a: float(np.mean(abs(frame[a]-frame.truth))) for a in ARMS}))
                print('Verified identical-query support comparison', fold, flush=True)
            finally:
                for db in databases.values(): db.close()
    frame = pd.concat(frames, ignore_index=True)
    targets, summary = summarize(frame, variables)
    if len(support) != 131 or len(frame) != 4569 or frame.patient.nunique() != 1851:
        raise ValueError('E23 fixed query population changed')
    PRIVATE.mkdir()
    pp = PRIVATE/'predictions.parquet'; frame.to_parquet(pp, index=False); bind(pp)
    summary['fit_support'] = {k: dict(min=int(min(s[k] for s in support)),
        median=float(np.median([s[k] for s in support])), max=int(max(s[k] for s in support)))
        for k in ('small_fit', 'full_fit')}
    write_json(OUTPUT, dict(summary=summary, targets=targets, support=support, bindings=bindings,
        checked_files=len(checked), raw_and_truth_equal=True, saved_predictions_reproduced=True,
        training_performed=False, runtime_modified=False, promotion_allowed=False,
        limitation='Reused development, masked-current task. Fit population and weights change with support; not sample-count causality or organ repair.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
