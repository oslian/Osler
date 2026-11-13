"""Evaluate frozen E29 adaptation on unused patients from fitting sources."""
import json
from numbers import Integral
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch

from scripts.audit_organ_assignment import pools
from scripts.cache_nonlinear_probe import decode_features
from scripts.diagnose_organ_assimilation import require_verification
from scripts.encoder_fold_scaling import transform_example
from scripts.probe_observed_content import contexts
from scripts.train_content_retention import replay
from scripts.train_organ_assimilation import AssimilatedReconstruction, DEST as PARENT
from scripts.train_organ_binding import prepare, select_arm, space_required, write_json
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.reconstruction_cache import ReconstructionCache

ROOT = PARENT.parent
DEST = ROOT/'assimilation_transfer_20261005'
PROTOCOL = Path('RECONSTRUCTION_ASSIMILATION_TRANSFER_PROTOCOL_20261005.md')
ARMS = ('immediate', 'delayed', 'body', 'unbridged', 'frozen')


def seen_people(seen, people, variables, roles):
    if len(set(people)) != len(people):
        raise ValueError('Duplicate people in exposure mapping')
    result = set()
    for row, target in seen:
        if (isinstance(row, bool) or isinstance(target, bool)
                or not isinstance(row, Integral) or not isinstance(target, Integral)
                or not 0 <= row < len(people) or not 0 <= target < len(variables)):
            raise ValueError('Invalid exposure index')
        patient = people[row]
        if roles.get(patient) != 'fit':
            raise ValueError('Original update outside fitting role')
        result.add(patient)
    return result


def choose_keys(keys, adapted, components, fit_components, roles, limit=1024):
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate candidate context')
    eligible = []
    for p, c in keys:
        if p not in roles or p not in components:
            raise ValueError('Missing role/provenance')
        if roles[p] != 'fit':
            raise ValueError('Nonfit query in fitting cache')
        if p not in adapted and components[p] in fit_components:
            eligible.append((p, c))
    if not eligible:
        raise ValueError('No same-source adaptation-unseen queries')
    return contexts(eligible, limit)


def summarize(frame, inventory):
    if (frame.empty or frame.duplicated(['patient', 'fold']).any()
            or frame[['patient', 'component', 'variable', 'fold']].isna().any().any()
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()
            or (frame.groupby('patient').component.nunique() > 1).any()
            or not set(frame.variable) <= set(inventory)):
        raise ValueError('Invalid diagnostic rows')
    results = {}
    for variable in inventory:
        f = frame[frame.variable == variable]
        if f.empty:
            results[variable] = dict(status='unsupported')
            continue
        errors = f[list(ARMS)].subtract(f.truth, axis=0).abs().groupby(f.patient).mean()
        link = f.drop_duplicates('patient').set_index('patient').component
        by_component = errors.groupby(link.reindex(errors.index)).mean()
        results[variable] = dict(status='scored', patients=len(errors),
            occurrences=len(f), components=len(by_component),
            epoch10_exposed_occurrences=int(f.epoch10_exposed.sum()),
            normalized_mae=dict(patient=errors.mean().to_dict(), component=by_component.mean().to_dict()))
    return results


def run_fold(fold, assessment):
    variables, old, old_bound, original, _, _, old_dest = prepare(fold)
    rp = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
    if sha256(rp) != assessment['bindings'][str(rp)]:
        raise ValueError('Changed E29 report')
    report = json.loads(rp.read_text())
    for p, h in report['contract']['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed E29 training dependency')
    fp, sp, pp = [ROOT/p for p in ('partitions_20260922.pt', 'scaling_20260922.pt', 'provenance.parquet')]
    plan = torch.load(fp, weights_only=False)['folds'][fold]
    scaling = torch.load(sp, weights_only=False)
    wp = ROOT/f'monitored_continuation_20260929/fold{0 if fold == 2 else fold}_real_epoch10.pt'
    checkpoint = torch.load(wp, weights_only=False)
    exposed = seen_people(checkpoint['seen'], scaling['people'], variables, plan)
    links = pd.read_parquet(pp)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    cp = ROOT/f'nonlinear_probe_20261001/fold{fold}_fit.sqlite'
    cache_report = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_fit_20261001.json')
    cr = json.loads(cache_report.read_text())
    if sha256(cp) != cr['private_sha256']:
        raise ValueError('Changed fitting features')
    hp = ROOT/'full_histories_20260922.sqlite'
    for path in (fp, sp, wp, hp):
        if sha256(path) != cr['bindings'][str(path)]:
            raise ValueError('Changed original input')
    models = {}
    paths = [Path(__file__), PROTOCOL, rp, fp, sp, pp, wp, cp, hp, cache_report,
             Path('scripts/audit_organ_assignment.py'), Path('scripts/probe_observed_content.py'),
             Path('scripts/cache_nonlinear_probe.py'), Path('scripts/encoder_fold_scaling.py'),
             Path('scripts/train_content_retention.py'), Path('scripts/train_organ_assimilation.py'),
             Path('scripts/organ_assimilation.py'), Path('scripts/train_organ_binding.py'),
             Path('whole_body_jepa/native_organ_observations.py'),
             Path('whole_body_jepa/native_body_reconstruction.py'),
             Path('whole_body_jepa/physiology_state.py'), Path('tests/test_assimilation_transfer.py')]
    for arm in ARMS[:3]:
        mp = PARENT/f'fold{fold}'/f'{arm}.pt'
        if sha256(mp) != report['fits'][arm]['sha256']:
            raise ValueError('Changed E29 weights')
        model = AssimilatedReconstruction(original, arm).eval()
        model.load_state_dict(torch.load(mp, weights_only=False), strict=True)
        models[arm] = model
        paths.append(mp)
    unbridged_path = old_dest/'binding.pt'
    old_report = json.loads(Path(f'artifacts/native_organ_binding_fold{fold}_20261004.json').read_text())
    if sha256(unbridged_path) != old_report['fits']['binding']['sha256']:
        raise ValueError('Changed unbridged weights')
    from copy import deepcopy
    unbridged = deepcopy(original).eval()
    unbridged.load_state_dict(torch.load(unbridged_path, weights_only=False), strict=True)
    paths.append(unbridged_path)
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    output = dest/'predictions.parquet'
    meta_path = Path(f'artifacts/native_assimilation_transfer_fold{fold}_20261005.json')
    bindings = {str(p): sha256(p) for p in paths}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['bindings'] != bindings or meta['prediction_sha256'] != sha256(output):
            raise ValueError('Changed completed transfer fold')
        return pd.read_parquet(output), meta_path
    db = sqlite3.connect(cp.resolve().as_uri()+'?mode=ro', uri=True)
    source = ReconstructionCache(hp)
    records, parity = [], 0.
    try:
        keys = choose_keys(db.execute('SELECT patient,target FROM features').fetchall(),
            {e['patient'] for e in old['fit']}, components,
            {e['component'] for e in old['fit']}, plan)
        with torch.inference_mode():
            for i, (patient, c) in enumerate(keys):
                e = transform_example(source.get(patient, c), scaling['scalers'][fold])
                clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, variables[c])
                body, organs, presence = encode_native_organ_observations(original.trajectory.jepa, variables, clean)
                manual, bound = pools(original.trajectory.jepa, variables, clean)
                torch.testing.assert_close(manual, organs, atol=1e-6, rtol=0)
                row = dict(body=body, organs=organs, presence=presence, times=times,
                    anchor=float(e['anchor_hour']), target=c, truth=float(e['truth_normalized']))
                frozen = float(original.readout(replay(original.trajectory.layer, row))[0, c])
                cached = decode_features(*db.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?', (patient, c)).fetchone())
                np.testing.assert_allclose(row['truth'], cached['truth'], atol=0, rtol=0)
                np.testing.assert_allclose(frozen, cached['prediction'], atol=1e-7, rtol=0)
                parity = max(parity, abs(frozen-float(cached['prediction'])))
                bound_row = select_arm(row, bound, 'binding')
                prediction = {arm: float(model(bound_row)) for arm, model in models.items()}
                prediction['unbridged'] = float(unbridged.readout(replay(unbridged.trajectory.layer, bound_row))[0, c])
                records.append(dict(patient=patient, component=components[patient], fold=fold,
                    variable=variables[c], context=c, truth=row['truth'], frozen=frozen,
                    epoch10_exposed=patient in exposed, **prediction))
                if (i+1) % 256 == 0:
                    print('E31 inference', fold, i+1, len(keys), flush=True)
    finally:
        db.close(); source.close()
    frame = pd.DataFrame(records)
    summarize(frame, variables)
    space_required()
    tmp = output.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(output)
    write_json(meta_path, dict(bindings=bindings, prediction_sha256=sha256(output),
        fold=fold, patients=len(frame), epoch10_exposed=int(frame.epoch10_exposed.sum()),
        frozen_prediction_max_difference=parity, runtime_modified=False, promotion_allowed=False))
    return frame, meta_path


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    ap = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    vp = Path('artifacts/native_organ_assimilation_verification_20261004.json')
    output = Path('artifacts/native_assimilation_transfer_20261005.json')
    if output.exists():
        raise ValueError('Completed diagnostic exists')
    assessment, verification = json.loads(ap.read_text()), json.loads(vp.read_text())
    require_verification(ap, verification)
    frames, bindings = [], {str(p): sha256(p) for p in (ap, vp, PROTOCOL, Path(__file__))}
    for fold in range(5):
        frame, path = run_fold(fold, assessment)
        frames.append(frame); bindings[str(path)] = sha256(path)
    frame = pd.concat(frames, ignore_index=True)
    results = summarize(frame, list(assessment['results']))
    reversed_targets = [v for v, r in assessment['results'].items() if r['status'] == 'scored'
        and r['fit_patient_mae']['immediate'] < r['fit_patient_mae']['unbridged']
        and r['normalized_mae']['patient']['immediate'] >= r['normalized_mae']['patient']['unbridged']]
    diagnosis = {}
    for v in reversed_targets:
        r = results[v]
        diagnosis[v] = ('unsupported' if r['status'] != 'scored' else
            'same_source_also_no_gain' if r['normalized_mae']['patient']['immediate'] >= r['normalized_mae']['patient']['unbridged']
            else 'same_source_mean_gain')
    summary = dict(occurrences=len(frame), patients=int(frame.patient.nunique()),
        epoch10_exposed_occurrences=int(frame.epoch10_exposed.sum()),
        scored_targets=sum(r['status'] == 'scored' for r in results.values()),
        prior_reversals=diagnosis, counts={s: list(diagnosis.values()).count(s) for s in
            ('unsupported', 'same_source_also_no_gain', 'same_source_mean_gain')})
    write_json(output, dict(summary=summary, results=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Adaptation-unseen, mostly encoder-exposed, fit-scaler-exposed patients. '
        'Descriptive three-role comparison, not causal source-shift attribution or new lockbox.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
