"""Independent E31 role selection, explicit-schedule replay and mean checks."""
from collections import defaultdict
from copy import deepcopy
import json
from math import fsum
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch

from scripts.audit_organ_assignment import pools
from scripts.encoder_fold_scaling import transform_example
from scripts.probe_assimilation_transfer import ARMS, DEST, ROOT
from scripts.probe_observed_content import digest
from scripts.train_organ_assimilation import AssimilatedReconstruction, DEST as PARENT
from scripts.train_organ_binding import prepare, write_json
from scripts.verify_organ_assimilation import direct_prediction
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def verify_summary(data, result, parent):
    if set(result['results']) != set(parent['results']):
        raise ValueError('Inventory changed')
    identity = ['patient', 'component', 'variable', 'fold', 'context']
    if data[identity].isna().any().any() or data.duplicated(['fold', 'patient']).any():
        raise ValueError('Invalid query identities')
    if data.groupby('patient').component.nunique().max() != 1:
        raise ValueError('Patient changed component')
    if not set(data.variable) <= set(result['results']):
        raise ValueError('Unknown target')
    if not np.isfinite(data[list(ARMS) + ['truth']].to_numpy()).all():
        raise ValueError('Nonfinite prediction')
    diagnosis = {}
    for variable, old in parent['results'].items():
        rows = data[data.variable == variable]
        saved = result['results'][variable]
        if rows.empty:
            if saved != {'status': 'unsupported'}:
                raise ValueError('Unsupported target changed')
        else:
            expected = dict(status='scored', patients=int(rows.patient.nunique()),
                occurrences=len(rows), components=int(rows.component.nunique()),
                epoch10_exposed_occurrences=int(rows.epoch10_exposed.sum()))
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError('Target support changed')
        if old['status'] != 'scored':
            continue
        if (old['fit_patient_mae']['immediate'] < old['fit_patient_mae']['unbridged']
                and old['normalized_mae']['patient']['immediate'] >= old['normalized_mae']['patient']['unbridged']):
            if rows.empty:
                diagnosis[variable] = 'unsupported'
            else:
                gains = defaultdict(list)
                for r in rows.itertuples(index=False):
                    gains[r.patient].append(abs(r.immediate-r.truth)-abs(r.unbridged-r.truth))
                gain = fsum(fsum(v)/len(v) for v in gains.values())/len(gains)
                diagnosis[variable] = 'same_source_also_no_gain' if gain >= 0 else 'same_source_mean_gain'
    expected = dict(occurrences=len(data), patients=int(data.patient.nunique()),
        epoch10_exposed_occurrences=int(data.epoch10_exposed.sum()),
        scored_targets=int(data.variable.nunique()), prior_reversals=diagnosis,
        counts={s: list(diagnosis.values()).count(s) for s in
                ('unsupported', 'same_source_also_no_gain', 'same_source_mean_gain')})
    if result['summary'] != expected:
        raise ValueError('Summary or reversal classification changed')


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    ap = Path('artifacts/native_assimilation_transfer_20261005.json')
    output = Path('artifacts/native_assimilation_transfer_verification_20261005.json')
    if output.exists():
        raise ValueError('Completed verification exists')
    result = json.loads(ap.read_text())
    bindings = {str(p): sha256(p) for p in (ap, Path(__file__), Path('scripts/verify_organ_assimilation.py'))}
    frames, errors = [], {}
    for p, h in result['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed assessment dependency')
    scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)
    roles = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds']
    provenance = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if provenance.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    component = dict(provenance.itertuples(index=False, name=None))
    for fold in range(5):
        rp = Path(f'artifacts/native_assimilation_transfer_fold{fold}_20261005.json')
        report = json.loads(rp.read_text())
        for p, h in report['bindings'].items():
            if sha256(Path(p)) != h:
                raise ValueError('Changed inference dependency')
        pp = DEST/f'fold{fold}'/'predictions.parquet'
        if sha256(pp) != report['prediction_sha256']:
            raise ValueError('Changed private predictions')
        frame = pd.read_parquet(pp)
        variables, old, _, original, _, _, old_dest = prepare(fold)
        adapted = {e['patient'] for e in old['fit']}
        fit_components = {e['component'] for e in old['fit']}
        by_patient = defaultdict(list)
        cp = ROOT/f'nonlinear_probe_20261001/fold{fold}_fit.sqlite'
        db = sqlite3.connect(cp.resolve().as_uri()+'?mode=ro', uri=True)
        try:
            for patient, c in db.execute('SELECT patient,target FROM features'):
                if roles[fold][patient] != 'fit':
                    raise ValueError('Wrong original role')
                if patient not in adapted and component[patient] in fit_components:
                    by_patient[patient].append(c)
        finally:
            db.close()
        people = sorted(by_patient, key=lambda p: (digest(str(p)), str(p)))[:1024]
        expected = [(p, min(by_patient[p], key=lambda c: (digest(str(p)+'/'+str(c)), c))) for p in people]
        if list(zip(frame.patient, frame.context)) != expected:
            raise ValueError('Third-role selection mismatch')
        ck = torch.load(ROOT/f'monitored_continuation_20260929/fold{0 if fold == 2 else fold}_real_epoch10.pt', weights_only=False)
        exposed = {scaling['people'][int(i)] for i, _ in ck['seen']}
        if frame.epoch10_exposed.tolist() != [p in exposed for p in frame.patient]:
            raise ValueError('Exposure mismatch')
        candidates = {}
        for arm in ARMS[:3]:
            candidate = AssimilatedReconstruction(original, arm).eval()
            candidate.load_state_dict(torch.load(PARENT/f'fold{fold}'/f'{arm}.pt', weights_only=False), strict=True)
            candidates[arm] = candidate
        unbridged = deepcopy(original).eval()
        unbridged.load_state_dict(torch.load(old_dest/'binding.pt', weights_only=False), strict=True)
        maximum = {a: 0. for a in ARMS}
        source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
        try:
            with torch.inference_mode():
                for row in frame.itertuples(index=False):
                    if row.component != component[row.patient] or row.fold != fold or row.variable != variables[row.context]:
                        raise ValueError('Prediction provenance mismatch')
                    e = transform_example(source.get(row.patient, row.context), scaling['scalers'][fold])
                    if row.truth != e['truth_normalized']:
                        raise ValueError('Truth changed')
                    clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, row.variable)
                    body, organs, presence = encode_native_organ_observations(original.trajectory.jepa, variables, clean)
                    _, bound = pools(original.trajectory.jepa, variables, clean)
                    example = dict(body=body, organs=organs, presence=presence, times=times,
                                   anchor=float(e['anchor_hour']), target=row.context)
                    actual = {a: direct_prediction(m, example, bound) for a, m in candidates.items()}
                    # Native API replay, independent of the cached replay helper.
                    original_propose = unbridged.trajectory.layer.update
                    cursor = [0]
                    def bound_update(state, b, o, mask, delta):
                        i = cursor[0]; cursor[0] += 1
                        return original_propose(state, b, bound[i:i+1], mask, delta)
                    unbridged.trajectory.layer.update = bound_update
                    try:
                        actual['unbridged'] = float(unbridged(e['tokens'], e['event_hours'], e['anchor_hour'], row.variable)[0])
                    finally:
                        unbridged.trajectory.layer.update = original_propose
                    if cursor[0] != len(times):
                        raise ValueError('Native update count differs')
                    actual['frozen'] = float(original(e['tokens'], e['event_hours'], e['anchor_hour'], row.variable)[0])
                    for a in ARMS:
                        np.testing.assert_allclose(actual[a], getattr(row, a), atol=1e-7, rtol=0)
                        maximum[a] = max(maximum[a], abs(actual[a]-getattr(row, a)))
        finally:
            source.close()
        errors[str(fold)] = maximum
        frames.append(frame)
        bindings.update({str(p): sha256(p) for p in (rp, pp)})
        print('E31 verified', fold, maximum, flush=True)
    data = pd.concat(frames, ignore_index=True)
    parent_path = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    if sha256(parent_path) != result['bindings'][str(parent_path)]:
        raise ValueError('Parent changed')
    verify_summary(data, result, json.loads(parent_path.read_text()))
    mean_roundoff = 0.
    for variable, saved in result['results'].items():
        selected = data[data.variable == variable]
        if selected.empty:
            if saved['status'] != 'unsupported':
                raise ValueError('Unsupported status mismatch')
            continue
        for arm in ARMS:
            per_patient, links = defaultdict(list), {}
            for row in selected.itertuples(index=False):
                per_patient[row.patient].append(abs(getattr(row, arm)-row.truth))
                links[row.patient] = row.component
            patient_means = {p: fsum(v)/len(v) for p, v in per_patient.items()}
            groups = defaultdict(list)
            for p, value in patient_means.items():
                groups[links[p]].append(value)
            actual = dict(patient=fsum(patient_means.values())/len(patient_means),
                component=fsum(fsum(v)/len(v) for v in groups.values())/len(groups))
            for level, value in actual.items():
                old = saved['normalized_mae'][level][arm]
                np.testing.assert_allclose(value, old, atol=1e-12, rtol=1e-12)
                mean_roundoff = max(mean_roundoff, abs(value-old))
    write_json(output, dict(complete=True, bindings=bindings, errors=errors,
        mean_roundoff=mean_roundoff, occurrences=len(data), runtime_modified=False,
        promotion_allowed=False, summary_verified=True,
        limitation='Independent selection/replay/mean arithmetic/summary; shared native primitives.'))


if __name__ == '__main__':
    main()
