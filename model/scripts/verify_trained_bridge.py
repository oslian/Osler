"""Independent scalar aggregation and preserved-input checks for E33."""
from collections import defaultdict
import json
from math import fsum, isfinite
from pathlib import Path

import pandas as pd

from scripts.audit_trained_bridge import DEST, OUTPUT, ROLES, check_bindings
from scripts.organ_assimilation import CHANNELS
from scripts.probe_assimilation_transfer import DEST as TRANSFER
from scripts.train_organ_assimilation import DEST as TRAINED, validate_identities
from scripts.train_organ_binding import write_json
from whole_body_jepa.multisource_intervals import sha256


def mean(xs):
    return fsum(xs)/len(xs)


def verify_results(frame, saved):
    validate_identities(frame)
    names = ['unbridged', 'frozen', *CHANNELS, *[c+'_off' for c in CHANNELS]]
    if 'raw' in frame: names.append('raw')
    by_target = defaultdict(list)
    for row in frame.to_dict('records'):
        if row['variable'] not in saved or not all(isfinite(row[c]) for c in ['truth', *names]):
            raise ValueError('Invalid saved prediction')
        by_target[row['variable']].append(row)
    max_difference = 0.
    for v, report in saved.items():
        rows = by_target[v]
        if not rows:
            if report != dict(status='unsupported'): raise ValueError('Unsupported target differs')
            continue
        people, link = defaultdict(list), {}
        for r in rows:
            people[r['patient']].append(r)
            link[r['patient']] = r['component']
        patients = {p: {c: mean([abs(r[c]-r['truth']) for r in rs]) for c in names}
                    for p, rs in people.items()}
        components = defaultdict(list)
        for p, losses in patients.items(): components[link[p]].append(losses)
        groups = {k: {c: mean([loss[c] for loss in ls]) for c in names} for k, ls in components.items()}
        support = dict(status='scored', patients=len(patients), occurrences=len(rows), components=len(groups))
        if any(report[k] != value for k, value in support.items()):
            raise ValueError('Support mismatch')
        for scope, values in (('patient', patients), ('component', groups)):
            expected = {c: mean([loss[c] for loss in values.values()]) for c in names}
            if set(report['normalized_mae'][scope]) != set(expected): raise ValueError('Model inventory mismatch')
            pairs = [(report['normalized_mae'][scope][c], x) for c, x in expected.items()]
            if set(report['contrasts']) != set(CHANNELS): raise ValueError('Channel inventory mismatch')
            for c in CHANNELS:
                terms = dict(bridge=mean([r[c]-r[c+'_off'] for r in values.values()]),
                    remaining=mean([r[c+'_off']-r['unbridged'] for r in values.values()]),
                    total=mean([r[c]-r['unbridged'] for r in values.values()]))
                if set(report['contrasts'][c][scope]) != set(terms): raise ValueError('Term inventory mismatch')
                pairs += [(report['contrasts'][c][scope][k], x) for k, x in terms.items()]
            for actual, expected_value in pairs:
                error = abs(actual-expected_value)
                if not isfinite(actual) or error > 1e-12*(1+abs(expected_value)):
                    raise ValueError('Aggregated error mismatch')
                max_difference = max(max_difference, error)
    return max_difference


def main():
    output = Path('artifacts/native_trained_bridge_ablation_verification_20261005.json')
    if output.exists(): raise ValueError('Completed verification exists')
    report = json.loads(OUTPUT.read_text())
    check_bindings(report['bindings'])
    parent = json.loads(Path('artifacts/native_organ_assimilation_assessment_20261004.json').read_text())
    if set(report['results']) != set(ROLES) or any(set(r) != set(parent['results']) for r in report['results'].values()):
        raise ValueError('Role/target inventory changed')
    bindings = {str(p): sha256(p) for p in (OUTPUT, Path(__file__), Path('tests/test_trained_bridge_verification.py'))}
    all_frames = {r: [] for r in ROLES}
    maximum = 0.
    for fold in range(5):
        mp = Path(f'artifacts/native_trained_bridge_ablation_fold{fold}_20261005.json')
        meta = json.loads(mp.read_text())
        check_bindings(meta['bindings']); check_bindings(meta['outputs'])
        if set(meta['results']) != set(ROLES) or any(set(r) != set(parent['results']) for r in meta['results'].values()):
            raise ValueError('Fold inventory changed')
        for role in ROLES:
            path = DEST/f'fold{fold}'/f'{role}.parquet'
            frame = pd.read_parquet(path)
            original_path = (TRANSFER/f'fold{fold}'/'predictions.parquet' if role == 'same_source'
                             else TRAINED/f'fold{fold}'/f'{role}_predictions.parquet')
            original = pd.read_parquet(original_path)
            pd.testing.assert_frame_equal(frame[list(original)], original, check_exact=True)
            maximum = max(maximum, verify_results(frame, meta['results'][role]))
            if any(x > 1e-7 for d in meta['parity'][role].values() for x in d.values()):
                raise ValueError('Replay not verified')
            all_frames[role].append(frame)
            bindings[str(path)] = sha256(path)
        bindings[str(mp)] = sha256(mp)
    erosion = {}
    for role, fs in all_frames.items():
        frame = pd.concat(fs, ignore_index=True)
        maximum = max(maximum, verify_results(frame, report['results'][role]))
        if len(frame) != report['occurrences'][role]: raise ValueError('Occurrence mismatch')
        if role == 'fit': continue
        erosion[role] = {}
        for v, r in report['results'][role].items():
            f = report['results']['fit'][v]
            if r['status'] != 'scored' or f['status'] != 'scored':
                erosion[role][v] = dict(status='unsupported')
                continue
            erosion[role][v] = dict(status='scored', contrasts={c: {scope: {
                k: r['contrasts'][c][scope][k]-f['contrasts'][c][scope][k]
                for k in ('bridge', 'remaining', 'total')} for scope in ('patient', 'component')}
                for c in CHANNELS})
    write_json(output, dict(complete=True, bindings=bindings, maximum_aggregation_roundoff=maximum,
        erosion=erosion, runtime_modified=False, promotion_allowed=False,
        limitation='Summary and preserved-prediction verification; inference parity checked during E33 replay. '
        'Erosion is descriptive, not composition-adjusted or causal training attribution.'))
    print('E33 summary verified', maximum, flush=True)


if __name__ == '__main__': main()
