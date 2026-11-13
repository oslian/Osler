"""Independently check E34 aggregates, decisions and saved frozen weights."""
from collections import defaultdict
import json
from math import fsum, isfinite
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import torch

from scripts.audit_trained_bridge import check_bindings
from scripts.organ_assimilation import CHANNELS
from scripts.train_frozen_bridge import CONTROLS, DEST, JOINT, NEW, OUTPUT, ROLES, SEED, TRANSFER
from scripts.train_organ_assimilation import validate_identities
from scripts.train_organ_binding import prepare, write_json
from whole_body_jepa.multisource_intervals import sha256


def mean(xs):
    return fsum(xs)/len(xs)


def compare(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError('Different report fields')
        return max((compare(actual[k], v) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Different report sequence')
        return max((compare(a, e) for a, e in zip(actual, expected)), default=0.)
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not isfinite(actual):
            raise ValueError('Nonfinite report')
        error = abs(actual-expected)
        if error > 1e-12*(1+abs(expected)):
            raise ValueError('Report numeric mismatch')
        return error
    if actual != expected or type(actual) is not type(expected):
        raise ValueError('Report decision mismatch')
    return 0.


def paired(values, seed, repetitions):
    x = np.asarray(values, dtype=float)
    result = dict(n=len(x), mean=float(x.mean()), p_less=None, ci95=None)
    if len(x) < 2:
        return result
    p = (0. if x.mean() < 0 else 1.) if x.min() == x.max() else float(
        ttest_1samp(x, 0., alternative='less').pvalue)
    rng = np.random.default_rng(seed)
    boot = []
    for offset in range(0, repetitions, 50):
        sampled = rng.integers(0, len(x), size=(min(50, repetitions-offset), len(x)))
        boot.extend(np.mean(x[sampled], axis=1))
    result.update(p_less=p, ci95=np.quantile(boot, [.025, .975]).tolist())
    return result


def independent_results(frame, inventory, selection=False, repetitions=5000):
    validate_identities(frame)
    names = [*NEW, 'bound_start', *CHANNELS, 'unbridged', 'frozen']
    if selection or 'raw' in frame:
        names.append('raw')
    if not set(names) <= set(frame):
        raise ValueError('Missing model columns')
    by_target = defaultdict(list)
    for r in frame.to_dict('records'):
        if r['variable'] not in inventory or not all(isfinite(r[k]) for k in ['truth', *names]):
            raise ValueError('Invalid prediction')
        by_target[r['variable']].append(r)
    result = {}
    for index, v in enumerate(inventory):
        rows = by_target[v]
        if not rows:
            result[v] = dict(status='unsupported')
            continue
        people, links = defaultdict(list), {}
        for r in rows:
            people[r['patient']].append(r)
            links[r['patient']] = r['component']
        losses = {p: {n: mean([abs(r[n]-r['truth']) for r in people[p]]) for n in names}
                  for p in sorted(people)}
        components = defaultdict(list)
        for p, loss in losses.items():
            components[links[p]].append(loss)
        groups = {c: {n: mean([r[n] for r in components[c]]) for n in names}
                  for c in sorted(components)}
        entry = dict(status='scored', patients=len(losses), components=len(groups), occurrences=len(rows),
                     normalized_mae={s: {n: mean([r[n] for r in ls.values()]) for n in names}
                                     for s, ls in (('patient', losses), ('component', groups))})
        if selection:
            comparisons = {}
            for j, control in enumerate(CONTROLS):
                stats = {scope: paired([r['fixed_immediate']-r[control] for r in ls.values()],
                                       SEED+index*1000+j+offset, repetitions)
                         for scope, ls, offset in (('patient', losses, 0), ('component', groups, 100))}
                passed = len(losses) >= 30 and len(groups) >= 3 and all(
                    s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
                    and s['p_less'] < .05/(53*7) for s in stats.values())
                comparisons[control] = dict(paired=stats, screen_pass=bool(passed))
            entry['comparisons'] = comparisons
        result[v] = entry
    return result


def diagnose(results):
    output = {}
    for role, targets in results.items():
        scored = {v: r for v, r in targets.items() if r['status'] == 'scored'}
        names = next(iter(scored.values()))['normalized_mae']['patient']
        output[role] = dict(targets=len(scored), macro={
            scope: {n: mean([r['normalized_mae'][scope][n] for r in scored.values()]) for n in names}
            for scope in ('patient', 'component')}, contrasts={})
        for control in ('bound_start', 'immediate', 'unbridged', 'frozen'):
            per_target = {}
            for v, r in scored.items():
                m = r['normalized_mae']['patient']
                f = results['fit'][v]
                fitted = (f['normalized_mae']['patient']['fixed_immediate']
                          - f['normalized_mae']['patient'][control]) if f['status'] == 'scored' else None
                per_target[v] = dict(query=m['fixed_immediate']-m[control], fit=fitted)
            output[role]['contrasts'][control] = dict(
                mean_gain=sum(r['query'] < 0 for r in per_target.values()),
                fit_gain_query_non_gain=sum(r['fit'] is not None and r['fit'] < 0 and r['query'] >= 0
                                           for r in per_target.values()), targets=per_target)
    return output


def main():
    vp = Path('artifacts/native_frozen_bridge_verification_20261005.json')
    if vp.exists():
        raise ValueError('Completed verification exists')
    torch.set_num_threads(4)
    report = json.loads(OUTPUT.read_text())
    check_bindings(report['bindings'])
    if set(report['results']) != set(ROLES):
        raise ValueError('Changed role inventory')
    bindings = {str(p): sha256(p) for p in (OUTPUT, Path(__file__), Path('tests/test_frozen_bridge_verification.py'))}
    data = {r: [] for r in ROLES}
    inventory = None
    for fold in range(5):
        variables, _, _, original, parent, extraction, _ = prepare(fold)
        if inventory is not None and inventory != variables:
            raise ValueError('Inconsistent target order')
        inventory = variables
        mp = Path(f'artifacts/native_frozen_bridge_fold{fold}_20261005.json')
        meta = json.loads(mp.read_text())
        contract = json.loads((DEST/f'fold{fold}'/'contract.json').read_text())
        if meta['contract'] != contract or contract['parent'] != parent or contract['extraction'] != extraction:
            raise ValueError('Changed initialization/input contract')
        check_bindings(contract['bindings']); check_bindings(meta['outputs'])
        if set(meta['fits']) != set(CHANNELS) or set(meta['parity']) != set(ROLES):
            raise ValueError('Incomplete checkpoint/replay inventory')
        for c in CHANNELS:
            path = DEST/f'fold{fold}'/f'{c}.pt'
            info = json.loads(path.with_suffix('.json').read_text())
            if (info != meta['fits'][c] or sha256(path) != info['sha256']
                    or info['contract_sha256'] != sha256(DEST/f'fold{fold}'/'contract.json')):
                raise ValueError('Changed checkpoint')
            saved = torch.load(path, weights_only=False)
            for k, value in original.state_dict().items():
                torch.testing.assert_close(saved['model.'+k], value, rtol=0, atol=0)
            if not all(torch.isfinite(t).all() for t in saved.values()):
                raise ValueError('Nonfinite saved weights')
            if [r['steps'] for r in info['training']] != [2048, 2048]:
                raise ValueError('Different training budget')
            bindings[str(path)] = sha256(path)
        for role in ROLES:
            if (set(meta['parity'][role]) != set(CHANNELS)
                    or any(not isfinite(x) or x > 1e-7 for x in meta['parity'][role].values())):
                raise ValueError('Replay not verified')
            path = DEST/f'fold{fold}'/f'{role}.parquet'
            frame = pd.read_parquet(path)
            prior = (TRANSFER/f'fold{fold}'/'predictions.parquet' if role == 'same_source'
                     else JOINT/f'fold{fold}'/f'{role}_predictions.parquet')
            original_frame = pd.read_parquet(prior)
            pd.testing.assert_frame_equal(frame[list(original_frame)], original_frame, check_exact=True)
            if len(frame) != (2048 if role == 'fit' else 1024):
                raise ValueError('Changed row inventory')
            data[role].append(frame); bindings[str(path)] = sha256(path)
        bindings[str(mp)] = sha256(mp)
        print('E34 weights and controls verified', fold, flush=True)
    frames = {r: pd.concat(fs, ignore_index=True) for r, fs in data.items()}
    actual = {r: independent_results(f, inventory, selection=r == 'selection') for r, f in frames.items()}
    maximum = compare(report['results'], actual)
    screens = {c: [v for v, r in actual['selection'].items()
                   if r['status'] == 'scored' and r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    summary = dict(occurrences={r: len(f) for r, f in frames.items()}, screens=screens,
                   full_screen=sorted(set.intersection(*(set(v) for v in screens.values()))))
    compare(report['summary'], summary)
    check_bindings(report['bindings'])
    write_json(vp, dict(complete=True, bindings=bindings, maximum_aggregation_roundoff=maximum,
                       diagnosis=diagnose(actual), summary=summary, runtime_modified=False, promotion_allowed=False,
                       limitation='Independent scalar aggregates/paired statistics and saved frozen-weight checks; '
                       'all-row explicit neural schedule parity is checked inside the training audit, not rerun here. '
                       'Reused development and descriptive reversals are not new external or causal evidence.'))
    print('E34 independently verified', maximum, summary, flush=True)


if __name__ == '__main__':
    main()
