"""Check E38 saved models, unchanged controls and independent scalar statistics."""
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_pair_gradient import (ARMS, CONTROLS, DEST, OUTPUT, ROOT, ROLES, SEED,
    BASE_DIR, prepare, select_arm, AssimilatedReconstruction, assert_frozen,
    same_source_examples, direct_prediction, validate_identities, check_bindings, write_json)
from scripts.verify_frozen_bridge import mean, paired, compare
from whole_body_jepa.multisource_intervals import sha256

VERIFICATION = Path('artifacts/native_pair_gradient_verification_20261005.json')


def scalar_results(frame, inventory, selection=False):
    validate_identities(frame)
    names = [n for n in ('pcgrad', *CONTROLS) if n in frame]
    if not set(ARMS) <= set(names) or (selection and not set(CONTROLS) <= set(names)):
        raise ValueError('Missing candidate/control')
    targets = defaultdict(list)
    for r in frame.to_dict('records'):
        if r['variable'] not in inventory or not all(isfinite(r[n]) for n in ['truth', *names]):
            raise ValueError('Invalid predictions/inventory')
        targets[r['variable']].append(r)
    results = {}
    for i, variable in enumerate(inventory):
        rows = targets[variable]
        if not rows:
            results[variable] = dict(status='unsupported')
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
            entry['comparisons'] = {}
            for j, control in enumerate(CONTROLS):
                stats = {scope: paired([r['pcgrad']-r[control] for r in ls.values()],
                                      SEED+i*1000+j+offset, 5000)
                         for scope, ls, offset in (('patient', losses, 0), ('component', groups, 100))}
                passed = len(losses) >= 30 and len(groups) >= 3 and all(
                    r['ci95'] is not None and r['ci95'][1] < 0 and r['p_less'] is not None
                    and r['p_less'] < .05/(53*9) for r in stats.values())
                entry['comparisons'][control] = dict(paired=stats, screen_pass=bool(passed))
        results[variable] = entry
    return results


def training_summary(history, arm):
    if len(history) != 2 or [r['epoch'] for r in history] != [1, 2]:
        raise ValueError('Incomplete fit')
    for r in history:
        if r['steps'] != 1024 or r['examples'] != 2048 or not isfinite(r['loss']):
            raise ValueError('Changed fit budget/loss')
        if not 0 <= r['projected_pairs'] <= r['conflicting_pairs'] <= r['steps']:
            raise ValueError('Invalid projection counts')
        if arm == 'mean' and r['projected_pairs'] != 0:
            raise ValueError('Control performed surgery')
        for prefix in ('actual', 'proposed'):
            if not 0 <= r[prefix+'_harmful_dots'] <= r['examples']:
                raise ValueError('Invalid harm count')
            if not isfinite(r['maximum_'+prefix+'_harm']) or r['maximum_'+prefix+'_harm'] < 0:
                raise ValueError('Invalid harm magnitude')
    keys = ('steps', 'examples', 'conflicting_pairs', 'projected_pairs',
            'proposed_harmful_dots', 'actual_harmful_dots')
    return {k: sum(r[k] for r in history) for k in keys}


def diagnose(results):
    out = {}
    for role, targets in results.items():
        scored = {v: r for v, r in targets.items() if r['status'] == 'scored'}
        names = next(iter(scored.values()))['normalized_mae']['patient']
        out[role] = dict(targets=len(scored), macro={
            scope: {n: mean([r['normalized_mae'][scope][n] for r in scored.values()]) for n in names}
            for scope in ('patient', 'component')}, contrasts={})
        for ref in names:
            if ref == 'pcgrad':
                continue
            differences = {}
            for v, r in scored.items():
                f = results['fit'][v]
                fit = None if f['status'] != 'scored' or ref not in f['normalized_mae']['patient'] else (
                    f['normalized_mae']['patient']['pcgrad']-f['normalized_mae']['patient'][ref])
                differences[v] = dict(query=r['normalized_mae']['patient']['pcgrad']-r['normalized_mae']['patient'][ref], fit=fit)
            out[role]['contrasts'][ref] = dict(mean_gain=sum(r['query'] < 0 for r in differences.values()),
                fit_gain_query_non_gain=sum(r['fit'] is not None and r['fit'] < 0 and r['query'] >= 0
                                           for r in differences.values()), targets=differences)
    return out


def main():
    if VERIFICATION.exists():
        raise ValueError('Completed verification exists')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = json.loads(OUTPUT.read_text())
    if not report['complete'] or report['runtime_modified'] or report['promotion_allowed']:
        raise ValueError('Invalid experiment state')
    check_bindings(report['bindings'])
    paths = [OUTPUT, Path(__file__), Path('tests/test_pair_gradient_verification.py'),
             Path('scripts/verify_frozen_bridge.py')]
    data = {r: [] for r in ROLES}
    logs, parity, inventory = {}, {}, None
    for fold in range(5):
        for stage, suffix in (('content_training_20261003', ''), ('organ_binding_20261004', '_binding')):
            for role in ('fit', 'selection'):
                for ext in ('joblib', 'json'):
                    if not (ROOT/stage/f'fold{fold}'/f'{role}{suffix}.{ext}').is_file():
                        raise ValueError('Missing original cache')
        variables, examples, bound, original, parent, extraction, _ = prepare(fold)
        if inventory is not None and variables != inventory:
            raise ValueError('Changed inventory')
        inventory = variables
        dest = DEST/f'fold{fold}'
        cp = dest/'contract.json'
        contract = json.loads(cp.read_text())
        if contract['parent'] != parent or contract['extraction'] != extraction:
            raise ValueError('Changed extraction')
        check_bindings(contract['bindings'])
        models, model_paths = {}, [cp]
        logs[str(fold)], parity[str(fold)] = {}, {}
        for arm in ARMS:
            path = dest/f'{arm}.pt'
            mp = path.with_suffix('.json')
            meta = json.loads(mp.read_text())
            if meta['contract'] != sha256(cp) or meta['sha256'] != sha256(path) or meta['arm'] != arm:
                raise ValueError('Changed checkpoint binding')
            logs[str(fold)][arm] = training_summary(meta['training'], arm)
            m = AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
            m.load_state_dict(torch.load(path, weights_only=False), strict=True)
            assert_frozen(m, original)
            if not all(torch.isfinite(t).all() for t in m.state_dict().values()):
                raise ValueError('Nonfinite checkpoint')
            models[arm] = m
            model_paths.extend((path, mp))
        patients = {}
        for role in ROLES:
            path, mp = dest/f'{role}.parquet', dest/f'{role}.json'
            source = BASE_DIR/f'fold{fold}'/f'{role}.parquet'
            meta = json.loads(mp.read_text())
            expected = {str(p): sha256(p) for p in (source, *model_paths)}
            if meta['bindings'] != expected or meta['sha256'] != sha256(path):
                raise ValueError('Changed role binding')
            f, prior = pd.read_parquet(path), pd.read_parquet(source)
            validate_identities(f)
            pd.testing.assert_frame_equal(f[list(prior)], prior, check_exact=True)
            if len(f) != (2048 if role == 'fit' else 1024) or meta['rows'] != len(f):
                raise ValueError('Changed role size')
            patients[role] = set(f.patient)
            if role == 'same_source':
                rows = same_source_examples(f, original, variables, fold)
            else:
                actual_ids = list(f[['patient', 'component', 'context', 'truth']].itertuples(index=False, name=None))
                if actual_ids != [(e['patient'], e['component'], e['target'], e['truth']) for e in examples[role]]:
                    raise ValueError('Changed identity/truth')
                rows = (select_arm(e, b, 'binding') for e, b in zip(examples[role], bound[role]))
            errors, count = {a: 0. for a in ARMS}, 0
            with torch.inference_mode():
                for i, e in enumerate(rows):
                    for arm, m in models.items():
                        pred = direct_prediction(m, e, e['organs'])
                        expected_point = float(f[arm].iloc[i])
                        if not isfinite(pred) or abs(pred-expected_point) > 1e-7:
                            raise ValueError('Saved prediction replay mismatch')
                        errors[arm] = max(errors[arm], abs(pred-expected_point))
                    count += 1
            if count != len(f):
                raise ValueError('Incomplete replay')
            parity[str(fold)][role] = errors
            data[role].append(f)
            paths.extend((path, mp, source))
            print('E38 verified replay', fold, role, count, errors, flush=True)
        if any(patients[a] & patients[b] for a, b in (('fit', 'selection'), ('fit', 'same_source'), ('selection', 'same_source'))):
            raise ValueError('Role overlap')
        for arm, m in models.items():
            saved = torch.load(dest/f'{arm}.pt', weights_only=False)
            for k, value in m.state_dict().items():
                torch.testing.assert_close(value, saved[k], atol=0, rtol=0)
        paths.extend(model_paths)
    frames = {r: pd.concat(fs, ignore_index=True) for r, fs in data.items()}
    actual = {r: scalar_results(f, inventory, r == 'selection') for r, f in frames.items()}
    error = compare(report['results'], actual)
    screens = {c: [v for v, r in actual['selection'].items()
                   if r['status'] == 'scored' and r['comparisons'][c]['screen_pass']] for c in CONTROLS}
    summary = dict(screens=screens, full_screen=sorted(set.intersection(*(set(x) for x in screens.values()))),
                   occurrences={r: len(f) for r, f in frames.items()})
    compare(report['summary'], summary)
    check_bindings(report['bindings'])
    write_json(VERIFICATION, dict(complete=True, bindings={str(p): sha256(p) for p in paths},
        parity=parity, maximum_aggregation_roundoff=error, training=logs, diagnosis=diagnose(actual),
        summary=summary, runtime_modified=False, promotion_allowed=False,
        limitation='Reused development, current reconstruction; no untouched external/future/causal claim.'))
    print('E38 independently verified', error, summary, flush=True)


if __name__ == '__main__':
    main()
