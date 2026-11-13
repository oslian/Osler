"""Conditional nested hospital CV; no encoder independence or promotion claim."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from scripts.refit_anchored_body_readout import STRENGTHS, fit, source_score, source_weights
from whole_body_jepa.multisource_intervals import sha256


def choose(x, y, groups, sites, oldw, oldb, solver=fit):
    """All selection inputs belong to the current outer training fold."""
    scores = {'unchanged': source_score(abs(x @ oldw + oldb - y), groups, sites)}
    folds = list(GroupKFold(n_splits=min(4, len(set(sites)))).split(x, y, sites))
    for strength in STRENGTHS:
        predicted = np.empty(len(y))
        for train, valid in folds:
            weight, bias = solver(x[train], y[train], source_weights(groups[train], sites[train]), oldw, oldb, strength)
            predicted[valid] = x[valid] @ weight + bias
        scores[str(strength)] = source_score(abs(predicted - y), groups, sites)
    return min(scores, key=scores.get), scores


def nested_predictions(x, y, groups, sites, oldw, oldb, solver=fit):
    predicted = np.empty(len(y)); details = []
    for train, valid in GroupKFold(n_splits=min(5, len(set(sites)))).split(x, y, sites):
        name, scores = choose(x[train], y[train], groups[train], sites[train], oldw, oldb, solver)
        if name == 'unchanged':
            weight, bias = oldw, oldb
        else:
            weight, bias = solver(x[train], y[train], source_weights(groups[train], sites[train]), oldw, oldb, float(name))
        predicted[valid] = x[valid] @ weight + bias
        details.append(dict(choice=name, inner_scores=scores, train_rows=len(train), test_rows=len(valid)))
    return predicted, details


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    parent_path = Path('artifacts/native_reconstruction_anchored_readout_20260920.json')
    out = Path('artifacts/native_reconstruction_nested_readout_cv_20260920.json')
    if out.exists():
        raise ValueError('Completed audit exists')
    parent = json.loads(parent_path.read_text())
    for path, digest in parent['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed parent dependency')
    cp = root / 'bounded_site_training/bounded.pt'
    mp = root / 'site_balance_manifest_20260920.pt'
    ck = torch.load(cp, weights_only=False)
    manifest = torch.load(mp, weights_only=False)
    targets = list(parent['choices']); data = {t: [] for t in targets}
    digest = hashlib.sha256()
    for i, shard in enumerate(manifest['shards']):
        path = root / 'frozen_readout_refit_20260920' / f'features_{i:05d}.pt'
        digest.update((path.name + sha256(path)).encode())
        if shard['role'] != 'fit':
            continue
        saved = torch.load(path, weights_only=False)
        if saved['shard'] != shard:
            raise ValueError('Cache identity mismatch')
        for row in saved['rows']:
            data[row['target']].append(dict(row, group=shard['group']))
    spectrum = Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    if digest.hexdigest() != json.loads(spectrum.read_text())['cache_digest']:
        raise ValueError('Changed body cache')
    bindings = {str(p): sha256(p) for p in (Path(__file__), parent_path, cp, mp, spectrum, Path('scripts/refit_anchored_body_readout.py'))}
    private = root / 'nested_readout_cv_20260920'; private.mkdir(exist_ok=True)
    results = {}
    with threadpool_limits(limits=4):
        for target, rows in data.items():
            saved_path = private / f'{target}.json'
            if saved_path.exists():
                saved = json.loads(saved_path.read_text())
                if saved['bindings'] != bindings:
                    raise ValueError('Resume mismatch')
                results[target] = saved['result']; continue
            x = np.stack([r['body'] for r in rows]).astype(float)
            y = np.array([r['truth'] for r in rows])
            groups = np.array([r['group'] for r in rows]); sites = np.array([str(r['site']) for r in rows])
            col = ck['variables'].index(target)
            oldw = ck['best_state_dict']['readout.weight'][col].numpy().astype(float)
            oldb = float(ck['best_state_dict']['readout.bias'][col])
            before = source_score(abs(x @ oldw + oldb - y), groups, sites)
            if not np.isclose(before, parent['choices'][target]['cv_scores']['unchanged'], atol=1e-10, rtol=0):
                raise ValueError('Original score mismatch')
            prediction, folds = nested_predictions(x, y, groups, sites, oldw, oldb)
            after = source_score(abs(prediction - y), groups, sites)
            choice = parent['choices'][target]
            result = dict(before=before, nested_score=after, nested_delta=after-before,
                          ordinary_selected_delta=choice['cv_scores'][choice['name']]-before, folds=folds)
            temp = saved_path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(bindings=bindings, result=result), indent=2)+'\n'); temp.replace(saved_path)
            results[target] = result
            print(target, result['ordinary_selected_delta'], result['nested_delta'], flush=True)
    summary = dict(targets=len(results), ordinary_delta=float(np.mean([v['ordinary_selected_delta'] for v in results.values()])),
                   nested_delta=float(np.mean([v['nested_delta'] for v in results.values()])),
                   nested_improving=sum(v['nested_delta']<0 for v in results.values()),
                   nested_worsening=sum(v['nested_delta']>0 for v in results.values()))
    out.write_text(json.dumps(dict(targets=results, summary=summary, bindings=bindings,
        cache_digest=digest.hexdigest(), runtime_modified=False, promotion_allowed=False,
        limitation='Nested readout selection only. Encoder and original anchor already saw these hospitals; conditional descriptive audit, not end-to-end heldout validation. Outer folds overlap in training. No new model selected from outer results.'), indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
