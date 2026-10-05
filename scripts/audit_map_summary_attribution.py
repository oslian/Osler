"""Score existing summary controls without fitting or outcome-based routing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/map_event_history_20260912.json')
    out = Path('artifacts/map_summary_attribution_20260912.json')
    if out.exists(): raise ValueError('fresh output required')
    for p, h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p)) != h: raise ValueError('parent dependency changed')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    labels = dict(np.load(root / 'bedside_future_labels.npz'))
    col = labels['variables'].tolist().index('map')
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'}); q['truth'] = labels['future'][:, col]
    items = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        p = dict(np.load(root / 'map_event_history' / f'predictions_{seed}.npz'))
        roles = dict(np.load(root / f'partition_{seed}.npz')); item = dict(seed=seed, roles={})
        for role in ('evaluation', 'hospital'):
            mask = roles[role][valid] & labels['paired'][:, col]
            points = {k: v[mask] for k, v in p.items()}
            item['roles'][role] = {ref: paired_model_errors(q.loc[mask], points, ref,
                seed=20260912, repetitions=20000) for ref in ('lag_full', 'own')}
        items.append(item)
    out.write_text(json.dumps(dict(complete=True, runtime_modified=False, fitting_performed=False,
        seeds=items, bindings={str(p): sha256(p) for p in (parent, Path(__file__),
            root / 'labels.npz', root / 'queries.parquet', root / 'bedside_future_labels.npz',
            Path('whole_body_jepa/change_evaluation.py'))}), indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        for ref in ('lag_full', 'own'):
            scores = [s['roles'][role][ref]['support'] for s in items]
            print(role, 'support vs', ref, 'median delta', np.median([s['mean'] for s in scores]),
                  'gain/harm', sum(s['ci95'][1] < 0 for s in scores), sum(s['ci95'][0] > 0 for s in scores))


if __name__ == '__main__': main()
