"""Recheck completed MAP predictions using common patient bootstrap draws."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/native_bedside_map_training_20260911.json')
    output = Path('artifacts/native_map_common_bootstrap_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    report = json.loads(parent.read_text())
    assert report['complete'] and report['target'] == 'map'
    for path, digest in report['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'changed dependency: {path}')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    labels = dict(np.load(root / 'bedside_future_labels.npz'))
    col = labels['variables'].tolist().index('map')
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'})
    q['truth'] = labels['future'][:, col]
    results = []
    for original in report['seeds']:
        seed = original['seed']
        points = dict(np.load(root / f'bedside_map_predictions_{seed}.npz'))
        np.testing.assert_array_equal(points['persistence'], labels['current'][:, col])
        roles = dict(np.load(root / f'partition_{seed}.npz'))
        item = {'seed': seed, 'roles': {}}
        for role in ('evaluation', 'hospital'):
            mask = roles[role][valid] & labels['paired'][:, col]
            frame = q.loc[mask]
            assert frame.subject_id.nunique() == original['roles'][role]['patients']
            predictions = {name: values[mask] for name, values in points.items()}
            item['roles'][role] = {
                str(draw_seed): {
                    ref: paired_model_errors(frame, predictions, ref,
                                             seed=draw_seed, repetitions=20000)
                    for ref in ('own', 'support', 'persistence')}
                for draw_seed in (20260912, 20260913, 20260914)}
        results.append(item)
    output.write_text(json.dumps({
        'complete': True, 'status': 'candidate_only', 'runtime_modified': False,
        'training_performed': False, 'bootstrap_draws': 20000, 'seeds': results,
        'verified_parent_bindings': len(report['bindings']),
        'bindings': {str(p): sha256(p) for p in (
            parent, Path(__file__), Path('whole_body_jepa/change_evaluation.py'),
            root / 'labels.npz', root / 'queries.parquet', root / 'bedside_future_labels.npz')},
    }, indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        for ref, model in (('own', 'bedside'), ('support', 'bedside'),
                           ('persistence', 'own'), ('persistence', 'bedside')):
            print(role, model, 'vs', ref, [
                (d, sum(s['roles'][role][str(d)][ref][model]['ci95'][1] < 0 for s in results),
                 sum(s['roles'][role][str(d)][ref][model]['ci95'][0] > 0 for s in results))
                for d in (20260912, 20260913, 20260914)])


if __name__ == '__main__':
    main()
