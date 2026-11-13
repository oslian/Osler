"""Distinguish sampled query rows from unsampled rows of fitting patients."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_map_expanded_exposure import replay
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    q = pd.read_parquet(root / 'queries.parquet').rename(columns={'uniquepid': 'subject_id'})
    labels = np.load(root / 'map_all_query_labels.npz')
    q['truth'] = labels['future'][:, 0]
    results = []
    paths = [Path(__file__), Path('scripts/audit_map_expanded_exposure.py')]
    for seed in (7, 19, 31, 43, 59, 71, 83):
        parent = Path(f'artifacts/map_expanded_fit_gap_{seed}_20260912.json')
        report = json.loads(parent.read_text())
        for path, digest in report['bindings'].items():
            assert sha256(Path(path)) == digest
        original_path = Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        original = json.loads(original_path.read_text())
        fit = np.load(root / f'partition_{seed}.npz')['fit'] & labels['paired'][:, 0]
        pools = [np.flatnonzero(fit & q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit, 'subject_id']))]
        snapshots = replay(pools, seed)
        seen = {}
        for arm in ('own', 'full'):
            seen[arm] = np.zeros(len(q), bool)
            seen[arm][list(snapshots[original['arms'][arm]['selected_epoch'] - 1])] = True
        predictions = dict(np.load(root / f'map_expanded_fit_gap_predictions_{seed}.npz'))
        groups = dict(sampled_both=fit & seen['own'] & seen['full'],
                      sampled_neither=fit & ~seen['own'] & ~seen['full'])
        scores = {}
        for name, mask in groups.items():
            f = q.loc[mask]
            p = {k: a[mask] for k, a in predictions.items()}
            scores[name] = dict(rows=int(mask.sum()), patients=f.subject_id.nunique(),
                mae={k: patient_equal_mae(f, f.truth, a) for k, a in p.items()},
                comparison=paired_model_errors(f, p, 'own', seed=20260912, repetitions=20000)['full'])
        results.append(dict(seed=seed, groups=scores))
        paths.extend([parent, original_path])
    Path('artifacts/map_sampled_fit_20260912.json').write_text(json.dumps(dict(
        results=results, training_performed=False, runtime_modified=False,
        caveat='Unsampled query rows may share outcomes/history with sampled rows; not held-out patients.',
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for group in ('sampled_both', 'sampled_neither'):
        cs = [r['groups'][group]['comparison'] for r in results]
        print(group, 'gain', sum(c['ci95'][1] < 0 for c in cs),
              'harm', sum(c['ci95'][0] > 0 for c in cs),
              'mean_deltas', [round(c['mean'], 5) for c in cs])


if __name__ == '__main__':
    main()
