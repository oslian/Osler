"""Replay the frozen MAP sampler and summarize source-tune curves only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def replay(pools, seed, epochs=32, steps=16, batch_size=32):
    rng = np.random.default_rng(seed)
    counts = {}
    snapshots = []
    for _ in range(epochs):
        for _ in range(steps):
            for i in rng.integers(len(pools), size=batch_size):
                row = int(rng.choice(pools[i]))
                counts[row] = counts.get(row, 0) + 1
        snapshots.append(dict(counts))
    return snapshots


def summarize(counts, pools):
    draws = np.array([sum(counts.get(int(row), 0) for row in pool) for pool in pools])
    return dict(row_coverage=len(counts) / sum(map(len, pools)),
                patient_coverage=float(np.mean(draws > 0)),
                median_draws_per_patient=float(np.median(draws)),
                total_draws=int(draws.sum()))


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    query_path = root / 'queries.parquet'
    label_path = root / 'map_all_query_labels.npz'
    q = pd.read_parquet(query_path)
    eligible = np.load(label_path)['paired'][:, 0]
    paths = [Path(__file__), query_path, label_path]
    results = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        report_path = Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        report = json.loads(report_path.read_text())
        assert report['complete']
        for path, digest in report['bindings'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        partition = root / f'partition_{seed}.npz'
        fit = np.load(partition)['fit'] & eligible
        pools = [np.flatnonzero(fit & q.uniquepid.eq(s).to_numpy())
                 for s in sorted(set(q.loc[fit, 'uniquepid']))]
        assert len(pools) == report['fit_patients']
        assert sum(map(len, pools)) == report['fit_rows']
        snapshots = replay(pools, seed)
        arms = {}
        for arm, record in report['arms'].items():
            curve = np.array([entry['tune_mae'] for entry in record['log']])
            selected = record['selected_epoch']
            assert selected and selected == int(np.argmin(curve)) + 1
            arms[arm] = dict(selected_epoch=selected,
                exposure_at_selection=summarize(snapshots[selected - 1], pools),
                best_first16=float(curve[:16].min()),
                best_last16=float(curve[16:].min()),
                last8_minus_first8=float(curve[-8:].mean() - curve[:8].mean()))
        own = np.array([x['tune_mae'] for x in report['arms']['own']['log']])
        full = np.array([x['tune_mae'] for x in report['arms']['full']['log']])
        results.append(dict(seed=seed, arms=arms,
            exposure_at32=summarize(snapshots[-1], pools),
            full_better_same_epoch_count=int(np.sum(full < own)),
            best_full_minus_best_own=float(full.min() - own.min())))
        paths.extend([report_path, partition])
    output = dict(status='diagnostic_only', runtime_modified=False, results=results,
        limitations=['No training-set prediction errors measured.',
                     'Tune curve comparisons are descriptive, not promotion tests.',
                     'Row coverage is not unique measurement-event coverage.',
                     'Low coverage does not prove additional compute improves accuracy.'],
        bindings={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    Path('artifacts/map_expanded_exposure_20260912.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
