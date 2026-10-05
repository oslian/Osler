"""Compare selected models under their value training objective, not just MAE."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def smooth_l1(error):
    absolute = np.abs(error)
    return np.where(absolute < 1, .5 * absolute ** 2, absolute - .5)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    assets = load_release(Path('models'))
    scale = float(assets.scaler.scales[assets.checkpoint['variables'].index('map')])
    q = pd.read_parquet(root / 'queries.parquet')
    labels = np.load(root / 'map_all_query_labels.npz')
    old = np.load(root / 'labels.npz')['paired'][:, 0]
    paths = [Path(__file__), root / 'queries.parquet', root / 'map_all_query_labels.npz']
    results = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        parent = Path(f'artifacts/map_expanded_fit_gap_{seed}_20260912.json')
        d = json.loads(parent.read_text())
        for path, digest in d['bindings'].items():
            assert sha256(Path(path)) == digest
        prediction_path = root / f'map_expanded_fit_gap_predictions_{seed}.npz'
        p = np.load(prediction_path)
        partition = root / f'partition_{seed}.npz'
        roles = np.load(partition)
        scores = {}
        for role in ('fit', 'tune'):
            mask = roles[role] & labels['paired'][:, 0]
            if role == 'tune':
                mask &= old
            scores[role] = {}
            for arm in ('own', 'full'):
                loss = smooth_l1((p[arm][mask] - labels['future'][mask, 0]) / scale)
                scores[role][arm] = float(pd.Series(loss, index=q.loc[mask, 'uniquepid']).groupby(level=0).mean().mean())
        results.append(dict(seed=seed, roles=scores))
        paths.extend([parent, prediction_path, partition])
    Path('artifacts/map_value_objective_20260912.json').write_text(json.dumps(dict(
        scale=scale, results=results, training_performed=False, runtime_modified=False,
        limitations=['Value objective only; excludes JEPA latent loss and gradients.',
                     'Selected checkpoints, not complete optimization paths.',
                     'Comparisons are descriptive; no revised promotion gate.'],
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    print('completed', len(results), 'seeds')


if __name__ == '__main__':
    main()
