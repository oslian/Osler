"""Source-fixed raw-history comparison on MAP availability anchors."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.audit_potassium_raw_history import history_features
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def feature_view(x, n, col, allowed, mode):
    """Same lag/mask/age contract; support arm removes only other numbers."""
    if mode not in ('own', 'full', 'support'):
        raise ValueError('unknown mode')
    lag = x.shape[1] // (3*n)
    v = np.array(x, copy=True).reshape(len(x), 3, lag, n)
    v[:, 0, :, ~allowed] = 0
    v[:, 1, :, ~allowed] = 0
    v[:, 2, :, ~allowed] = 7
    if mode == 'own':
        return v[:, :, :, col].reshape(len(x), -1)
    if mode == 'support':
        other = np.arange(n) != col
        v[:, 0, :, other] = 0
    return v.reshape(len(x), -1)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_anchor_inputs_20260912.json')
    output = Path('artifacts/map_native_anchor_tree_20260912.json')
    dest = root/'tree_trial'
    if output.exists() or dest.exists():
        raise ValueError('fresh outputs required')
    ref = json.loads(parent.read_text())
    for p, h in ref['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError(f'changed binding: {p}')
    labels = dict(np.load(root/'labels.npz'))
    eligible = labels['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[eligible].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][eligible, 0]
    current = labels['current'][eligible, 0]
    assets = load_release(Path('models'))
    n = len(assets.checkpoint['variables'])
    col = assets.checkpoint['variables'].index('map')
    scale = float(assets.scaler.scales[col])
    y = (q.truth.to_numpy()-current)/scale
    compact = np.full(len(eligible), -1, int)
    compact[eligible] = np.arange(eligible.sum())
    dest.mkdir()
    x = None
    seen = np.zeros(len(q), bool)
    for shard in ref['shards']:
        if sha256(Path(shard['path'])) != shard['sha256']:
            raise ValueError('changed shard')
        a = dict(np.load(shard['path']))
        keep = eligible[a['global_query_rows']]
        indices = compact[a['global_query_rows'][keep]]
        f = history_features(a, a['query_rows'][keep], col, 'full')
        if x is None:
            x = np.lib.format.open_memmap(dest/'features.npy', mode='w+', dtype='float32', shape=(len(q), f.shape[1]))
        assert not seen[indices].any()
        x[indices] = f
        seen[indices] = True
    assert seen.all()
    x.flush()
    lag = x.shape[1] // (3*n)
    bindings = [parent, Path(__file__), Path('scripts/audit_potassium_raw_history.py'),
                Path('whole_body_jepa/change_evaluation.py'), root/'queries.parquet', root/'labels.npz']
    results = []
    for seed in (7,19,31,43,59,71,83):
        path = root/f'partition_{seed}.npz'
        roles = {k:v[eligible] for k,v in dict(np.load(path)).items()}
        assert np.all(np.stack(list(roles.values())).sum(axis=0) == 1)
        fit = roles['fit']
        # Only fitting patients define channel support; no tune or test values.
        observed = np.asarray(x[fit, lag*n:2*lag*n]).reshape(fit.sum(), lag, n).any(axis=1)
        counts = pd.DataFrame(observed).groupby(q.loc[fit, 'subject_id'].to_numpy()).any().sum().to_numpy()
        allowed = counts > 0
        subjects = q.loc[fit, 'subject_id']
        w = 1/subjects.groupby(subjects).transform('size').to_numpy(float)
        w *= len(w)/w.sum()
        points = dict(persistence=current)
        item = dict(seed=seed, source_supported_variables=int(allowed.sum()), roles={})
        for mode in ('own', 'full', 'support'):
            train = feature_view(x[fit], n, col, allowed, mode)
            model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200,
                learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=30,
                l2_regularization=10., max_bins=63, early_stopping=False, random_state=seed)
            model.fit(train, y[fit], sample_weight=w)
            del train
            pred = np.empty(len(q))
            for start in range(0, len(q), 4096):
                end = min(len(q), start+4096)
                pred[start:end] = current[start:end]+scale*model.predict(feature_view(x[start:end], n, col, allowed, mode))
            points[mode] = pred
            modelpath = dest/f'{mode}_{seed}.joblib'
            joblib.dump(dict(model=model, allowed=allowed, target_index=col, target_scale=scale,
                             variables=assets.checkpoint['variables'], mode=mode), modelpath)
            bindings.append(modelpath)
            print(f'Seed {seed} {mode} complete', flush=True)
        for role in ('evaluation', 'hospital'):
            m = roles[role]
            frame = q.loc[m]
            p = {k:v[m] for k,v in points.items()}
            item['roles'][role] = dict(rows=int(m.sum()), patients=int(frame.subject_id.nunique()),
                mae={k:patient_equal_mae(frame, frame.truth, v) for k,v in p.items()},
                versus_own=paired_model_errors(frame, p, 'own', seed=20260912),
                versus_support=paired_model_errors(frame, p, 'support', seed=20260912),
                versus_persistence=paired_model_errors(frame, p, 'persistence', seed=20260912))
        prediction = dest/f'predictions_{seed}.npz'
        np.savez_compressed(prediction, **points)
        bindings.extend([path, prediction])
        results.append(item)
    report = dict(complete=True, runtime_modified=False, promotion_allowed=False,
                  protocol='Fixed 15-leaf 200-iteration HGB; no tune-based selection. Patient equal fit/score; same native task.',
                  scope='Raw fixed-lag diagnostic, not JEPA training or full external/calibration gate',
                  seeds=results, bindings={str(p):sha256(p) for p in bindings})
    output.write_text(json.dumps(report, indent=2)+'\n')
    for role in ('evaluation', 'hospital'):
        print(role, {m:float(np.median([r['roles'][role]['mae'][m] for r in results])) for m in points})
        for refname in ('own', 'support', 'persistence'):
            stats = [r['roles'][role]['versus_'+refname]['full'] for r in results]
            print('full vs', refname, 'gain/harm', sum(s['ci95'] is not None and s['ci95'][1]<0 for s in stats),
                  sum(s['ci95'] is not None and s['ci95'][0]>0 for s in stats))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
