"""Matched MAP fixed-lag versus quality-qualified event-summary trial."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from whole_body_jepa.event_window_history import event_windows, select_event_features
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    out = Path('artifacts/map_event_history_20260912.json')
    if out.exists():
        raise ValueError('fresh output required')
    parent = Path('artifacts/native_map_raw_history_20260912.json')
    for p, digest in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p)) != digest:
            raise ValueError(f'changed parent dependency: {p}')
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']; col = variables.index('map')
    base = dict(np.load(root / 'body_arrays_bedside.npz')); rows = base['query_rows']
    labels = dict(np.load(root / 'bedside_future_labels.npz'))
    j = labels['variables'].tolist().index('map')
    eligible = labels['paired'][:, j]
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'}); q['truth'] = labels['future'][:, j]
    current = labels['current'][:, j]; scale = assets.scaler.scales[col]
    y = (q.truth.to_numpy() - current) / scale
    labs = pd.read_parquet(root / 'body_labs.parquet')
    if 'quality_valid' not in labs:
        labs['quality_valid'] = True
    raw = pd.concat([labs, pd.read_parquet(root / 'qualified_bedside_events.parquet')], ignore_index=True)
    print('Building legal event histories', flush=True)
    # Retain all queries; source eligibility and scoring masks remain unchanged.
    summaries = event_windows(q, raw, variables, assets.scaler.medians, assets.scaler.scales)
    private = root / 'map_event_history'; private.mkdir(exist_ok=True)
    np.savez_compressed(private / 'summary_features.npz', features=summaries)
    items = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        roles = {k: v[valid] & eligible for k, v in dict(np.load(root / f'partition_{seed}.npz')).items()}
        fit, tune = roles['fit'], roles['tune']
        allowed = source_patient_support(base, rows, q.subject_id, fit) > 0
        subjects = q.loc[fit, 'subject_id']
        w = 1 / subjects.groupby(subjects).transform('size').to_numpy(float); w *= len(w) / w.sum()
        points = {'persistence': current}; item = dict(seed=seed, selection={}, roles={})
        for mode in ('own', 'full', 'support', 'lag_own', 'lag_full'):
            kind = mode.removeprefix('lag_'); channels = allowed.copy()
            if kind == 'own': channels[np.arange(len(variables)) != col] = False
            view = restrict_input_support(base, channels, assets.model.variable_module_ids.numpy())
            if kind == 'support': view['current'][:, np.arange(len(variables)) != col] = 0
            x = history_features(view, rows, col, 'full').astype(float)
            if not mode.startswith('lag_'):
                x = np.concatenate([x, select_event_features(summaries, allowed, col, kind)], axis=1)
            best = None; board = []
            for leaves in (7, 15, 31):
                model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200,
                    learning_rate=.05, max_leaf_nodes=leaves, min_samples_leaf=30,
                    l2_regularization=10., max_bins=63, early_stopping=False,
                    random_state=seed).fit(x[fit], y[fit], sample_weight=w)
                pred = current[tune] + scale * model.predict(x[tune])
                mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], pred)
                board.append(dict(leaves=leaves, mae=mae))
                if best is None or mae < best[0]: best = (mae, model)
            points[mode] = current + scale * best[1].predict(x)
            item['selection'][mode] = board
        prior = np.load(root / f'map_raw_predictions_{seed}.npz')
        for now, old in (('lag_own', 'own'), ('lag_full', 'real')):
            np.testing.assert_array_equal(points[now], prior[old])
        for role in ('evaluation', 'hospital'):
            m = roles[role]; f = q.loc[m]; p = {k: v[m] for k, v in points.items()}
            refs = {r: paired_model_errors(f, p, r, seed=20260912, repetitions=20000)
                    for r in ('own', 'support', 'lag_full', 'lag_own', 'persistence')}
            item['roles'][role] = dict(mae={k: patient_equal_mae(f, f.truth, v) for k, v in p.items()},
                comparisons={f'{a}_vs_{b}': refs[b][a] for a, b in (
                    ('full', 'own'), ('full', 'support'), ('full', 'lag_full'),
                    ('own', 'lag_own'), ('full', 'persistence'))})
        np.savez_compressed(private / f'predictions_{seed}.npz', **points)
        items.append(item); print('completed', seed, flush=True)
    paths = [parent, Path(__file__), Path('whole_body_jepa/event_window_history.py'),
             root / 'body_labs.parquet', root / 'qualified_bedside_events.parquet']
    paths += sorted(private.glob('*.npz'))
    out.write_text(json.dumps(dict(complete=True, status='diagnostic_only', runtime_modified=False,
        seeds=items, controls_exact=True, bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        print(role, {k: float(np.median([s['roles'][role]['mae'][k] for s in items])) for k in points},
              {k: (sum(s['roles'][role]['comparisons'][k]['ci95'][1] < 0 for s in items),
                   sum(s['roles'][role]['comparisons'][k]['ci95'][0] > 0 for s in items))
               for k in items[0]['roles'][role]['comparisons']})


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
