"""Matched raw-history controls for recorded pressor context; retrospective only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_potassium_raw_history import history_features
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true', help='Compare fixed lag-history additions to current-rate control')
    history_trial = parser.parse_args().history
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    prefix = 'map_pressor_history_training' if history_trial else 'map_recorded_pressor_training'
    output = Path(f'artifacts/{prefix}_20260912.json')
    if output.exists(): raise ValueError('fresh output required')
    parent = Path('artifacts/map_recorded_pressors_20260912.json')
    for path, digest in json.loads(parent.read_text())['bindings'].items():
        assert sha256(Path(path)) == digest
    assets = load_release(Path('models')); col = assets.checkpoint['variables'].index('map')
    scale = assets.scaler.scales[col]
    base = dict(np.load(root / 'map_all_query_inputs.npz')); rows = base['query_rows']
    labels = np.load(root / 'map_all_query_labels.npz'); eligible = labels['paired'][:, 0]
    old = np.load(root / 'labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').rename(columns={'uniquepid': 'subject_id'})
    q['truth'] = labels['future'][:, 0]; current = labels['current'][:, 0]
    y = (q.truth.to_numpy() - current) / scale
    pressors = np.load(root / 'map_recorded_pressors.npz')
    assert not bool(pressors['online_eligible'])
    rate = pressors['rate'].reshape(len(q), -1)
    known = pressors['known'].reshape(len(q), -1).astype(float)
    state = pressors['status']
    support = np.concatenate([known, pressors['age_hours'],
        (state == 'invalid_latest').astype(float), (state == 'conflicting_latest').astype(float)], axis=1)
    items = []; paths = [Path(__file__), parent, Path('scripts/audit_potassium_raw_history.py')]
    if history_trial:
        history_parent = Path('artifacts/map_pressor_history_20260912.json')
        for path, digest in json.loads(history_parent.read_text())['bindings'].items():
            assert sha256(Path(path)) == digest
        paths.append(history_parent)
        past = np.load(root / 'map_pressor_history.npz')
        lag_numeric = np.concatenate([past[f'{key}_lag{lag}'].reshape(len(q), -1)
            for lag in (1, 3) for key in ('rate', 'delta')], axis=1)
        lag_support = np.concatenate([a.reshape(len(q), -1) for lag in (1, 3) for a in (
            past[f'known_lag{lag}'], past[f'delta_known_lag{lag}'], past[f'age_hours_lag{lag}'],
            past[f'record_gap_hours_lag{lag}'], past[f'status_lag{lag}'] == 'invalid_latest',
            past[f'status_lag{lag}'] == 'conflicting_latest')], axis=1).astype(float)
    for seed in (7, 19, 31, 43, 59, 71, 83):
        partition = root / f'partition_{seed}.npz'; paths.append(partition)
        roles = dict(np.load(partition)); fit = roles['fit'] & eligible; tune = roles['tune'] & eligible & old
        allowed = source_patient_support(base, rows, q.subject_id, fit) > 0
        view = restrict_input_support(base, allowed, assets.model.variable_module_ids.numpy())
        history = history_features(view, rows, col, 'full').astype(float)
        subjects = q.loc[fit, 'subject_id']; weights = 1 / subjects.groupby(subjects).transform('size').to_numpy(float)
        weights *= len(weights) / weights.sum()
        source_rate = rate.copy(); source_rate[:, ~np.isfinite(rate[fit]).any(axis=0)] = np.nan
        points = {'persistence': current}; selection = {}
        for arm in ('baseline', 'support', 'rate'):
            extra = np.concatenate([source_rate, support], axis=1)
            if history_trial:
                source_lag = lag_numeric.copy()
                source_lag[:, ~np.isfinite(source_lag[fit]).any(axis=0)] = np.nan
                history_extra = np.concatenate([source_lag, lag_support], axis=1)
                if arm == 'baseline': history_extra[:] = 0
                elif arm == 'support': history_extra[:, :lag_numeric.shape[1]] = 0
                extra = np.concatenate([extra, history_extra], axis=1)
            else:
                if arm == 'baseline': extra[:] = 0
                elif arm == 'support': extra[:, :rate.shape[1]] = 0
            x = np.concatenate([history, extra], axis=1)
            best = None; board = []
            for leaves in (7, 15, 31):
                model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, learning_rate=.05,
                    max_leaf_nodes=leaves, min_samples_leaf=30, l2_regularization=10., max_bins=63,
                    early_stopping=False, random_state=seed).fit(x[fit], y[fit], sample_weight=weights)
                mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], current[tune] + scale * model.predict(x[tune]))
                board.append(dict(leaves=leaves, mae=mae))
                if best is None or mae < best[0]: best = (mae, model)
            points[arm] = current + scale * best[1].predict(x); selection[arm] = board
        reference_path = root / (f'map_recorded_pressor_predictions_{seed}.npz' if history_trial
                                 else f'map_selection_predictions_{seed}.npz')
        paths.append(reference_path)
        previous = np.load(reference_path)
        np.testing.assert_array_equal(points['baseline'][eligible], previous['rate' if history_trial else 'expanded_full'][eligible])
        scored = {}
        for role in ('evaluation', 'hospital'):
            scored[role] = {}
            for subset, subset_mask in (('old', old), ('all', np.ones(len(q), bool))):
                mask = roles[role] & eligible & subset_mask; frame = q.loc[mask]
                p = {k: a[mask] for k, a in points.items()}
                scored[role][subset] = dict(mae={k: patient_equal_mae(frame, frame.truth, a) for k, a in p.items()},
                    rate_comparisons={ref: paired_model_errors(frame, p, ref, seed=20260912, repetitions=20000)['rate']
                                      for ref in ('baseline', 'support', 'persistence')})
        private = root / f'{prefix}_predictions_{seed}.npz' if history_trial else root / f'map_recorded_pressor_predictions_{seed}.npz'
        np.savez_compressed(private, **points); paths.append(private)
        items.append(dict(seed=seed, selection=selection, roles=scored)); print('completed', seed, flush=True)
    output.write_text(json.dumps(dict(complete=True, seeds=items, baseline_exact=True,
        status='retrospective_diagnostic_only', history_trial=history_trial, online_eligible=False, runtime_modified=False,
        limitations=['Recorded infusion time lacks independent availability.', 'Not neural JEPA retraining.',
                     'Support controls include treatment presence/age; not a randomized placebo.',
                     'Repeated source splits and hospital inspection, no untouched external test.'],
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        for subset in ('old', 'all'):
            ss = [i['roles'][role][subset] for i in items]
            print(role, subset, {k: float(np.median([s['mae'][k] for s in ss])) for k in points},
                {r: (sum(s['rate_comparisons'][r]['ci95'][1] < 0 for s in ss),
                     sum(s['rate_comparisons'][r]['ci95'][0] > 0 for s in ss)) for r in ('baseline', 'support', 'persistence')})


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
