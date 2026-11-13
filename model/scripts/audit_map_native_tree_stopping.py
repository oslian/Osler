"""Tune-only staged selection, same fixed fitted trees and candidate epochs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_map_native_anchor_tree import feature_view
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private = root/'tree_trial'
    parent = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_native_tree_stopping_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    original = json.loads(parent.read_text())
    for p, h in original['bindings'].items():
        assert sha256(Path(p)) == h, p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][valid, 0]
    current = labels['current'][valid, 0]
    x = np.load(private/'features.npy', mmap_mode='r')
    grid = (0, 25, 50, 100, 150, 200)
    results = []
    for entry in original['seeds']:
        seed = entry['seed']
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        tune = roles['tune']
        points = dict(persistence=current)
        old = dict(np.load(private/f'predictions_{seed}.npz'))
        item = dict(seed=seed, selection={}, roles={})
        for mode in ('own', 'full', 'support'):
            pack = joblib.load(private/f'{mode}_{seed}.joblib')
            model = pack['model']
            view = feature_view(x[tune], len(pack['variables']), pack['target_index'], pack['allowed'], mode)
            board = {0:patient_equal_mae(q.loc[tune], q.loc[tune,'truth'], current[tune])}
            for step, delta in enumerate(model.staged_predict(view), 1):
                if step in grid:
                    board[step] = patient_equal_mae(q.loc[tune], q.loc[tune,'truth'], current[tune]+pack['target_scale']*delta)
            selected = min(board, key=board.get)
            item['selection'][mode] = dict(iterations=selected, tune_mae_by_iteration=board)
            del view
            pred = current.copy()
            if selected:
                for start in range(0, len(q), 4096):
                    end = min(len(q), start+4096)
                    view = feature_view(x[start:end], len(pack['variables']), pack['target_index'], pack['allowed'], mode)
                    for step, delta in enumerate(model.staged_predict(view), 1):
                        if step == selected:
                            pred[start:end] += pack['target_scale']*delta
                            break
            points[mode] = pred
            # The last-stage tune result must reproduce the stored final model.
            np.testing.assert_allclose(board[200], patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],old[mode][tune]), rtol=1e-12)
            print(f'Seed {seed} {mode} selected {selected}', flush=True)
        for role in ('evaluation', 'hospital'):
            m = roles[role]
            frame = q.loc[m]
            p = {k:v[m] for k,v in points.items()}
            p['old_full'] = old['full'][m]
            item['roles'][role] = dict(mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                versus_own=paired_model_errors(frame,p,'own',seed=20260912),
                versus_old_full=paired_model_errors(frame,p,'old_full',seed=20260912))
        np.savez_compressed(private/f'stopped_predictions_{seed}.npz', **points)
        results.append(item)
    bindings = [parent, Path(__file__), Path('scripts/train_map_native_anchor_tree.py'), private/'features.npy']
    bindings += [private/f'stopped_predictions_{s}.npz' for s in (7,19,31,43,59,71,83)]
    output.write_text(json.dumps(dict(complete=True, grid=grid, seeds=results,
        training_performed=False, runtime_modified=False, promotion_allowed=False,
        selection_data='original source tune only; evaluation already previously inspected',
        bindings={str(p):sha256(p) for p in bindings}), indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role, {k:float(np.median([r['roles'][role]['mae'][k] for r in results])) for k in p})
        for ref in ('own','old_full'):
            c=[r['roles'][role]['versus_'+ref]['full'] for r in results]
            print('full vs',ref,'gain/harm',sum(v['ci95'][1]<0 for v in c),sum(v['ci95'][0]>0 for v in c))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
