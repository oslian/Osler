"""Evaluate selected MAP weights on fit and unchanged tune patients, without fitting."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_availability_encoder import predict
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import apply_numeric_rescale, restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import append_events, expand_history_projection
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    col = variables.index('map')
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    q = pd.read_parquet(root / 'queries.parquet').rename(columns={'uniquepid': 'subject_id'})
    labels = np.load(root / 'map_all_query_labels.npz')
    eligible = labels['paired'][:, 0]
    old = np.load(root / 'labels.npz')['paired'][:, 0]
    q['truth'] = labels['future'][:, 0]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for seed in (7, 19, 31, 43, 59, 71, 83):
        output = Path(f'artifacts/map_expanded_fit_gap_{seed}_20260912.json')
        if output.exists():
            raise ValueError('fresh output required')
        parent = Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        report = json.loads(parent.read_text())
        for path, digest in report['bindings'].items():
            assert sha256(Path(path)) == digest, path
        roles = dict(np.load(root / f'partition_{seed}.npz'))
        masks = dict(fit=roles['fit'] & eligible,
                     tune=roles['tune'] & eligible & old)
        saved = np.load(root / f'map_expanded_native_predictions_{seed}.npz')
        score_rows = np.flatnonzero(masks['fit'] | masks['tune'] | (old & eligible))
        points = {'persistence': labels['current'][:, 0]}
        for arm in ('own', 'full'):
            checkpoint = torch.load(root / f'map_expanded_native_{arm}_{seed}.pt',
                                    map_location='cpu', weights_only=False)
            base = dict(np.load(root / 'map_all_query_inputs.npz'))
            native = dict(np.load(root / 'map_all_native_tokens.npz'))
            arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
            arrays = apply_numeric_rescale(arrays,
                np.array(checkpoint['numeric_center'], dtype=np.float32),
                np.array(checkpoint['numeric_scale'], dtype=np.float32))
            # Reconstruct the exact training transform, including its known mask.
            ec = np.array(checkpoint['event_center'], dtype=np.float64)
            es = np.array(checkpoint['event_scale'], dtype=np.float64)
            from whole_body_jepa.history_event_input import prepare_events
            events, fitted_center, fitted_scale = prepare_events(
                values, np.unique(arrays['history'][rows[masks['fit']]]))
            np.testing.assert_array_equal(ec, fitted_center)
            np.testing.assert_array_equal(es, fitted_scale)
            if arm == 'own':
                events[:] = 0
            view = append_events(restrict_input_support(arrays,
                np.array(checkpoint['allowed_variables']), assets.model.variable_module_ids.numpy()), events)
            model = deepcopy(assets.model)
            expand_history_projection(model, 6)
            model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
            model.value_delta_activation = torch.nn.Identity()
            model.belief_input_mode = 'observed_steps'
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            point = np.full(len(q), np.nan)
            point[score_rows] = predict(model, view, rows[score_rows], col, median, scale)
            # Batch membership may differ; tolerate floating-point roundoff only.
            np.testing.assert_allclose(point[old][eligible[old]],
                saved['expanded_' + arm][eligible[old]], rtol=0, atol=1e-5)
            points[arm] = point
            print('replayed', seed, arm, flush=True)
            del model, view, arrays, base, native, events, checkpoint
        scores = {}
        for role, mask in masks.items():
            frame = q.loc[mask]
            p = {key: value[mask] for key, value in points.items()}
            scores[role] = dict(rows=int(mask.sum()), patients=frame.subject_id.nunique(),
                mae={k: patient_equal_mae(frame, frame.truth, a) for k, a in p.items()},
                full_vs_own=paired_model_errors(frame, p, 'own', seed=20260912, repetitions=20000)['full'])
        path = root / f'map_expanded_fit_gap_predictions_{seed}.npz'
        np.savez_compressed(path, **points)
        output.write_text(json.dumps(dict(complete=True, training_performed=False,
            runtime_modified=False, seed=seed, roles=scores,
            bindings={str(p): sha256(p) for p in (parent, Path(__file__), path)}), indent=2) + '\n')
        print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
