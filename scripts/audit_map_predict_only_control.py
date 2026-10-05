"""Verify event-time prediction without correction on real source histories."""
from copy import copy, deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_numeric_control import without_other_numbers
from whole_body_jepa.native_predict_only_control import PredictOnlyControl
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output = Path('artifacts/map_predict_only_control_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    q = pd.read_parquet(root/'queries.parquet')
    eligible = np.load(root/'labels.npz')['paired'][:, 0]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)), dict(np.load(root/'native_query_shards'/p.name))
    raw = RaggedNativeBatch(shards(), len(q)); assets = load_release(Path('models'))
    col = assets.checkpoint['variables'].index('map'); n = len(assets.checkpoint['variables'])
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    paths = [Path(__file__), Path('whole_body_jepa/native_predict_only_control.py'),
        Path('whole_body_jepa/native_projection_control.py')]
    results = []
    for seed in (7, 19, 31, 43, 59, 71, 83):
        parent = Path(f'artifacts/map_control_factorial_{seed}_20260912.json')
        for name, h in json.loads(parent.read_text())['bindings'].items():
            if sha256(Path(name)) != h: raise ValueError(f'changed dependency: {name}')
        paths.append(parent)
        ck = torch.load(root/f'map_factorial_own_history_control_query_{seed}.pt',
            map_location='cpu', weights_only=False)
        fit = np.load(root/f'partition_{seed}.npz')['fit'] & eligible
        pools = [np.asarray(g.index) for _, g in q.loc[fit].groupby('uniquepid', sort=True)]
        rng = np.random.default_rng(seed+20260913)
        selected = np.array([rng.choice(pools[i]) for i in rng.choice(len(pools), 96, replace=False)])
        views = {}
        for name in ('own', 'control'):
            allowed = np.array(ck['allowed_variables'], bool)
            if name == 'own': allowed[np.arange(n) != col] = False
            view, ec, es = transformed_storage(raw, np.array(ck['numeric_center'], np.float32),
                np.array(ck['numeric_scale'], np.float32), allowed,
                assets.model.variable_module_ids.numpy(), np.flatnonzero(fit), own=name=='own')
            np.testing.assert_array_equal(ec, ck['event_center'])
            np.testing.assert_array_equal(es, ck['event_scale'])
            views[name] = view if name=='own' else without_other_numbers(view, col)
        reference = copy(views['own']); reference.query = views['control'].query
        model = deepcopy(assets.model); expand_history_projection(model, 6)
        model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
        model.belief_input_mode = 'observed_steps'; model.value_delta_activation = torch.nn.Identity()
        model.load_state_dict(ck['state_dict']); model.eval()
        with torch.inference_mode():
            expected = model(*reference.batch(selected))['value'][:, col].numpy()*scale+median
        saved = np.load(root/f'map_factorial_predictions_{seed}.npz')['own_history_control_query'][selected]
        np.testing.assert_allclose(expected, saved, rtol=1e-6, atol=1e-4)
        new_filter = PredictOnlyControl(model.input_dim, model.belief_filter.hidden, n, col,
            int(model.variable_module_ids[col]))
        new_filter.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = new_filter
        x = views['control'].batch(selected)
        active = x[0][:, :, n:2*n].bool().any(-1) | x[0][:, :, -2].bool()
        target = x[0][:, :, n+col].bool()
        positions = torch.arange(active.shape[1])[None, :]
        last_event = torch.where(active, positions, -1).max(-1).values
        last_target = torch.where(target, positions, -1).max(-1).values
        matched_end = ((last_event == last_target) & (last_target >= 0)).numpy()
        with torch.inference_mode():
            actual = model(*x)['value'][:, col].numpy()*scale+median
        np.testing.assert_allclose(actual[matched_end], expected[matched_end], rtol=1e-6, atol=1e-4)
        results.append(dict(seed=seed, source_fit_patients=len(selected),
            matched_endpoint=int(matched_end.sum()), excluded_later_events=int((~matched_end).sum()),
            extra_events_mean=float((active.sum(-1)-target.sum(-1)).float().mean()),
            max_abs_difference=float(np.max(np.abs(actual[matched_end]-expected[matched_end])))))
        print('complete', seed, results[-1], flush=True)
        del model, new_filter, views, reference, view, x
    output.write_text(json.dumps(dict(complete=True, results=results, role='source_fit_only',
        original_predictions_reproduced=True, training_performed=False, runtime_modified=False,
        limitation='Structural replay on matched endpoints, not a new accuracy gain or full-organ implementation.',
        bindings={str(p):sha256(p) for p in paths}), indent=2)+'\n')


if __name__ == '__main__': main()
