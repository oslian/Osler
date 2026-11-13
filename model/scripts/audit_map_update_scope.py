"""Fixed-weight MAP update-scope intervention, not a trained repair."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from osler_jepa.patient_state import NeuralBeliefFilter
from scripts.train_availability_encoder import predict
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import apply_numeric_rescale, restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors


class MapUpdateFilter(NativeEventFilter):
    def __init__(self, input_dim, hidden, mask_index):
        super().__init__(input_dim, hidden)
        self.mask_index = mask_index

    def forward(self, observations, observation_mask=None):
        if observation_mask is None:
            raise ValueError('explicit observation mask required')
        active = observation_mask.bool() & (observations[..., self.mask_index] > 0)
        return NeuralBeliefFilter.forward(self, observations, active)


def main():
    output = Path('artifacts/map_update_scope_20260912.json')
    if output.exists(): raise ValueError('fresh output required')
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; col = variables.index('map')
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    q = pd.read_parquet(root / 'queries.parquet').rename(columns={'uniquepid': 'subject_id'})
    labels = np.load(root / 'map_all_query_labels.npz'); old = np.load(root / 'labels.npz')['paired'][:, 0]
    eligible = labels['paired'][:, 0]; q['truth'] = labels['future'][:, 0]
    paths = [Path(__file__)]; results = []
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for seed in (7, 19, 31, 43, 59, 71, 83):
        parent = Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        d = json.loads(parent.read_text())
        for p, digest in d['bindings'].items(): assert sha256(Path(p)) == digest
        paths.append(parent)
        roles = dict(np.load(root / f'partition_{seed}.npz')); fit = roles['fit'] & eligible
        cp = torch.load(root / f'map_expanded_native_full_{seed}.pt', map_location='cpu', weights_only=False)
        base = dict(np.load(root / 'map_all_query_inputs.npz')); native = dict(np.load(root / 'map_all_native_tokens.npz'))
        arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
        arrays = apply_numeric_rescale(arrays, np.array(cp['numeric_center'], np.float32), np.array(cp['numeric_scale'], np.float32))
        events, center, scales = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
        np.testing.assert_array_equal(center, cp['event_center']); np.testing.assert_array_equal(scales, cp['event_scale'])
        view = append_events(restrict_input_support(arrays, np.array(cp['allowed_variables']), assets.model.variable_module_ids.numpy()), events)
        model = deepcopy(assets.model); expand_history_projection(model, 6)
        model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
        model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
        model.load_state_dict(cp['state_dict']); model.eval()
        saved = dict(np.load(root / f'map_expanded_native_predictions_{seed}.npz'))
        original = predict(model, view, rows[old], col, median, scale)
        np.testing.assert_array_equal(original, saved['expanded_full'])
        replacement = MapUpdateFilter(model.input_dim, model.belief_filter.hidden, len(variables) + col)
        replacement.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = replacement
        changed = predict(model, view, rows[old], col, median, scale)
        scores = {}
        for role in ('tune', 'evaluation', 'hospital'):
            m = (roles[role] & eligible)[old]; f = q.loc[old].loc[m]
            p = dict(original=original[m], map_updates=changed[m], own=saved['expanded_own'][m])
            scores[role] = dict(mae={k: patient_equal_mae(f, f.truth, a) for k, a in p.items()},
                comparisons={ref: paired_model_errors(f, p, ref, seed=20260912, repetitions=20000)['map_updates'] for ref in ('original', 'own')})
        private = root / f'map_update_scope_{seed}.npz'; np.savez_compressed(private, original=original, map_updates=changed); paths.append(private)
        results.append(dict(seed=seed, roles=scores)); print('complete', seed, flush=True)
    output.write_text(json.dumps(dict(results=results, training_performed=False, runtime_modified=False,
        limitations=['Removes non-MAP event content and update cadence together.', 'Fixed weights, not a train-matched intervention.'],
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for role in ('tune', 'evaluation', 'hospital'):
        for ref in ('original', 'own'):
            cs = [r['roles'][role]['comparisons'][ref] for r in results]
            print(role, ref, 'gain', sum(c['ci95'][1] < 0 for c in cs), 'harm', sum(c['ci95'][0] > 0 for c in cs))


if __name__ == '__main__': main()
