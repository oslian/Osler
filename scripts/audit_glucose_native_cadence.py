"""Fixed-weight sensitivity to non-glucose recurrent updates, not promotion."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from osler_jepa.patient_state import NeuralBeliefFilter
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support


class GlucoseUpdateFilter(NativeEventFilter):
    def forward(self, observations, observation_mask=None):
        return NeuralBeliefFilter.forward(self, observations, observations[..., self.mask_column].bool())


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/glucose_native_cadence_20260911.json')
    reference = Path('artifacts/rate_glucose_jepa_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(reference.read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('reference input changed')
    base, native = (dict(np.load(root / name)) for name in ('body_arrays.npz', 'native_tokens.npz'))
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']; n = len(variables); k = variables.index('glucose')
    arrays, rows, values = native_arrays(base, native, n, base['presence'].shape[1])
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    labels = np.load(root / 'multitarget_labels.npz')
    frame = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    frame = frame.rename(columns={'uniquepid': 'subject_id'})
    frame['truth'] = labels['future'][:, k]
    torch.set_num_threads(4)
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        saved = torch.load(root / f'glucose_native_real_{seed}.pt', weights_only=True)
        model = deepcopy(assets.model)
        expand_history_projection(model, saved['event_width'])
        model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
        model.load_state_dict(saved['state_dict'])
        model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
        view = restrict_input_support(arrays, np.array(saved['allowed_variables']), model.variable_module_ids.numpy())
        numbers = np.where(np.isfinite(values), (values - saved['event_center']) / saved['event_scale'], 0.)
        view = append_events(view, np.column_stack([numbers, np.isfinite(values)]).astype(np.float32))
        original = predict(model, view, rows, k, assets.scaler.medians[k], assets.scaler.scales[k])
        np.testing.assert_allclose(original, np.load(root / f'glucose_native_predictions_{seed}.npz')['real'], atol=1e-5, rtol=0)
        restricted = GlucoseUpdateFilter(model.input_dim, model.belief_filter.hidden)
        restricted.load_state_dict(model.belief_filter.state_dict()); restricted.mask_column = n + k
        model.belief_filter = restricted
        altered = predict(model, view, rows, k, assets.scaler.medians[k], assets.scaler.scales[k])
        partitions = np.load(root / f'partition_{seed}.npz')
        item = dict(seed=seed, roles={})
        for role in ('fit', 'tune', 'evaluation', 'hospital'):
            mask = partitions[role][valid] & labels['paired'][:, k]
            f = frame.loc[mask]
            item['roles'][role] = dict(
                original_mae=patient_equal_mae(f, f.truth, original[mask]),
                glucose_updates_mae=patient_equal_mae(f, f.truth, altered[mask]),
                comparison=paired(f, altered[mask], original[mask], f'glucose-cadence:{seed}:{role}', 2000))
        results.append(item)
        print('completed', seed, flush=True)
    output.write_text(json.dumps(dict(complete=True, status='sensitivity_only', runtime_modified=False,
        fitting_performed=False, binding={str(p): sha256(p) for p in (reference, Path(__file__))}, seeds=results), indent=2) + '\n')
    for role in ('fit', 'tune', 'evaluation', 'hospital'):
        g = [s['roles'][role] for s in results]
        print(role, {m: float(np.median([x[m] for x in g])) for m in ('original_mae', 'glucose_updates_mae')},
              'gain/harm', sum(x['comparison']['ci95'][1] < 0 for x in g), sum(x['comparison']['ci95'][0] > 0 for x in g))


if __name__ == '__main__':
    main()
