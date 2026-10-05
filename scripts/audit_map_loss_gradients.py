"""Source-fit value/JEPA gradient geometry at selected MAP checkpoints."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch, configure, teacher_target
from scripts.audit_magnesium_auxiliary_gradient import geometry
from whole_body_jepa.availability_inputs import apply_numeric_rescale, restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/map_loss_gradients_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']; col = variables.index('map')
    q = pd.read_parquet(root / 'queries.parquet')
    labels = np.load(root / 'map_all_query_labels.npz')
    truth = (labels['future'][:, 0] - assets.scaler.medians[col]) / assets.scaler.scales[col]
    probes = []; paths = [Path(__file__), Path('scripts/audit_magnesium_auxiliary_gradient.py')]
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for seed in (7, 19, 31, 43, 59, 71, 83):
        parent = Path(f'artifacts/map_expanded_native_{seed}_20260912.json')
        report = json.loads(parent.read_text())
        for path, digest in report['bindings'].items():
            assert sha256(Path(path)) == digest
        paths.append(parent)
        fit = np.load(root / f'partition_{seed}.npz')['fit'] & labels['paired'][:, 0]
        pools = [np.flatnonzero(fit & q.uniquepid.eq(s).to_numpy()) for s in sorted(set(q.loc[fit, 'uniquepid']))]
        rng = np.random.default_rng(seed + 20260912)
        batches = [np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)]) for _ in range(3)]
        for arm in ('own', 'full'):
            checkpoint = torch.load(root / f'map_expanded_native_{arm}_{seed}.pt', map_location='cpu', weights_only=False)
            base = dict(np.load(root / 'map_all_query_inputs.npz'))
            native = dict(np.load(root / 'map_all_native_tokens.npz'))
            arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
            arrays = apply_numeric_rescale(arrays, np.array(checkpoint['numeric_center'], np.float32),
                                          np.array(checkpoint['numeric_scale'], np.float32))
            events, center, scale = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
            np.testing.assert_array_equal(center, checkpoint['event_center'])
            np.testing.assert_array_equal(scale, checkpoint['event_scale'])
            if arm == 'own': events[:] = 0
            view = append_events(restrict_input_support(arrays, np.array(checkpoint['allowed_variables']),
                                 assets.model.variable_module_ids.numpy()), events)
            model = deepcopy(assets.model); expand_history_projection(model, 6)
            model.belief_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
            model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
            model.load_state_dict(checkpoint['state_dict']); configure(model, True)
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and
                     n.split('.')[0] in ('belief_filter', 'variable_mixer', 'module_mixer', 'state_encoder')]
            for bi, chosen in enumerate(batches):
                x = batch(view, rows[chosen]); result = model(*x)
                y = torch.tensor(truth[chosen], dtype=torch.float32)
                value = F.smooth_l1_loss(result['value'][:, col], y)
                latent = _masked_latent_prediction_loss(result['predicted_target_latent'][:, col],
                    teacher_target(model, x[1], y, col), torch.ones(len(y), dtype=torch.bool))
                a = torch.autograd.grad(value, [p for _, p in named], retain_graph=True, allow_unused=True)
                b = torch.autograd.grad(latent, [p for _, p in named], allow_unused=True)
                groups = {'all': geometry(a, b)}
                for group in sorted(set(n.split('.')[0] for n, _ in named)):
                    ids = [i for i, (n, _) in enumerate(named) if n.startswith(group + '.')]
                    groups[group] = geometry([a[i] for i in ids], [b[i] for i in ids])
                probes.append(dict(seed=seed, arm=arm, batch=bi, value_loss=float(value.detach()),
                                   latent_loss=float(latent.detach()), groups=groups))
            del model, view, arrays, base, native, checkpoint, events, result, x, a, b
        print('complete', seed, flush=True)
    output.write_text(json.dumps(dict(probes=probes, role='source_fit_only', training_performed=False,
        runtime_modified=False, parameter_scope=['belief_filter', 'variable_mixer', 'module_mixer', 'state_encoder'],
        geometry_semantics='primary=value; retention=JEPA, unit coefficient as trained',
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for arm in ('own', 'full'):
        gs = [p['groups']['all'] for p in probes if p['arm'] == arm]
        print(arm, 'opposing', sum(g['cosine'] < 0 for g in gs), 'of', len(gs),
              'median_cosine', np.median([g['cosine'] for g in gs]),
              'median_ratio', np.median([g['retention_to_primary'] for g in gs]))


if __name__ == '__main__': main()
