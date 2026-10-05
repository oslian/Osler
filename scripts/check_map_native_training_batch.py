"""Exercise the full ragged MAP cache through a real JEPA training step."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from scripts.train_availability_encoder import configure, teacher_target
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output = Path('artifacts/map_native_training_batch_20260912.json')
    if output.exists():
        raise ValueError('fresh report required')
    parent = Path('artifacts/map_native_query_history_complete_20260912.json')
    report = json.loads(parent.read_text())
    for name, expected in report['bindings'].items():
        if sha256(Path(name)) != expected:
            raise ValueError('changed input dependency')
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)), dict(np.load(root/'native_query_shards'/p.name))
    storage = RaggedNativeBatch(shards(), report['queries'])
    labels = np.load(root/'labels.npz')
    roles = np.load(root/'partition_7.npz')
    fit = np.flatnonzero(roles['fit'] & labels['paired'][:, 0])
    event_rows = np.unique(np.concatenate([storage.sequences[i] for i in fit]))
    events, _, _ = prepare_events(storage.event_values, event_rows)
    storage.input = np.column_stack([storage.input[:, :-1], events, storage.input[:, -1:]]).astype(np.float32)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    assets = load_release(Path('models'))
    model = assets.model
    expand_history_projection(model, 6)
    native_filter = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
    native_filter.load_state_dict(model.belief_filter.state_dict())
    model.belief_filter = native_filter
    model.belief_input_mode = 'observed_steps'
    model.value_delta_activation = torch.nn.Identity()
    parameters = configure(model, True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    col = assets.checkpoint['variables'].index('map')
    selected = fit[np.linspace(0, len(fit)-1, 4, dtype=int)]
    inputs = storage.batch(selected)
    result = model(*inputs)
    truth = torch.tensor((labels['future'][selected, 0]-assets.scaler.medians[col])/
                         assets.scaler.scales[col], dtype=torch.float32)
    loss = F.smooth_l1_loss(result['value'][:, col], truth) + _masked_latent_prediction_loss(
        result['predicted_target_latent'][:, col], teacher_target(model, inputs[1], truth, col),
        torch.ones(len(truth), dtype=torch.bool))
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    projection = model.belief_filter.observation_projection[0].weight
    if projection.grad is None or not torch.isfinite(projection.grad).all() or projection.grad.norm() == 0:
        raise ValueError('invalid history projection gradient')
    gradient = float(projection.grad.norm())
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    before = projection.detach().clone()
    optimizer.step()
    model.update_target_encoder()
    change = float((projection.detach()-before).norm())
    if not np.isfinite(change) or change <= 0:
        raise ValueError('history projection did not update')
    result = dict(complete=True, queries=report['queries'],
        event_rows=len(storage.input), max_history=storage.width,
        fitted_step_examples=len(selected), loss=float(loss.detach()),
        history_projection_gradient_norm=gradient, history_projection_update_norm=change,
        runtime_modified=False, model_saved=False,
        scope='one source-fit training step, not efficacy or a completed training experiment',
        bindings={str(p):sha256(p) for p in (parent, Path(__file__),
            Path('whole_body_jepa/ragged_native_batch.py'), root/'partition_7.npz')})
    output.write_text(json.dumps(result, indent=2)+'\n')
    print({k:v for k,v in result.items() if k != 'bindings'})


if __name__ == '__main__':
    main()
