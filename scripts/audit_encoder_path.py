"""Structural probes, not accuracy evidence, for the availability experiment."""
import json
from copy import deepcopy
from pathlib import Path

import torch

from scripts.train_availability_encoder import configure, teacher_target
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def probe(model, index):
    torch.manual_seed(1701)
    configure(model, True)
    n = model.n_state
    history = torch.zeros(4, 10, model.input_dim)
    history[:, :, :n] = torch.randn(4, 10, n) * .2
    history[:, :, n:2*n] = 1
    history[:, :, 3*n+1:-1] = 1
    history = history.requires_grad_()
    current = history[:, -1, :n].detach().clone().requires_grad_()
    mask = torch.ones_like(current)
    presence = torch.ones(4, model.module_count)
    horizon = torch.full((4,), 6.)
    output = model(history, current, mask, torch.zeros_like(current), presence, horizon)
    truth = current[:, index].detach() + torch.tensor([-.8, -.2, .3, .9])
    teacher = teacher_target(model, current, truth, index)
    losses = {
        'value': torch.nn.functional.smooth_l1_loss(output['value'][:, index], truth),
        'jepa': _masked_latent_prediction_loss(
            output['predicted_target_latent'][:, index], teacher,
            torch.ones(4, dtype=torch.bool)),
    }
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    result = {'model_class': type(model).__name__, 'gradients': {}}
    others = [i for i in range(n) if i != index]
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, [history, current] + [p for _, p in params],
                                        retain_graph=True, allow_unused=True)
        active = [key for (key, _), grad in zip(params, gradients[2:])
                  if grad is not None and bool(grad.abs().sum() > 0)]
        result['gradients'][name] = {
            'non_target_history_norm': float(gradients[0][:, :, others].norm()),
            'non_target_current_norm': float(gradients[1][:, others].norm()),
            'active_parameter_tensors': len(active),
            'variable_mixer_active': any(k.startswith('variable_mixer.') for k in active),
            'module_mixer_active': any(k.startswith('module_mixer.') for k in active),
            'belief_filter_active': any(k.startswith('belief_filter.') for k in active),
        }
    with torch.no_grad():
        context = output['context_latent']
        organs = output['organ_latents']
        base = model._predict_state_trajectory(context, organs, presence, horizon[:, None])['body']
        changed = model._predict_state_trajectory(context, organs + 10, presence, horizon[:, None])['body']
        result['separate_organ_argument_max_effect'] = float((changed-base).abs().max())
    result['teacher_parameters_trainable'] = any(
        p.requires_grad for name, p in model.named_parameters()
        if name.startswith(('target_state_encoder.', 'target_variable_mixer.')))
    return result


def main():
    torch.set_num_threads(2)
    assets = load_release(Path('models'))
    index = assets.checkpoint['variables'].index('glucose')
    root = Path('/private/tmp/encoder_actual_time_weights_20260910')
    report = dict(status='diagnostic_only', synthetic_inputs=True,
                  accuracy_test=False, runtime_modified=False, checkpoints=[])
    for path in sorted(root.glob('*_unbounded_*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = deepcopy(assets.model)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.value_delta_activation = torch.nn.Identity()
        model.belief_input_mode = 'observed_steps'
        report['checkpoints'].append(dict(checkpoint=path.name, sha256=sha256(path),
                                           **probe(model, index)))
    if len(report['checkpoints']) != 14:
        raise ValueError('expected fourteen actual-time identity checkpoints')
    output = Path('artifacts/encoder_path_20260910.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
