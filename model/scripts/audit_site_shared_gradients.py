"""Source-fit gradient diagnostic; no weight update or evaluation selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_reconstruction_source_fit import durable
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    out = Path('artifacts/native_reconstruction_site_gradients_20260917.json')
    if out.exists():
        raise ValueError('existing diagnostic')
    report_path = Path('artifacts/native_reconstruction_site_training_epoch8_20260917.json')
    report = json.loads(report_path.read_text())
    for p, digest in report['bindings'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('changed training dependency')
    split = Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    roles = torch.load(split, weights_only=False)['roles']
    root = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    index_path = root / 'expanded_reconstruction_20260913/index.pt'
    index = torch.load(index_path, weights_only=False)
    variables = index['variables']
    targets = report['source_targets']
    examples = {t: [] for t in targets}
    values = {t: [] for t in targets}
    for shard in index['shards']:
        if roles[shard['patient']] != 'fit':
            continue
        path = durable(shard['path'])
        if sha256(path) != shard['sha256']:
            raise ValueError('changed shard')
        data = torch.load(path, weights_only=False)
        if data['patient'] != shard['patient']:
            raise ValueError('wrong patient')
        for c, e in data['samples'].items():
            target = variables[c]
            if target not in examples:
                continue
            key = hashlib.sha256(f"site-gradient-v1:{data['patient']}:{target}".encode()).hexdigest()
            examples[target].append((key, e))
            values[target].append(e['truth_normalized'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = load_release(Path('models'))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    checkpoint = Path('data/research_recovery/site_reconstruction_training_20260917/real.pt')
    ck = torch.load(checkpoint, weights_only=False)
    if ck['bindings'] != report['bindings'] or ck['epoch'] != 8:
        raise ValueError('checkpoint mismatch')
    model.load_state_dict(ck['best_state_dict'])
    # Scope is the persistent physiology layer, not the encoder or target readout.
    named = list(model.trajectory.layer.named_parameters())
    parameters = [p for _, p in named]
    gradients = []
    rows = {}
    for target in targets:
        samples = [e for _, e in sorted(examples[target], key=lambda x: x[0])[:8]]
        model.zero_grad(set_to_none=True)
        losses = []
        for e in samples:
            prediction = model(input_tokens(e, 'real', len(variables)), e['event_hours'], e['anchor_hour'], target)
            loss = torch.nn.functional.smooth_l1_loss(prediction, prediction.new_tensor([e['truth_normalized']]))
            (loss / len(samples)).backward()
            losses.append(float(loss.detach()))
        gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters]).detach().double()
        if not torch.isfinite(gradient).all():
            raise ValueError('nonfinite gradient')
        gradients.append(gradient)
        x = np.asarray(values[target])
        rows[target] = dict(sampled_patients=len(samples), available_fit_patients=len(x),
                            fit_mad=float(np.median(abs(x-np.median(x)))),
                            sampled_smooth_l1=float(np.mean(losses)), shared_gradient_norm=float(gradient.norm()))
        print('source gradient', target, flush=True)
    matrix = torch.stack(gradients)
    mean = matrix.mean(0)
    for i, target in enumerate(targets):
        other = (matrix.sum(0)-matrix[i])/(len(targets)-1)
        norm = float(matrix[i].norm())
        denominator = norm * float(other.norm())
        rows[target].update(
            cosine_with_other_targets=None if denominator == 0 else float(matrix[i].dot(other))/denominator,
            local_loss_derivative_under_mean_descent=-float(matrix[i].dot(mean)))
    dependencies = [Path(__file__), report_path, split, index_path, checkpoint]
    out.write_text(json.dumps(dict(
        complete=True, targets=rows, parameter_names=[n for n, _ in named],
        parameter_count=sum(p.numel() for p in parameters), selected_epoch=ck['selected_epoch'],
        bindings={str(p): sha256(p) for p in dependencies},
        limitation='Eight deterministic source-fit patients per target at one selected endpoint. '
        'Persistent-layer raw mean gradients only; not AdamW updates, clipping, all epochs, '
        'full-population gradients, encoder gradients or evidence of causal generalization repair.',
        runtime_modified=False, promotion_allowed=False), indent=2)+'\n')
    print('complete', out, flush=True)


if __name__ == '__main__':
    main()
