"""Frozen structural audit of immediate organ-to-body observation reachability."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_organ_binding import ARMS, prepare, write_json
from whole_body_jepa.multisource_intervals import sha256


PROTOCOL = Path('RECONSTRUCTION_ORGAN_REACHABILITY_PROTOCOL_20261004.md')
OUTPUT = Path('artifacts/native_organ_reachability_20261004.json')


def check_path(model):
    layer = model.trajectory.layer
    parameter = next(layer.parameters())
    generator = torch.Generator().manual_seed(20261004)

    def noise(shape):
        return torch.randn(shape, generator=generator, dtype=parameter.dtype,
                           device=parameter.device)

    body = noise((1, layer.body_dim)).requires_grad_()
    organs = noise((1, layer.organ_count, layer.organ_dim)).requires_grad_()
    presence = torch.ones_like(organs[..., 0])
    initial = layer.initialize(noise(body.shape), noise(organs.shape), presence)
    zero = body.new_zeros(1)
    state = layer.update(initial, body, organs, presence, zero)
    changed = layer.update(initial, body, organs + .25 * noise(organs.shape), presence, zero)
    output = model.readout(state.body)
    body_gradient, organ_gradient = torch.autograd.grad(
        output.sum(), (body, organs), allow_unused=True)

    def difference(a, b):
        return float((a.detach() - b.detach()).abs().max())

    one = body.new_ones(1)
    later = layer.advance(state, one)
    later_changed = layer.advance(changed, one)
    # A fresh graph keeps the positive-time control independent of the first grad call.
    repeated = layer.update(initial, body, organs, presence, zero)
    future = model.readout(layer.advance(repeated, one).body)
    future_gradient = torch.autograd.grad(future.sum(), organs, allow_unused=True)[0]
    result = dict(
        immediate_body_difference=difference(state.body, changed.body),
        immediate_organ_difference=difference(state.organs, changed.organs),
        immediate_prediction_difference=difference(output, model.readout(changed.body)),
        immediate_body_gradient=float(body_gradient.detach().norm()) if body_gradient is not None else 0.,
        immediate_organ_path_exists=organ_gradient is not None,
        immediate_organ_gradient=float(organ_gradient.detach().norm()) if organ_gradient is not None else 0.,
        one_hour_body_difference=difference(later.body, later_changed.body),
        one_hour_prediction_difference=difference(model.readout(later.body), model.readout(later_changed.body)),
        one_hour_organ_path_exists=future_gradient is not None,
        one_hour_organ_gradient=float(future_gradient.detach().norm()) if future_gradient is not None else 0.,
    )
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite structural diagnostic')
    return result


def timing_support(examples, bound, variables):
    if not examples or len(examples) != len(bound):
        raise ValueError('Aligned nonempty contexts required')
    results = {v: dict(contexts=0, zero_gap=0, single_event_zero_gap=0,
                       zero_gap_with_changed_last_organ=0) for v in variables}
    gaps = []
    for e, candidate in zip(examples, bound):
        times = e['times']
        if (not len(times) or not torch.isfinite(times).all()
                or (times[1:] <= times[:-1]).any()
                or candidate.shape != e['organs'].shape
                or not torch.isfinite(candidate).all()
                or not torch.isfinite(e['organs']).all()):
            raise ValueError('Invalid context or binding proposal')
        gap = float(e['anchor']) - float(times[-1])
        if not np.isfinite(gap) or gap < 0:
            raise ValueError('Nonfinite or negative anchor gap')
        gaps.append(gap)
        row = results[variables[e['target']]]
        row['contexts'] += 1
        if gap == 0:
            row['zero_gap'] += 1
            row['single_event_zero_gap'] += int(len(times) == 1)
            present = e['presence'][-1] > 0
            row['zero_gap_with_changed_last_organ'] += int(
                bool(((candidate[-1]-e['organs'][-1])[present] != 0).any()))
    total = {key: sum(r[key] for r in results.values()) for key in next(iter(results.values()))}
    return dict(total=total, targets=results, minimum_gap_hours=min(gaps),
                median_gap_hours=float(np.median(gaps)), maximum_gap_hours=max(gaps))


def main():
    if OUTPUT.exists():
        raise ValueError('Completed diagnostic exists; do not repeat')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    bindings = {str(p): sha256(p) for p in (Path(__file__), PROTOCOL,
        Path('whole_body_jepa/physiology_state.py'), Path('scripts/train_organ_binding.py'))}
    folds = {}
    for fold in range(5):
        variables, examples, bound, original, contract, _, dest = prepare(fold)
        report_path = Path(f'artifacts/native_organ_binding_fold{fold}_20261004.json')
        report = json.loads(report_path.read_text())
        if report['contract'] != contract:
            raise ValueError('E27 report contract changed')
        bindings[str(report_path)] = sha256(report_path)
        models = {}
        for arm in ARMS:
            path = dest/f'{arm}.pt'
            if sha256(path) != report['fits'][arm]['sha256']:
                raise ValueError('E27 checkpoint changed')
            bindings[str(path)] = sha256(path)
            model = deepcopy(original).eval()
            model.load_state_dict(torch.load(path, weights_only=False), strict=True)
            models[arm] = check_path(model)
        folds[str(fold)] = dict(models=models,
            support={role: timing_support(rows, bound[role], variables) for role, rows in examples.items()})
        print('E28 frozen path and timing checked', fold, flush=True)
    write_json(OUTPUT, dict(folds=folds, bindings=bindings, runtime_modified=False,
        promotion_allowed=False, outcome_scoring=False,
        limitation='Branch-specific synthetic sensitivity and context exposure, not error attribution; body bypass remains.'))
    print('E28 diagnostic complete', flush=True)


if __name__ == '__main__':
    main()
