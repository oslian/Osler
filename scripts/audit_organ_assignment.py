"""Audit value/identity collisions without fitting or changing serving models."""
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual

ROOT = Path('data/research_recovery/additional_development_domains_20260921')
PROTOCOL = Path('RECONSTRUCTION_ORGAN_ASSIGNMENT_PROTOCOL_20261004.md')
OUTPUT = Path('artifacts/native_organ_assignment_20261004.json')
TOLERANCE = 1e-10
VALUES = ((-1., 1.), (0., 1.), (1., 2.))


def pools(model, variables, tokens):
    """Independent native formula and parameter-free, research-only binding."""
    n = len(variables)
    mask = tokens[:, n:2*n]
    observed = mask.bool()
    current = torch.where(observed, tokens[:, :n], torch.zeros_like(mask))
    ages = torch.where(observed, tokens[:, 2*n:3*n], torch.full_like(mask, 7))
    projected = model.token_projection(torch.stack((current, mask, ages), -1))
    identity = model.variable_embedding + model.module_token_embedding(model.variable_module_ids)
    weights = mask[:, None] * mask.new_tensor([
        [v in members for v in variables] for members in CANONICAL_ORGAN_VARIABLES.values()
    ])[None]
    count = weights.sum(-1, keepdim=True).clamp_min(1)
    combined = projected + identity[None]
    native = torch.einsum('bnd,bon->bod', combined, weights) / count
    bound = torch.einsum('bnd,bon->bod', F.silu(combined), weights) / count
    return native, bound


def cases(model, variables):
    n = len(variables)
    original, swapped, metadata = [], [], []
    for organ, members in CANONICAL_ORGAN_VARIABLES.items():
        columns = [i for i, name in enumerate(variables) if name in members]
        for left, right in itertools.combinations(columns, 2):
            for scope in ('pair', 'full_organ'):
                for value_index, (a, b) in enumerate(VALUES):
                    token = model.variable_embedding.new_zeros(model.input_dim)
                    token[2*n:3*n] = 7
                    active = [left, right] if scope == 'pair' else columns
                    token[[n+i for i in active]] = 1
                    token[[2*n+i for i in active]] = 0
                    token[left], token[right] = a, b
                    changed = token.clone()
                    changed[left], changed[right] = b, a
                    original.append(token)
                    swapped.append(changed)
                    metadata.append(dict(organ=organ, left=variables[left], right=variables[right],
                                         scope=scope, value_case=value_index))
    if not metadata:
        raise ValueError('No same-organ variable pairs')
    return torch.stack(original), torch.stack(swapped), metadata


def evolved_organs(trajectory, body, organs, presence):
    layer = trajectory.layer
    if layer.body_feedback:
        raise ValueError('Protocol requires the original no-body-feedback organ path')
    initial = layer.initialize(torch.zeros_like(body), torch.zeros_like(organs),
                               torch.zeros_like(presence))
    zero = body.new_zeros(len(body))
    state = layer.update(initial, body, organs, presence, zero)
    return state.organs, layer.advance(state, zero + 6).organs


def audit(trajectory, chunk_size=64):
    if trajectory.training:
        raise ValueError('Frozen evaluation required')
    model, variables = trajectory.jepa, trajectory.variables
    original, swapped, metadata = cases(model, variables)
    records = []
    with torch.inference_mode():
        for start in range(0, len(metadata), chunk_size):
            x, y = original[start:start+chunk_size], swapped[start:start+chunk_size]
            b0, o0, p0 = encode_native_organ_observations(model, variables, x)
            b1, o1, p1 = encode_native_organ_observations(model, variables, y)
            manual0, bound0 = pools(model, variables, x)
            manual1, bound1 = pools(model, variables, y)
            torch.testing.assert_close(p0, p1, rtol=0, atol=0)
            torch.testing.assert_close(o0, manual0, rtol=0, atol=TOLERANCE)
            torch.testing.assert_close(o1, manual1, rtol=0, atol=TOLERANCE)
            post0, future0 = evolved_organs(trajectory, b0, o0, p0)
            post1, future1 = evolved_organs(trajectory, b1, o1, p1)
            new_post0, new_future0 = evolved_organs(trajectory, b0, bound0, p0)
            new_post1, new_future1 = evolved_organs(trajectory, b1, bound1, p1)
            contrasts = {
                'body_proposal': (b0, b1), 'organ_proposal': (o0, o1),
                'organ_post_update': (post0, post1), 'organ_after_6h': (future0, future1),
                'binding_proposal': (bound0, bound1),
                'binding_post_update': (new_post0, new_post1),
                'binding_after_6h': (new_future0, new_future1),
                'formula_parity': (o0, manual0),
            }
            deltas = {}
            for name, (a, b) in contrasts.items():
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite diagnostic state')
                deltas[name] = (a-b).abs().flatten(1).max(1).values.tolist()
            for index, meta in enumerate(metadata[start:start+len(x)]):
                records.append({**meta, **{k: v[index] for k, v in deltas.items()}})
    return records


def summarize(records):
    names = ('body_proposal', 'organ_proposal', 'organ_post_update', 'organ_after_6h',
             'binding_proposal', 'binding_post_update', 'binding_after_6h', 'formula_parity')
    return dict(cases=len(records), metrics={name: dict(
        separated=sum(row[name] > TOLERANCE for row in records),
        min_difference=min(row[name] for row in records),
        max_difference=max(row[name] for row in records)) for name in names})


def main():
    if OUTPUT.exists():
        raise ValueError('Fresh aggregate output required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    schema_path = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(schema_path, map_location='cpu', weights_only=False)
    variables = schema['variables']
    if len(variables) != 53:
        raise ValueError('Expected original 53-variable schema')
    paths = [Path(__file__), PROTOCOL, schema_path, Path('scripts/train_encoder_crossfit.py'),
             Path('whole_body_jepa/native_organ_observations.py'),
             Path('whole_body_jepa/native_organ_state_residual.py'),
             Path('whole_body_jepa/physiology_state.py'), Path('whole_body_joint_jepa.py'),
             Path('osler_jepa/organ_schema.py')]
    results = []
    for fold in range(5):
        source_fold = 0 if fold == 2 else fold
        path = ROOT / f'monitored_continuation_20260929/fold{source_fold}_real_epoch10.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint['fold'], checkpoint['epoch'], checkpoint['arm']) != (source_fold, 10, 'real'):
            raise ValueError('Incorrect frozen source')
        model = NativeBodyReconstruction(NativeOrganStateResidual(
            fresh_encoder(schema), variables, ('map',))).eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.double()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        records = audit(model.trajectory)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        result = dict(fold=fold, source_fold=source_fold, **summarize(records), records=records)
        results.append(result)
        paths.append(path)
        print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    report = dict(protocol=str(PROTOCOL), tolerance=TOLERANCE, dtype='float64',
                  data='synthetic normalized observations, frozen trained weights',
                  unique_checkpoints=4, fold_mappings=5, results=results,
                  bindings={str(p): sha256(p) for p in paths},
                  runtime_modified=False, training_performed=False, promotion_allowed=False,
                  limitation='Path-specific assignment invariance; no real-outcome efficacy or global root-cause proof.')
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
