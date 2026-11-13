"""Real source-event JEPA -> anatomical proposals -> persistent replay smoke audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES


def main():
    output = Path('artifacts/native_organ_replay_bridge_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    q = pd.read_parquet(root/'queries.parquet')
    paths = [Path(__file__), Path('whole_body_jepa/native_organ_observations.py'),
        Path('whole_body_jepa/physiology_state.py'), Path('osler_jepa/organ_schema.py')]
    clocks = [np.array([np.nan])]
    def shards():
        for index, p in enumerate(sorted((root/'input_shards').glob('shard_*.npz'))):
            report = Path(f'artifacts/map_unclipped_clock_{index:03d}_20260913.json')
            for name, h in json.loads(report.read_text())['bindings'].items():
                if sha256(Path(name)) != h: raise ValueError('changed raw clock dependency')
            paths.append(report)
            native = dict(np.load(root/'unclipped_clock_shards'/p.name))
            clocks.append(native['event_hours'])
            yield dict(np.load(p)), native
    raw = RaggedNativeBatch(shards(), len(q)); hours = np.concatenate(clocks)
    transforms = dict(np.load(root/'neural_transforms.npz'))
    fit = np.load(root/'partition_7.npz')['fit']
    labels = np.load(root/'labels.npz')['paired'][:, 0]
    assets = load_release(Path('models')); model = assets.model
    storage, _, _ = transformed_storage(raw, transforms['7_numeric_center'],
        transforms['7_numeric_scale'], transforms['7_allowed'],
        model.variable_module_ids.numpy(), np.flatnonzero(fit & labels), own=False)
    expand_history_projection(model, 6); model.belief_input_mode = 'observed_steps'; model.eval()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.manual_seed(7)
    organs = tuple(CANONICAL_ORGAN_VARIABLES)
    layer = PersistentPhysiologyStateLayer(body_dim=model.latent, organ_dim=model.token_dim,
        organ_count=len(organs), adjacency=torch.eye(len(organs)), body_feedback=False)
    rows = q.loc[fit].groupby('uniquepid', sort=True).head(1).index[:8]
    results = []
    for row in rows:
        indices = storage.sequences[row]
        if len(indices) == 0: continue
        time = hours[indices]
        if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
            raise ValueError('ungrouped/nonfinite/nonmonotone available times')
        x = torch.from_numpy(storage.input[indices].copy()); x[:, -7:-1] = 0
        model.zero_grad(set_to_none=True); layer.zero_grad(set_to_none=True)
        body, observed_organs, presence = encode_native_organ_observations(model,
            assets.checkpoint['variables'], x)
        initial = layer.initialize(torch.zeros(1, model.latent),
            torch.zeros(1, len(organs), model.token_dim), torch.zeros(1, len(organs)))
        result = layer.replay_observations(initial, body[None], observed_organs[None],
            presence[None], torch.from_numpy(time-time[0])[None].float())
        objective = result.body.square().mean()+result.organs.square().mean()
        objective.backward()
        token_grad = float(model.token_projection[0].weight.grad.norm())
        update_grad = float(layer.organ_update_gate[0].weight.grad.norm())
        if not torch.isfinite(objective) or not np.isfinite([token_grad, update_grad]).all():
            raise ValueError('nonfinite integrated path')
        results.append(dict(events=len(indices), supported_organs=int(presence.any(0).sum()),
            elapsed_hours=float(result.elapsed_hours[0]),
            token_gradient_norm=token_grad, organ_update_gradient_norm=update_grad))
        print('completed source stream', len(results), flush=True)
    output.write_text(json.dumps(dict(complete=True, role='source_fit_integration_only',
        organ_names=organs, results=results, treatment_context_used=False,
        objective='state squared norm only: gradient smoke check, not future supervision',
        training_performed=False, runtime_modified=False, promotion=False,
        bindings={str(p):sha256(p) for p in paths}), indent=2)+'\n')


if __name__ == '__main__': main()
