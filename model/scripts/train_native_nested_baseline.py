"""Matched own/full JEPA training on MAP availability-native queries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.train_availability_encoder import configure, teacher_target
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_joint_jepa import _masked_latent_prediction_loss


def predict(model, storage, rows, col, median, scale):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            value = model(*storage.batch(rows[start:start+128]))['value'][:, col]
            result.append(value.numpy()*scale+median)
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7])
    args = parser.parse_args()
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_training_transforms_20260912.json')
    for name, expected in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name)) != expected:
            raise ValueError('changed training dependency')
    verified = Path('artifacts/map_native_query_history_complete_20260912.json')
    for name, expected in json.loads(verified.read_text())['bindings'].items():
        if sha256(Path(name)) != expected:
            raise ValueError('changed input dependency')
    q = pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels = dict(np.load(root/'labels.npz'))
    eligible = labels['paired'][:, 0]
    q['truth'] = labels['future'][:, 0]
    transforms = dict(np.load(root/'neural_transforms.npz'))
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)), dict(np.load(root/'native_query_shards'/p.name))
    raw = RaggedNativeBatch(shards(), len(q))
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    col = variables.index('map')
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    truth = (q.truth.to_numpy()-median)/scale
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        output = Path(f'artifacts/map_nested_baseline_{seed}_20260913.json')
        if output.exists():
            raise ValueError('fresh seed output required')
        roles = dict(np.load(root/f'nested_partition_{seed}.npz'))
        fit = roles['fit'] & eligible
        tune_rows = np.flatnonzero(roles['tune'] & eligible)
        fit_rows = np.flatnonzero(fit)
        pools = [np.asarray(g.index) for _,g in q.loc[fit].groupby('subject_id', sort=True)]
        points = {'persistence':labels['current'][:, 0]}
        item = dict(seed=seed, arms={}, roles={})
        paths = [parent, verified, Path('artifacts/native_nested_partition_20260913.json'), Path('NATIVE_NESTED_BASELINE_PROTOCOL_20260913.md'), Path(__file__), root/f'nested_partition_{seed}.npz',
            Path('whole_body_jepa/ragged_native_batch.py'), Path('whole_body_jepa/ragged_native_transform.py')]
        for arm in ('own',):
            checkpoint = root/f'nested_baseline_{arm}_{seed}.pt'
            if checkpoint.exists():
                raise ValueError('checkpoint overwrite blocked')
            allowed = transforms[f'{seed}_allowed'].copy()
            if arm == 'own':
                allowed[np.arange(len(allowed)) != col] = False
            storage, ec, es = transformed_storage(raw,
                transforms[f'{seed}_numeric_center'], transforms[f'{seed}_numeric_scale'],
                allowed, assets.model.variable_module_ids.numpy(), fit_rows, own=arm=='own')
            # Own-only input removes all events; fit-dependent event transforms are not used.
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = deepcopy(assets.model)
            expand_history_projection(model, 6)
            native = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
            native.load_state_dict(model.belief_filter.state_dict())
            model.belief_filter = native
            model.belief_input_mode = 'observed_steps'
            model.value_delta_activation = torch.nn.Identity()
            params = configure(model, True)
            optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
            best_mae = patient_equal_mae(q.iloc[tune_rows], q.truth.iloc[tune_rows],
                predict(model, storage, tune_rows, col, median, scale))
            best, selected, log = deepcopy(model.state_dict()), 0, []
            print('initial', seed, arm, best_mae, flush=True)
            for epoch in range(32):
                for _ in range(16):
                    chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)])
                    x = storage.batch(chosen)
                    result = model(*x)
                    y = torch.tensor(truth[chosen], dtype=torch.float32)
                    loss = F.smooth_l1_loss(result['value'][:, col], y) + _masked_latent_prediction_loss(
                        result['predicted_target_latent'][:, col], teacher_target(model, x[1], y, col),
                        torch.ones(len(y), dtype=torch.bool))
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                    optimizer.step()
                    model.update_target_encoder()
                mae = patient_equal_mae(q.iloc[tune_rows], q.truth.iloc[tune_rows],
                    predict(model, storage, tune_rows, col, median, scale))
                if mae < best_mae:
                    best_mae, best, selected = mae, deepcopy(model.state_dict()), epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
                print('epoch', seed, arm, epoch+1, mae, flush=True)
            model.load_state_dict(best)
            pred = np.full(len(q), np.nan)
            evaluation_rows = np.flatnonzero(eligible)
            pred[evaluation_rows] = predict(model, storage, evaluation_rows, col, median, scale)
            points[arm] = pred
            torch.save(dict(state_dict=best, variables=variables, allowed_variables=allowed.tolist(),
                numeric_center=transforms[f'{seed}_numeric_center'].tolist(),
                numeric_scale=transforms[f'{seed}_numeric_scale'].tolist(),
                event_center=ec.tolist(), event_scale=es.tolist(), history_contract=arm,
                status='candidate_only', selected_epoch=selected), checkpoint)
            paths.append(checkpoint)
            item['arms'][arm] = dict(selected_epoch=selected, log=log, best_tune_mae=best_mae)
            del storage, model, best, optimizer, native
        for role in ('evaluation', 'hospital'):
            mask = roles[role] & eligible
            frame = q.loc[mask]
            p = {k:v[mask] for k,v in points.items()}
            refs = {r:paired_model_errors(frame,p,r,seed=20260912,repetitions=20000)
                    for r in ('own','persistence')}
            item['roles'][role] = dict(mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                comparisons={'own_vs_persistence':refs['persistence']['own']})
        prediction_path = root/f'nested_baseline_predictions_{seed}.npz'
        np.savez_compressed(prediction_path, **points)
        paths.append(prediction_path)
        output.write_text(json.dumps(dict(complete=True,status='candidate_only',runtime_modified=False,
            **item,bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
        print('completed', seed, json.dumps(item['roles']), flush=True)


if __name__ == '__main__':
    main()

