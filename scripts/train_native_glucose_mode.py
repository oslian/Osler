"""Test existing student dropout, retaining deterministic EMA and evaluation."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_glucose_mode_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    ref = json.loads(Path('artifacts/rate_glucose_jepa_20260911.json').read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('input changed')
    def load(name):
        with np.load(root / name) as f: return {k: f[k] for k in f.files}
    base, native, labels = load('body_arrays.npz'), load('native_tokens.npz'), load('multitarget_labels.npz')
    valid = load('labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid': 'subject_id'})
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; col = variables.index('glucose')
    median, scale = assets.scaler.medians[col], assets.scaler.scales[col]
    eligible = labels['paired'][:, col]; q['truth'] = labels['future'][:, col]; truth = (q.truth.to_numpy() - median) / scale
    arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
    report = dict(complete=False, status='candidate_only', runtime_modified=False, seeds=[], binding=dict(ref['binding']))
    for p in (Path(__file__), Path('NATIVE_GLUCOSE_MODE_PROTOCOL_20260911.md')): report['binding'][str(p)] = sha256(p)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed = entry['seed']; roles = {a: b[valid] & eligible for a, b in load(f'partition_{seed}.npz').items()}
        fit, tune = roles['fit'], roles['tune']
        pools = [np.flatnonzero(fit & q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit, 'subject_id']))]
        allowed = source_patient_support(base, base['query_rows'], q.subject_id, fit) > 0
        events, center, event_scale = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
        old = load(f'glucose_native_predictions_{seed}.npz')
        points = {'own_eval': old['own'], 'whole_eval': old['real'], 'persistence': old['persistence']}
        item = dict(seed=seed, arms={}, roles={})
        for arm, original in [('own_train', 'own'), ('whole_train', 'real')]:
            channels = allowed.copy()
            if original == 'own': channels[np.arange(len(channels)) != col] = False
            view = restrict_input_support(arrays, channels, assets.model.variable_module_ids.numpy()); e = events.copy()
            if original == 'own': e[:] = 0
            view = append_events(view, e); torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = deepcopy(assets.model); expand_history_projection(model, 6)
            nf = NativeEventFilter(model.input_dim, model.belief_filter.hidden); nf.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = nf
            model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
            # Reproduce the complete original predictions before changing training.
            checkpoint = torch.load(root / f'glucose_native_{original}_{seed}.pt', map_location='cpu', weights_only=False)
            initial = deepcopy(model.state_dict()); model.load_state_dict(checkpoint['state_dict'])
            np.testing.assert_allclose(predict(model, view, rows, col, median, scale), old[original], atol=1e-4, rtol=1e-6)
            model.load_state_dict(initial)
            params = configure(model, True); optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
            best_mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], predict(model, view, rows[tune], col, median, scale))
            best, selected, log = deepcopy(model.state_dict()), 0, []
            for epoch in range(32):
                model.train()
                assert model.variable_mixer.training and not model.target_variable_mixer.training
                for _ in range(16):
                    chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)])
                    inputs = batch(view, rows[chosen]); result = model(*inputs); y = torch.tensor(truth[chosen], dtype=torch.float32)
                    teacher = teacher_target(model, inputs[1], y, col)
                    loss = F.smooth_l1_loss(result['value'][:, col], y) + _masked_latent_prediction_loss(result['predicted_target_latent'][:, col], teacher, torch.ones(len(y), dtype=torch.bool))
                    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.)
                    optimizer.step(); model.update_target_encoder()
                mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], predict(model, view, rows[tune], col, median, scale))
                if mae < best_mae: best_mae, best, selected = mae, deepcopy(model.state_dict()), epoch + 1
                log.append(dict(epoch=epoch + 1, tune_mae=mae))
            model.load_state_dict(best); points[arm] = predict(model, view, rows, col, median, scale)
            path = root / f'glucose_mode_{arm}_{seed}.pt'
            if path.exists(): raise ValueError('checkpoint overwrite blocked')
            torch.save(dict(state_dict=best, variables=variables, allowed_variables=channels.tolist(), event_center=center.tolist(), event_scale=event_scale.tolist(), event_width=6, arm=arm, value_delta_activation='identity', belief_input_mode='observed_steps', status='candidate_only'), path)
            report['binding'][str(path)] = sha256(path); item['arms'][arm] = dict(selected_epoch=selected, tune_mae=best_mae, log=log)
            print('complete', seed, arm, selected, flush=True)
        for role in ('evaluation', 'hospital'):
            m = roles[role]; f = q.loc[m]
            contrasts = [('own_train', 'own_eval'), ('whole_train', 'whole_eval'), ('whole_train', 'own_train'), ('whole_train', 'persistence')]
            item['roles'][role] = dict(mae={a: patient_equal_mae(f, f.truth, p[m]) for a, p in points.items()}, comparisons={f'{a}_vs_{b}': paired(f, points[a][m], points[b][m], f'mode:{seed}:{role}:{a}:{b}', 2000) for a, b in contrasts})
        path = root / f'glucose_mode_predictions_{seed}.npz'; np.savez_compressed(path, **points)
        report['binding'][str(path)] = sha256(path); report['seeds'].append(item)
        (root / 'glucose_mode_progress.json').write_text(json.dumps(report, indent=2) + '\n')
    report['complete'] = True; output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
