"""Native glucose training with fixed anchor-observed output retention."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from scripts.audit_native_glucose_retention import all_points
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support, source_patient_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def retention_loss(prediction, teacher, mask):
    per_element = F.smooth_l1_loss(prediction, teacher, reduction='none')
    count = mask.sum(0); valid = count > 0
    if not bool(valid.any()): return prediction.sum() * 0.
    return ((per_element * mask).sum(0) / count.clamp_min(1))[valid].mean()


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_retention_repair_20260911.json')
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
    medians, scales = assets.scaler.medians, assets.scaler.scales
    eligible = labels['paired'][:, col]; q['truth'] = labels['future'][:, col]; truth = (q.truth.to_numpy() - medians[col]) / scales[col]
    arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
    report = dict(complete=False, runtime_modified=False, status='candidate_only', seeds=[], binding=dict(ref['binding']))
    for p in (Path(__file__), Path('NATIVE_RETENTION_REPAIR_PROTOCOL_20260911.md'), Path('models/whole_body_joint_jepa.pt'), Path('scripts/audit_native_glucose_retention.py')):
        report['binding'][str(p)] = sha256(p)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed = entry['seed']; roles = {a: b[valid] & eligible for a, b in load(f'partition_{seed}.npz').items()}
        fit, tune = roles['fit'], roles['tune']
        pools = [np.flatnonzero(fit & q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit, 'subject_id']))]
        checkpoint = torch.load(root / f'glucose_native_real_{seed}.pt', map_location='cpu', weights_only=False)
        allowed = np.asarray(checkpoint['allowed_variables'])
        events, center, event_scale = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
        view = append_events(restrict_input_support(arrays, allowed, assets.model.variable_module_ids.numpy()), events)
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = deepcopy(assets.model); expand_history_projection(model, 6)
        nf = NativeEventFilter(model.input_dim, model.belief_filter.hidden); nf.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = nf
        model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
        initial_state = deepcopy(model.state_dict())
        initial = all_points(model, view, rows, medians, scales)
        teacher_points = torch.tensor((initial - medians) / scales, dtype=torch.float32)
        supported = source_patient_support(base, base['query_rows'], q.subject_id, fit) >= 30
        supported[col] = False
        retain_mask = torch.tensor((view['mask'][rows] > 0) & supported[None, :])
        model.load_state_dict(checkpoint['state_dict']); original = all_points(model, view, rows, medians, scales)
        old = load(f'glucose_native_predictions_{seed}.npz')
        np.testing.assert_allclose(original[:, col], old['real'], atol=1e-4, rtol=1e-6)
        model.load_state_dict(initial_state); params = configure(model, True)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        best_mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], initial[tune, col]); best = deepcopy(initial_state); selected = 0; log = []
        for epoch in range(32):
            losses = []
            for _ in range(16):
                chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)])
                inputs = batch(view, rows[chosen]); result = model(*inputs); y = torch.tensor(truth[chosen], dtype=torch.float32)
                teacher = teacher_target(model, inputs[1], y, col)
                primary = F.smooth_l1_loss(result['value'][:, col], y) + _masked_latent_prediction_loss(result['predicted_target_latent'][:, col], teacher, torch.ones(len(y), dtype=torch.bool))
                retention = retention_loss(result['value'], teacher_points[chosen], retain_mask[chosen])
                loss = primary + retention
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step(); model.update_target_encoder(); losses.append(float(retention.detach()))
            mae = patient_equal_mae(q.loc[tune], q.loc[tune, 'truth'], predict(model, view, rows[tune], col, medians[col], scales[col]))
            if mae < best_mae: best_mae, best, selected = mae, deepcopy(model.state_dict()), epoch + 1
            log.append(dict(epoch=epoch+1, tune_mae=mae, retention_loss=float(np.mean(losses))))
        model.load_state_dict(best); final = all_points(model, view, rows, medians, scales)
        path = root / f'glucose_retention_{seed}.pt'
        if path.exists(): raise ValueError('checkpoint overwrite blocked')
        torch.save(dict(state_dict=best, allowed_variables=allowed.tolist(), variables=variables, event_center=center.tolist(), event_scale=event_scale.tolist(), value_delta_activation='identity', belief_input_mode='observed_steps', event_width=6, retention_weight=1., status='candidate_only'), path)
        report['binding'][str(path)] = sha256(path)
        item = dict(seed=seed, selected_epoch=selected, log=log, retained_variables=[v for v, s in zip(variables, supported) if s], roles={})
        for role in ('evaluation', 'hospital'):
            result = {}
            for j, v in enumerate(variables):
                m = roles[role] & labels['paired'][:, j]; f = q.loc[m].copy(); f['truth'] = labels['future'][m, j]
                if f.subject_id.nunique() < 30: continue
                predictions = dict(final=final[m,j], original=original[m,j], initial=initial[m,j], persistence=labels['current'][m,j])
                if j == col: predictions['own'] = old['own'][m]
                result[v] = dict(patients=int(f.subject_id.nunique()), mae={a:patient_equal_mae(f,f.truth,p) for a,p in predictions.items()}, comparisons={f'final_vs_{a}': paired(f,predictions['final'],p,f'retain:{seed}:{role}:{v}:{a}',2000) for a,p in predictions.items() if a != 'final'})
            item['roles'][role] = result
        path = root / f'glucose_retention_predictions_{seed}.npz'; np.savez_compressed(path, final=final, original=original, initial=initial)
        report['binding'][str(path)] = sha256(path); report['seeds'].append(item)
        (root / 'glucose_retention_progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print('complete',seed,selected,flush=True)
    report['complete'] = True; output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
