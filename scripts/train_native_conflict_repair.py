"""Matched old/replayed native glucose training on identical clean outcomes."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 31, 43, 59, 71, 83])
    args = parser.parse_args()
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_conflict_repair_training_20260911.json')
    if output.exists():
        raise ValueError('fresh report required')
    reference = json.loads(Path('artifacts/native_conflict_replay_tokens_20260911.json').read_text())
    for name, digest in reference['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('input dependency changed')
    old_reference = json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
    for name in ('native_tokens.npz', 'native_events.parquet', 'native_schedule.npz'):
        path = root/name
        if sha256(path) != old_reference['binding'][str(path)]:
            raise ValueError('original native benchmark changed')
    partitions = {e['seed']: e['partition_sha256'] for e in
        json.loads(Path('artifacts/rate_anchored_potassium_20260911.json').read_text())['seeds']}
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']
    col = variables.index('glucose'); median = assets.scaler.medians[col]; scale = assets.scaler.scales[col]
    labels = dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    valid = np.load(root/'labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    eligible = labels['paired'][:, col]
    q['truth'] = labels['future'][:, col]; truth = (q.truth.to_numpy()-median)/scale
    inputs = {}
    bindings = dict(reference['binding'])
    for path in (root/'queries.parquet', root/'labels.npz'):
        bindings[str(path)] = sha256(path)
    for mode, base_name, native_name in [('old', 'body_arrays.npz', 'native_tokens.npz'),
            ('clean', 'body_arrays_conflict_checked.npz', 'native_tokens_conflict_replay.npz')]:
        base = dict(np.load(root/base_name)); native = dict(np.load(root/native_name))
        np.testing.assert_allclose(base['current'][base['query_rows'][eligible], col]*scale+median,
                                   labels['current'][eligible, col], atol=1e-4, rtol=1e-6)
        arrays, rows, values = native_arrays(base, native, len(variables), base['presence'].shape[1])
        inputs[mode] = (base, arrays, rows, values)
        for p in (root/base_name, root/native_name): bindings[str(p)] = sha256(p)
    bindings[str(Path(__file__))] = sha256(Path(__file__))
    report = dict(complete=False, status='candidate_only', runtime_modified=False,
        contract='same clean fit/tune/evaluation labels; input replay comparison', seeds=[], binding=bindings)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        path = root/f'partition_{seed}.npz'
        if sha256(path) != partitions[seed]: raise ValueError('partition changed')
        report['binding'][str(path)] = sha256(path)
        roles = {a:b[valid]&eligible for a,b in dict(np.load(path)).items()}
        fit, tune = roles['fit'], roles['tune']
        pools = [np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        points = {'persistence': labels['current'][:, col]}; item = dict(seed=seed, arms={}, roles={})
        for mode, (base, arrays, rows, values) in inputs.items():
            allowed = source_patient_support(base, base['query_rows'], q.subject_id, fit)>0
            events, center, event_scale = prepare_events(values, np.unique(arrays['history'][rows[fit]]))
            for context in ('own', 'full'):
                arm = f'{mode}_{context}'; path = root/f'conflict_repair_{arm}_{seed}.pt'
                if path.exists(): raise ValueError('checkpoint overwrite blocked')
                channels = allowed.copy(); e = events.copy()
                if context == 'own':
                    channels[np.arange(len(channels)) != col] = False; e[:] = 0
                view = restrict_input_support(arrays, channels, assets.model.variable_module_ids.numpy())
                view = append_events(view, e); torch.manual_seed(seed); rng = np.random.default_rng(seed)
                model = deepcopy(assets.model); expand_history_projection(model, e.shape[1])
                nf = NativeEventFilter(model.input_dim, model.belief_filter.hidden)
                nf.load_state_dict(model.belief_filter.state_dict()); model.belief_filter = nf
                model.value_delta_activation = torch.nn.Identity(); model.belief_input_mode = 'observed_steps'
                params = configure(model, True); optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                best_mae = patient_equal_mae(q.loc[tune], q.loc[tune,'truth'], predict(model,view,rows[tune],col,median,scale))
                best = deepcopy(model.state_dict()); selected = 0; log = []
                for epoch in range(32):
                    for _ in range(16):
                        chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools), size=32)])
                        x = batch(view, rows[chosen]); result = model(*x)
                        y = torch.tensor(truth[chosen], dtype=torch.float32)
                        teacher = teacher_target(model, x[1], y, col)
                        loss = F.smooth_l1_loss(result['value'][:,col], y)+_masked_latent_prediction_loss(
                            result['predicted_target_latent'][:,col], teacher, torch.ones(len(y),dtype=torch.bool))
                        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step(); model.update_target_encoder()
                    mae = patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
                    if mae < best_mae: best_mae,best,selected = mae,deepcopy(model.state_dict()),epoch+1
                    log.append(dict(epoch=epoch+1,tune_mae=mae))
                model.load_state_dict(best); points[arm] = predict(model,view,rows,col,median,scale)
                torch.save(dict(state_dict=best, variables=variables, allowed_variables=channels.tolist(),
                    event_center=center.tolist(),event_scale=event_scale.tolist(),event_width=6,
                    value_delta_activation='identity',belief_input_mode='observed_steps',native_history=True,
                    history_contract=mode,status='candidate_only'),path)
                report['binding'][str(path)] = sha256(path)
                item['arms'][arm] = dict(selected_epoch=selected,tune_mae=best_mae,log=log)
                print('complete',seed,arm,selected,flush=True)
        for role in ('evaluation','hospital'):
            m=roles[role]; f=q.loc[m]
            comparisons=[('clean_full','old_full'),('clean_own','old_own'),('clean_full','clean_own'),
                         ('old_full','old_own'),('clean_full','persistence'),('clean_own','persistence')]
            item['roles'][role]=dict(patients=int(f.subject_id.nunique()),
                mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'conflict-{seed}-{role}-{a}-{b}',1000)
                             for a,b in comparisons})
        path=root/f'conflict_repair_predictions_{seed}.npz'; np.savez_compressed(path,**points)
        report['binding'][str(path)]=sha256(path); report['seeds'].append(item)
        (root/'conflict_repair_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True; output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
