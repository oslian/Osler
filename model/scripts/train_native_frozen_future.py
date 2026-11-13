"""Freeze only shared future predictor during matched native glucose training."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch,predict,teacher_target
from scripts.audit_native_glucose_retention import all_points
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events,append_events,expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def frozen_future_parameters(model):
    for name,p in model.named_parameters():
        if name.startswith('latent_predictor.'): p.requires_grad_(False)
    model.eval()
    return [p for p in model.parameters() if p.requires_grad]


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');out=Path('artifacts/native_frozen_future_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_selected_retention_20260911.json').read_text())
    for name,digest in ref['binding'].items():
        if sha256(Path(name))!=digest:raise ValueError('changed input: '+name)
    def load(name):
        with np.load(root/name) as f:return {k:f[k] for k in f.files}
    base,native,labels=load('body_arrays.npz'),load('native_tokens.npz'),load('multitarget_labels.npz')
    valid=load('labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('glucose')
    medians,scales=assets.scaler.medians,assets.scaler.scales
    q['truth']=labels['future'][:,col];truth=(q.truth.to_numpy()-medians[col])/scales[col]
    arrays,rows,values=native_arrays(base,native,len(variables),base['presence'].shape[1])
    report=dict(complete=False,status='candidate_only',runtime_modified=False,causal_claim_allowed=False,seeds=[],binding=ref['binding'].copy())
    for p in (Path(__file__),Path('NATIVE_FROZEN_FUTURE_PROTOCOL_20260911.md')):report['binding'][str(p)]=sha256(p)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed=entry['seed'];roles={k:v[valid]&labels['paired'][:,col] for k,v in load(f'partition_{seed}.npz').items()};fit,tune=roles['fit'],roles['tune']
        pools=[np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        events,center,event_scale=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
        old=load(f'glucose_native_predictions_{seed}.npz')
        points={'original_own':old['own'],'original_whole':old['real'],'persistence':labels['current'][:,col]}
        item=dict(seed=seed,arms={},roles={})
        for arm in ('own','whole'):
            cp=torch.load(root/f'glucose_native_{"own" if arm=="own" else "real"}_{seed}.pt',map_location='cpu',weights_only=False)
            e=events.copy()
            if arm=='own':e[:]=0
            view=append_events(restrict_input_support(arrays,np.asarray(cp['allowed_variables']),assets.model.variable_module_ids.numpy()),e)
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=deepcopy(assets.model);expand_history_projection(model,6)
            nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            initial_state=deepcopy(model.state_dict());initial=all_points(model,view,rows,medians,scales)
            params=frozen_future_parameters(model);trainable={n for n,p in model.named_parameters() if p.requires_grad}
            optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            best_mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],initial[tune,col]);best=deepcopy(initial_state);selected=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    inputs=batch(view,rows[chosen]);result=model(*inputs);y=torch.tensor(truth[chosen],dtype=torch.float32)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(result['predicted_target_latent'][:,col],teacher_target(model,inputs[1],y,col),torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,medians[col],scales[col]))
                if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best);final=all_points(model,view,rows,medians,scales)
            other=np.arange(len(variables))!=col
            other_shift=float(np.max(np.abs(final[:,other]-initial[:,other])))
            for n,t in best.items():
                if n.startswith('latent_predictor.') and not torch.equal(t,initial_state[n]):raise ValueError('shared future changed: '+n)
            points[arm]=final[:,col];
            path=root/f'glucose_frozen_future_all_{arm}_{seed}.npz';np.savez_compressed(path,final=final,initial=initial);report['binding'][str(path)]=sha256(path)
            path=root/f'glucose_frozen_future_{arm}_{seed}.pt'
            if path.exists():raise ValueError('checkpoint overwrite blocked')
            torch.save(dict(state_dict=best,variables=variables,allowed_variables=cp['allowed_variables'],event_center=center.tolist(),event_scale=event_scale.tolist(),event_width=6,value_delta_activation='identity',belief_input_mode='observed_steps',status='candidate_only'),path)
            report['binding'][str(path)]=sha256(path)
            item['arms'][arm]=dict(selected_epoch=selected,log=log,other_output_max_shift=other_shift,shared_future_exactly_preserved=True,trainable=sorted(trainable))
            print('complete',seed,arm,selected,flush=True)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'private:{seed}:{role}:{a}:{b}',2000) for a,b in [('whole','own'),('whole','original_whole'),('whole','original_own'),('own','original_own'),('whole','persistence')]})
        path=root/f'glucose_frozen_future_predictions_{seed}.npz';np.savez_compressed(path,**points);report['binding'][str(path)]=sha256(path)
        report['seeds'].append(item);(root/'glucose_frozen_future_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

