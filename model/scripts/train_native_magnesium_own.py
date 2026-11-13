"""Glucose matched native-event diagnostic on existing patient roles."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import batch,configure,predict,teacher_target
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import source_patient_support,restrict_input_support
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events,append_events,expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');out=Path('artifacts/native_magnesium_own_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    bindings={}
    for report in ('rate_multitarget_labels','native_rate_jepa_tokens','rate_anchor_body_inputs'):
        ref=json.loads(Path(f'artifacts/{report}_20260911.json').read_text())
        for name,digest in ref['binding'].items():
            if Path(name).suffix in ('.npz','.parquet'):
                if sha256(Path(name))!=digest:raise ValueError('input changed')
                bindings[name]=digest
    with np.load(root/'body_arrays.npz') as f:base={k:f[k] for k in f.files}
    with np.load(root/'native_tokens.npz') as f:native={k:f[k] for k in f.files}
    with np.load(root/'multitarget_labels.npz') as f:labels={k:f[k] for k in f.files}
    old=np.load(root/'labels.npz');valid=old['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('magnesium')
    np.testing.assert_array_equal(labels['variables'],variables)
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')];q['truth']=labels['future'][:,col];truth=(q.truth.to_numpy()-median)/scale
    np.testing.assert_allclose(base['current'][base['query_rows'][eligible],col]*scale+median,labels['current'][eligible,col],atol=1e-4,rtol=1e-6)
    arrays,rows,values=native_arrays(base,native,len(variables),base['presence'].shape[1])
    prior=json.loads(Path('artifacts/rate_anchored_potassium_20260911.json').read_text())
    for p in (Path(__file__),Path('NATIVE_MAGNESIUM_OWN_PROTOCOL_20260911.md')):bindings[str(p)]=sha256(p)
    report=dict(complete=False,status='candidate_only',runtime_modified=False,pretraining_overlap_not_excluded=True,seeds=[],binding=bindings)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for entry in prior['seeds']:
        seed=entry['seed'];path=root/f'partition_{seed}.npz'
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        report['binding'][str(path)]=sha256(path)
        roles={a:b[valid]&eligible for a,b in dict(np.load(path)).items()};fit,tune=roles['fit'],roles['tune']
        pools=[np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        allowed=source_patient_support(base,base['query_rows'],q.subject_id,fit)>0
        events,center,event_scale=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
        points={'persistence':labels['current'][:,col], 'real':np.load(root/f'glucose_truth_auxiliary_predictions_real_{seed}.npz')['final'][:,col]};item=dict(seed=seed,arms={},roles={})
        for arm in ('own',):
            channels=allowed.copy()
            if arm=='own':channels[np.arange(len(channels))!=col]=False
            view=restrict_input_support(arrays,channels,assets.model.variable_module_ids.numpy());e=events.copy()
            if arm=='support':e[:,:3]=0
            if arm=='own':e[:]=0
            view=append_events(view,e);torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=deepcopy(assets.model);expand_history_projection(model,e.shape[1])
            nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            params=configure(model,True);optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            best_mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale));best=deepcopy(model.state_dict());epoch_selected=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    inputs=batch(view,rows[chosen]);result=model(*inputs);y=torch.tensor(truth[chosen],dtype=torch.float32)
                    teacher=teacher_target(model,inputs[1],y,col)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(result['predicted_target_latent'][:,col],teacher,torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
                if mae<best_mae:best_mae,best,epoch_selected=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best);points[arm]=predict(model,view,rows,col,median,scale)
            path=root/f'magnesium_native_{arm}_{seed}.pt'
            if path.exists():raise ValueError('checkpoint overwrite blocked')
            torch.save(dict(state_dict=best,variables=variables,allowed_variables=channels.tolist(),event_center=center.tolist(),event_scale=event_scale.tolist(),event_mode=arm,event_width=6,clock_clipped_hours=168,native_history=True,value_delta_activation='identity',belief_input_mode='observed_steps',status='candidate_only'),path)
            report['binding'][str(path)]=sha256(path);item['arms'][arm]=dict(selected_epoch=epoch_selected,tune_mae=best_mae,log=log)
            print('complete',seed,arm,epoch_selected,flush=True)
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(patients=int(f.subject_id.nunique()),mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'glucose-native-{seed}-{role}-{a}-{b}',1000) for a,b in [('real','own'),('real','persistence'),('own','persistence')]})
        predpath=root/f'magnesium_native_predictions_{seed}.npz';np.savez_compressed(predpath,**points);report['binding'][str(predpath)]=sha256(predpath)
        report['seeds'].append(item);(root/'magnesium_native_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

