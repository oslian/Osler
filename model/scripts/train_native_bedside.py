"""Matched bedside expansion trial with frozen query labels and patient roles."""
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
from whole_body_jepa.availability_inputs import source_patient_support, restrict_input_support, fit_numeric_rescale, apply_numeric_rescale
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,19,31,43,59,71,83])
    parser.add_argument('--rescale-bedside',action='store_true')
    parser.add_argument('--neutral-bedside',action='store_true')
    parser.add_argument('--target',choices=('glucose','map'),default='glucose')
    args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    if args.neutral_bedside and not args.rescale_bedside:
        parser.error('neutral bedside requires source-centered coordinates')
    tag='bedside_support' if args.neutral_bedside else ('bedside_rescale' if args.rescale_bedside else 'bedside')
    if args.target=='map':
        if args.neutral_bedside:parser.error('MAP includes its own support-control arm')
        tag='bedside_map'
    output=Path(f'artifacts/native_{tag}_training_20260911.json')
    if output.exists():raise ValueError('fresh outputs required')
    bindings={}
    for name in ('native_bedside_inputs','native_bedside_tokens'):
        reference=json.loads(Path(f'artifacts/{name}_20260911.json').read_text())
        for path,digest in reference['bindings'].items():
            if sha256(Path(path))!=digest:raise ValueError('input dependency changed')
            bindings[path]=digest
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    col=variables.index(args.target);median=assets.scaler.medians[col];scale=assets.scaler.scales[col]
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    if args.target=='map':
        label_path=root/'bedside_future_labels.npz'
        reference=json.loads(Path('artifacts/bedside_future_labels_20260912.json').read_text())
        for p,h in reference['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('bedside labels changed')
        bindings.update(reference['bindings'])
        bedside_labels=dict(np.load(label_path));j=bedside_labels['variables'].tolist().index('map')
        for field in ('current','future','paired'):
            labels[field][:,col]=bedside_labels[field][:,j]
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    eligible=labels['paired'][:,col];q['truth']=labels['future'][:,col]
    truth=(q.truth.to_numpy()-median)/scale
    for p in (root/'multitarget_labels_joint_conflict_checked.npz',Path(__file__)):
        bindings[str(p)]=sha256(p)
    partitions={e['seed']:e['partition_sha256'] for e in json.loads(Path('artifacts/rate_anchored_potassium_20260911.json').read_text())['seeds']}
    report=dict(complete=False,status='candidate_only',runtime_modified=False,seeds=[],bindings=bindings,
        target=args.target,
        numeric_mode='neutral' if args.neutral_bedside else ('source_rescale' if args.rescale_bedside else 'inherited'),
        comparison='same clean labels and roles; lab versus lab plus bedside; 32 epochs; source-only selection')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for seed in args.seeds:
        p=root/f'partition_{seed}.npz'
        if sha256(p)!=partitions[seed]:raise ValueError('partition changed')
        bindings[str(p)]=sha256(p)
        roles={a:b[valid]&eligible for a,b in dict(np.load(p)).items()}
        fit,tune=roles['fit'],roles['tune']
        pools=[np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        points={'persistence':labels['current'][:,col]};item=dict(seed=seed,arms={},roles={})
        arms=(
            ('lab','body_arrays_conflict_checked.npz','native_tokens_conflict_replay.npz'),
            ('bedside','body_arrays_bedside.npz','native_tokens_bedside.npz'))
        if args.target=='map':
            arms=tuple((a,'body_arrays_bedside.npz','native_tokens_bedside.npz') for a in ('own','bedside','support'))
        for arm,base_name,native_name in arms:
            checkpoint=root/f'{tag}_{arm}_{seed}.pt'
            if checkpoint.exists():raise ValueError('checkpoint overwrite blocked')
            base=dict(np.load(root/base_name));native=dict(np.load(root/native_name))
            for p in (root/base_name,root/native_name):bindings[str(p)]=sha256(p)
            np.testing.assert_allclose(base['current'][base['query_rows'][eligible],col]*scale+median,
                labels['current'][eligible,col],atol=1e-4,rtol=1e-6)
            arrays,rows,values=native_arrays(base,native,len(variables),base['presence'].shape[1])
            numeric_center=np.zeros(len(variables),np.float32)
            numeric_scale=np.ones(len(variables),np.float32)
            if args.rescale_bedside and arm!='lab':
                centers,scales=fit_numeric_rescale(arrays,rows,fit,col)
                for variable in ('heart_rate','map','o2sat','respiratory_rate','temperature'):
                    i=variables.index(variable);numeric_center[i]=centers[i];numeric_scale[i]=scales[i]
                arrays=apply_numeric_rescale(arrays,numeric_center,numeric_scale)
            allowed=source_patient_support(base,base['query_rows'],q.subject_id,fit)>0
            events,center,event_scale=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
            if arm=='own':
                allowed[np.arange(len(variables))!=col]=False;events[:]=0
            view=restrict_input_support(arrays,allowed,assets.model.variable_module_ids.numpy())
            if arm=='support':
                columns=np.flatnonzero(np.arange(len(variables))!=col)
                view['current'][:,columns]=0;view['input'][:,columns]=0
            if args.neutral_bedside and arm=='bedside':
                columns=[variables.index(v) for v in ('heart_rate','map','o2sat','respiratory_rate','temperature')]
                view['current'][:,columns]=0;view['input'][:,columns]=0
            view=append_events(view,events)
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=deepcopy(assets.model);expand_history_projection(model,events.shape[1])
            nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            params=configure(model,True);optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            best_mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
            best=deepcopy(model.state_dict());selected=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    chosen=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    x=batch(view,rows[chosen]);result=model(*x)
                    y=torch.tensor(truth[chosen],dtype=torch.float32)
                    teacher=teacher_target(model,x[1],y,col)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(
                        result['predicted_target_latent'][:,col],teacher,torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,median,scale))
                if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best);points[arm]=predict(model,view,rows,col,median,scale)
            torch.save(dict(state_dict=best,variables=variables,allowed_variables=allowed.tolist(),
                event_center=center.tolist(),event_scale=event_scale.tolist(),event_width=6,
                numeric_center=numeric_center.tolist(),numeric_scale=numeric_scale.tolist(),
                neutral_bedside=bool(args.neutral_bedside and arm=='bedside'),
                value_delta_activation='identity',belief_input_mode='observed_steps',native_history=True,
                history_contract=arm,status='candidate_only'),checkpoint)
            bindings[str(checkpoint)]=sha256(checkpoint)
            item['arms'][arm]=dict(selected_epoch=selected,tune_mae=best_mae,log=log)
            print('completed',seed,arm,selected,flush=True)
            del view,arrays,native,base,model,optimizer,nf,best,events
        contrasts=[('bedside','lab'),('bedside','persistence'),('lab','persistence')]
        if args.target=='map':contrasts=[('bedside','own'),('bedside','support'),('bedside','persistence'),('own','persistence')]
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            item['roles'][role]=dict(patients=int(f.subject_id.nunique()),
                mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'bedside-{seed}-{role}-{a}-{b}',1000)
                    for a,b in contrasts})
        p=root/f'{tag}_predictions_{seed}.npz';np.savez_compressed(p,**points);bindings[str(p)]=sha256(p)
        report['seeds'].append(item)
        (root/f'{tag}_training_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
