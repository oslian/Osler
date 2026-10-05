"""Matched frozen/updated context JEPA trial on availability-safe glucose."""
import argparse
import hashlib
from copy import deepcopy
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support, fit_numeric_rescale, apply_numeric_rescale
from whole_body_jepa.multisource_intervals import sha256, SEEDS as FULL_SEEDS
from whole_body_jepa.runtime import load_release
from scripts.train_release_masked_abc import CONTEXT_PREFIXES
from scripts.audit_event_aware_patient_state import patient_partition,paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_joint_jepa import _masked_latent_prediction_loss


SEEDS=(7,19,31)
ARMS=('own_frozen','whole_frozen','own_e2e','whole_e2e')
REPAIR_ARMS=tuple(f'{view}_{mode}' for view in ('own','whole')
                  for mode in ('bounded','unbounded','balanced','unbounded_balanced'))
CONFIRM_ARMS=tuple(f'{view}_{mode}' for view in ('own','whole') for mode in ('bounded','unbounded'))


def batch(arrays,rows):
    return (torch.from_numpy(arrays['input'][arrays['history'][rows]]),
            *[torch.from_numpy(arrays[k][rows]) for k in ('current','mask','ages','presence','horizon')])


def configure(model,update_context):
    # Existing EMA parameters stay frozen. No new backbone or readout.
    for name,p in model.named_parameters():
        if name.startswith(CONTEXT_PREFIXES) and not update_context:
            p.requires_grad_(False)
    model.eval()  # Identical deterministic dropout policy, gradients still enabled.
    return [p for p in model.parameters() if p.requires_grad]


def drop_context_variables(inputs, drop, membership):
    """Apply a sampled variable mask consistently across all history times."""
    history,current,mask,ages,presence,horizon=(x.clone() for x in inputs)
    n=current.shape[1]
    if drop.shape != current.shape or drop.dtype != torch.bool:
        raise ValueError('one boolean dropout flag per sample and variable required')
    current.masked_fill_(drop,0)
    mask.masked_fill_(drop,0)
    ages.masked_fill_(drop,7.)
    history[:,:,:n].masked_fill_(drop[:,None,:],0)
    history[:,:,n:2*n].masked_fill_(drop[:,None,:],0)
    history[:,:,2*n:3*n].masked_fill_(drop[:,None,:],7.)
    presence=(mask@membership.T>0).to(mask.dtype)
    history[:,:,3*n+1:-1]=(history[:,:,n:2*n]@membership.T>0).to(mask.dtype)
    return history,current,mask,ages,presence,horizon


def teacher_target(model,current,truth,index):
    with torch.no_grad():
        values=torch.zeros_like(current)
        mask=torch.zeros_like(current)
        values[:,index]=truth-current[:,index]
        mask[:,index]=1
        presence=(mask@model.variable_module_membership.T>0).to(current.dtype)
        return model._encode_body_latent(torch.zeros(len(current),model.hidden),values,mask,
                                         torch.zeros_like(mask),presence,target=True)


def source_gradient_weight(model, arrays, rows, truth, index, pools, seed):
    """Fix loss balance once from source-fit patients, never validation losses."""
    rng=np.random.default_rng(seed+170)
    params=[p for n,p in model.named_parameters() if n.startswith(CONTEXT_PREFIXES) and p.requires_grad]
    norms=[]
    for _ in range(3):
        selected=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
        inputs=batch(arrays,rows[selected])
        out=model(*inputs)
        y=torch.tensor(truth[selected],dtype=torch.float32)
        target=teacher_target(model,inputs[1],y,index)
        losses=(F.smooth_l1_loss(out['value'][:,index],y),
                _masked_latent_prediction_loss(out['predicted_target_latent'][:,index],target,
                                               torch.ones(len(y),dtype=torch.bool)))
        values=[]
        for i,loss in enumerate(losses):
            grads=torch.autograd.grad(loss,params,retain_graph=i==0,allow_unused=True)
            values.append(float(sum(g.square().sum() for g in grads if g is not None).sqrt()))
        norms.append(values)
    mean=np.mean(norms,axis=0)
    return float(np.clip(mean[0]/max(mean[1],1e-12),1.,20.)),norms


def predict(model,arrays,rows,index,median,scale,latent=False):
    model.eval()
    result=[]
    states=[]
    with torch.inference_mode():
        for start in range(0,len(rows),256):
            out=model(*batch(arrays,rows[start:start+256]))
            result.append(out['value'][:,index].numpy()*scale+median)
            if latent: states.append(out['context_latent'].numpy())
    prediction=np.concatenate(result)
    if not latent: return prediction
    x=np.concatenate(states).astype(float)
    s=np.linalg.svd(x-x.mean(0),compute_uv=False)
    prob=s*s/max(float(np.sum(s*s)),1e-12)
    rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))
    return prediction,dict(effective_rank=rank,mean_coordinate_std=float(x.std(0).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--repair-ablation',action='store_true')
    p.add_argument('--confirm-repair',action='store_true')
    p.add_argument('--epochs',type=int,choices=(8,32,64),default=8)
    p.add_argument('--source-rescale',action='store_true')
    p.add_argument('--context-dropout',action='store_true')
    p.add_argument('--history-mask',action='store_true')
    p.add_argument('--extra-patients',type=int,choices=(0,750,2000),default=0)
    p.add_argument('--actual-time',action='store_true')
    args=p.parse_args()
    if args.confirm_repair and args.repair_ablation: raise ValueError('choose one experiment')
    if args.epochs!=8 and not args.confirm_repair: raise ValueError('extended budget requires confirmation protocol')
    if args.source_rescale and (not args.confirm_repair or args.epochs!=32):raise ValueError('rescale uses matched32-epoch confirmation')
    if args.context_dropout and (not args.confirm_repair or args.epochs!=32 or args.source_rescale):
        raise ValueError('context dropout requires original-scale32-epoch confirmation')
    if args.history_mask and (not args.confirm_repair or args.epochs not in (32,64) or args.source_rescale or args.context_dropout):
        raise ValueError('history mask requires isolated original-scale32-epoch confirmation')
    if args.extra_patients and not args.history_mask:raise ValueError('source expansion requires observed_steps contract')
    if args.epochs==64 and (not args.history_mask or args.extra_patients!=2000):
        raise ValueError('64 epochs requires fixed2250-patient exposure protocol')
    if args.actual_time and args.epochs!=64:raise ValueError('actual time requires matched2250/64 protocol')
    arms=CONFIRM_ARMS if args.confirm_repair else REPAIR_ARMS if args.repair_ablation else ARMS
    seeds=FULL_SEEDS if args.confirm_repair else SEEDS
    protocol=Path('ENCODER_CONFIRM_PROTOCOL_20260910.md' if args.confirm_repair else
                  'ENCODER_REPAIR_PROTOCOL_20260910.md' if args.repair_ablation else 'AVAILABILITY_ENCODER_PROTOCOL_20260910.md')
    if args.epochs==32: protocol=Path('ENCODER_BUDGET_PROTOCOL_20260910.md')
    if args.source_rescale:protocol=Path('ENCODER_RESCALE_PROTOCOL_20260910.md')
    if args.context_dropout:protocol=Path('ENCODER_CONTEXT_DROPOUT_PROTOCOL_20260910.md')
    if args.history_mask:protocol=Path('ENCODER_HISTORY_MASK_PROTOCOL_20260910.md')
    if args.extra_patients:protocol=Path('ENCODER_LEARNING_CURVE_PROTOCOL_20260910.md')
    if args.epochs==64:protocol=Path('ENCODER_EXPANDED_BUDGET_PROTOCOL_20260910.md')
    if args.actual_time:protocol=Path('ENCODER_ACTUAL_TIME_PROTOCOL_20260910.md')
    if args.output.exists() or args.weights.exists(): raise ValueError('use fresh outputs')
    args.weights.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'))
    idx=assets.checkpoint['variables'].index('glucose')
    median=float(assets.scaler.medians[idx]); scale=float(assets.scaler.scales[idx])
    fullframes={r:pd.read_parquet(f'/private/tmp/availability_glucose_{r}.parquet') for r in ('development','eicu_hospital','mimic')}
    raw={db:pd.read_parquet(f'/private/tmp/availability_jepa_{db}_labs.parquet') for db in ('eicu','mimic')}
    extra_subjects=set()
    expansion_paths=[]
    if args.extra_patients:
        directory=Path('/private/tmp/encoder_source_expansion_20260910')
        preparation=json.loads(Path('artifacts/encoder_source_expansion_20260910.json').read_text())
        expansion_paths=[directory/'queries.parquet',directory/'labs.parquet']
        for path in expansion_paths:
            if sha256(path)!=preparation['binding'][str(path)]:raise ValueError('source expansion changed')
        extra=pd.read_parquet(expansion_paths[0])
        eligible=set(extra.loc[extra.fresh & extra.truth.notna(),'subject_id'])
        ordered=sorted(eligible,key=lambda s:hashlib.sha256(('learning-curve-v1:'+s).encode()).hexdigest())
        if len(ordered)<args.extra_patients:raise ValueError('insufficient additional source patients')
        extra_subjects=set(ordered[:args.extra_patients])
        if any(extra_subjects & set(f.subject_id) for f in fullframes.values()):raise ValueError('source patient overlap')
        extra=extra.loc[extra.subject_id.isin(extra_subjects)].reset_index(drop=True)
        fullframes['development']=pd.concat([fullframes['development'],extra],ignore_index=True)
        extra_raw=pd.read_parquet(expansion_paths[1])
        raw['eicu']=pd.concat([raw['eicu'],extra_raw.loc[extra_raw.stay_id.isin(extra.stay_id)]],ignore_index=True)
    c=duckdb.connect()
    starts=c.execute("SELECT stay_id,intime FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    c.close()
    origins={'mimic':dict(zip('mimic:'+starts.stay_id.astype(str),starts.intime)),
             'eicu':{s:pd.Timestamp('2100-01-01') for s in raw['eicu'].stay_id.unique()}}
    arrays,rows,data={},{},{}
    for role,f in fullframes.items():
        valid=(f.fresh & f.truth.notna()).to_numpy()
        data[role]=f.loc[valid].reset_index(drop=True)
        arrays[role],rows[role]={},{}
        for view in ('own','whole'):
            print('build '+role+' '+view,flush=True)
            db='mimic' if role=='mimic' else 'eicu'
            a,r=build_arrays(f,raw[db],assets,origins[db],target_only=view=='own',truthful=True)
            if args.actual_time and role=='development':
                hours=f.loc[valid,'actual_delta_t'].to_numpy(dtype=np.float32)
                if not np.isfinite(hours).all() or not ((hours>=5)&(hours<=7)).all():
                    raise ValueError('invalid elapsed-time supervision')
                a['horizon'][r[valid]]=hours
            arrays[role][view]=a
            rows[role][view]=r[valid]
    subjects=[set(f.subject_id) for f in data.values()]
    if any(subjects[i]&subjects[j] for i in range(3) for j in range(i)):
        raise ValueError('patient role overlap')
    report=dict(status='candidate_only',promotion_allowed=False,runtime_modified=False,causal_claim_allowed=False,
        untouched_lockbox=False,pretraining_patient_overlap_not_excluded=True,
        settings=dict(seeds=seeds,epochs=args.epochs,steps_per_epoch=16,batch_size=32,learning_rate=.0001,
                      weight_decay=.0001,jepa_weight='source gradient ratio capped [1,20] for balanced arms; otherwise1',
                      repair_ablation=args.repair_ablation,confirm_repair=args.confirm_repair,
                      source_rescale=args.source_rescale,
                      context_dropout=args.context_dropout,
                      history_mask=args.history_mask,
                      extra_patients=args.extra_patients,
                      actual_time=args.actual_time,
                      arms=arms,checkpoint_selection='source tune MAE including epoch zero'),
        binding={str(f):sha256(f) for f in (Path(__file__),Path('whole_body_jepa/availability_inputs.py'),
                 Path('models/whole_body_joint_jepa.pt'),Path('whole_body_joint_jepa.py'),protocol)},
        support={r:dict(rows=len(f),patients=int(f.subject_id.nunique())) for r,f in data.items()},
        frozen={},seeds=[])
    reference={}
    if args.history_mask:
        report['history_support']={}
        for role,views in arrays.items():
            report['history_support'][role]={}
            for view,a in views.items():
                empty=~a['mask'][a['history'][rows[role][view]]].any(axis=-1)
                report['history_support'][role][view]=dict(
                    empty_steps=int(empty.sum()),steps=int(empty.size),
                    queries_with_empty=int(empty.any(axis=-1).sum()),queries=len(empty))
    for role,f in data.items():
        reference[role]={v:predict(assets.model,arrays[role][v],rows[role][v],idx,median,scale) for v in ('own','whole')}
        report['frozen'][role]={v:patient_equal_mae(f,f.truth,y) for v,y in reference[role].items()}
    dev=data['development']
    y=(dev.truth.to_numpy()-median)/scale
    original_arrays=arrays
    if args.confirm_repair:
        report['data_binding']={str(path):sha256(path) for path in
            [Path(f'/private/tmp/availability_glucose_{role}.parquet') for role in fullframes]+
            [Path(f'/private/tmp/availability_jepa_{db}_labs.parquet') for db in raw]+expansion_paths}
    for seed in seeds:
        fit,tune=patient_partition(dev.subject_id,seed)
        if extra_subjects:
            original=~dev.subject_id.isin(extra_subjects).to_numpy()
            fit_base,tune_base=patient_partition(dev.loc[original,'subject_id'],seed)
            fit=np.ones(len(dev),dtype=bool);tune=np.zeros(len(dev),dtype=bool)
            fit[original]=fit_base;tune[original]=tune_base
        support=None
        centers=np.zeros(len(assets.checkpoint['variables']),np.float32)
        scales=np.ones_like(centers)
        if args.confirm_repair:
            support=source_patient_support(original_arrays['development']['whole'],rows['development']['whole'],dev.subject_id,fit)
            allowed=support>0
            arrays={role:{view:restrict_input_support(a,allowed,assets.model.variable_module_ids.numpy())
                          for view,a in views.items()} for role,views in original_arrays.items()}
            if args.source_rescale:
                centers,scales=fit_numeric_rescale(arrays['development']['whole'],rows['development']['whole'],fit,idx)
                arrays={role:{view:apply_numeric_rescale(a,centers,scales) for view,a in views.items()}
                        for role,views in arrays.items()}
        pools=[np.flatnonzero(fit & dev.subject_id.eq(s).to_numpy()) for s in sorted(set(dev.loc[fit,'subject_id']))]
        if len(pools)<50 or dev.loc[tune,'subject_id'].nunique()<30: raise ValueError('source support')
        result=dict(seed=seed,training={},evaluation={},fit_patients=int(dev.loc[fit,'subject_id'].nunique()),
                    tune_patients=int(dev.loc[tune,'subject_id'].nunique()))
        predictions={r:{} for r in ('eicu_hospital','mimic')}
        for arm in arms:
            view=arm.split('_')[0]
            torch.manual_seed(seed)
            model=deepcopy(assets.model)
            if args.history_mask:model.belief_input_mode='observed_steps'
            if 'unbounded' in arm:
                model.value_delta_activation=torch.nn.Identity()
            params=configure(model,args.repair_ablation or args.confirm_repair or arm.endswith('e2e'))
            jepa_weight,gradient_probe=(source_gradient_weight(model,arrays['development'][view],rows['development'][view],y,idx,pools,seed)
                                       if 'balanced' in arm else (1.,[]))
            optimizer=torch.optim.AdamW(params,lr=.0001,weight_decay=.0001)
            initial=assets.model.state_dict()
            best=deepcopy(model.state_dict())
            epoch_zero=predict(model,arrays['development'][view],rows['development'][view][tune],idx,median,scale)
            best_mae=patient_equal_mae(dev.loc[tune],dev.truth.to_numpy()[tune],epoch_zero)
            best_epoch=0
            log=[]
            grad_norm=0.
            rng=np.random.default_rng(seed)
            dropout_rng=np.random.default_rng(seed+73019)
            for epoch in range(args.epochs):
                losses=[]
                for step in range(16):
                    selected=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    inputs=batch(arrays['development'][view],rows['development'][view][selected])
                    if args.context_dropout and view=='whole':
                        drop=dropout_rng.random((len(selected),model.n_state))<.5
                        drop[:,idx]=False
                        inputs=drop_context_variables(inputs,torch.from_numpy(drop),model.variable_module_membership)
                    output=model(*inputs)
                    truth=torch.from_numpy(y[selected].astype(np.float32))
                    target=teacher_target(model,inputs[1],truth,idx)
                    jepa=_masked_latent_prediction_loss(output['predicted_target_latent'][:,idx],target,
                                                        torch.ones(len(selected),dtype=torch.bool))
                    value=F.smooth_l1_loss(output['value'][:,idx],truth)
                    loss=value+jepa_weight*jepa
                    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm=max(grad_norm,float(sum(p.grad.detach().square().sum() for n,p in model.named_parameters()
                        if n.startswith(CONTEXT_PREFIXES) and p.grad is not None))**.5)
                    torch.nn.utils.clip_grad_norm_(params,1.)
                    optimizer.step()
                    model.update_target_encoder()
                    losses.append([float(value.detach()),float(jepa.detach())])
                pred=predict(model,arrays['development'][view],rows['development'][view][tune],idx,median,scale)
                mae=patient_equal_mae(dev.loc[tune],dev.truth.to_numpy()[tune],pred)
                if mae<best_mae:
                    best_mae=mae;best_epoch=epoch+1;best=deepcopy(model.state_dict())
                entry=dict(epoch=epoch+1,tune_mae=mae,value_loss=float(np.mean(losses,axis=0)[0]),jepa_loss=float(np.mean(losses,axis=0)[1]))
                log.append(entry)
                print(json.dumps(dict(seed=seed,arm=arm,**entry)),flush=True)
            model.load_state_dict(best)
            changed=[n for n,v in best.items() if n.startswith(CONTEXT_PREFIXES) and not torch.equal(v,initial[n])]
            if arm.endswith('frozen') and changed: raise ValueError('frozen context changed')
            path=args.weights/f'{arm}_{seed}.pt'
            torch.save(dict(state_dict=best,seed=seed,arm=arm,status='candidate_only',
                            value_delta_activation='identity' if 'unbounded' in arm else 'tanh',jepa_weight=jepa_weight,
                            input_policy='source_seen_only' if args.confirm_repair else 'all',
                            variables=assets.checkpoint['variables'],
                            allowed_variables=allowed.tolist() if args.confirm_repair else None,
                            source_patient_support=support.tolist() if args.confirm_repair else None,
                            numeric_centers=centers.tolist(),numeric_scales=scales.tolist(),
                            training_context_dropout=.5 if args.context_dropout and view=='whole' else 0.,
                            belief_input_mode=model.belief_input_mode,
                            training_horizon_mode='actual' if args.actual_time else 'nominal6',
                            base_sha256=report['binding']['models/whole_body_joint_jepa.pt']),path)
            result['training'][arm]=dict(best_epoch=best_epoch,tune_mae=best_mae,log=log,
                context_gradient_max=grad_norm,selected_context_tensors_changed=len(changed),
                checkpoint=str(path),sha256=sha256(path),jepa_weight=jepa_weight,gradient_probe=gradient_probe)
            for role in predictions:
                predictions[role][arm],rank=predict(model,arrays[role][view],rows[role][view],idx,median,scale,latent=True)
                result['evaluation'].setdefault(role,{}).setdefault('latent',{})[arm]=rank
                if args.actual_time:
                    a=arrays[role][view]
                    original_horizon=a['horizon'].copy()
                    try:
                        a['horizon'][rows[role][view]]=data[role].actual_delta_t.to_numpy(dtype=np.float32)
                        timed=predict(model,a,rows[role][view],idx,median,scale)
                    finally:
                        a['horizon']=original_horizon
                    np.save(args.weights/f'actual_predictions_{role}_{arm}_{seed}.npy',timed)
            del model,optimizer
        comparisons=([(f'{v}_unbounded',f'{v}_bounded') for v in ('own','whole')]
                     +[(f'whole_{m}',f'own_{m}') for m in ('bounded','unbounded')]
                     +[(a,'persistence') for a in arms] if args.confirm_repair else
                     [(f'{v}_{m}',f'{v}_bounded') for v in ('own','whole') for m in ('unbounded','balanced','unbounded_balanced')]
                     +[(f'whole_{m}',f'own_{m}') for m in ('bounded','unbounded','balanced','unbounded_balanced')]
                     +[(f'{v}_{m}','persistence') for v in ('own','whole') for m in ('bounded','unbounded','balanced','unbounded_balanced')]
                     if args.repair_ablation else
                     [('whole_e2e','whole_frozen'),('own_e2e','own_frozen'),('whole_e2e','own_e2e'),('whole_e2e','persistence'),('whole_e2e','initial_whole')])
        for role,preds in predictions.items():
            f=data[role]
            preds.update(persistence=f.current.to_numpy(),initial_whole=reference[role]['whole'])
            if args.confirm_repair:
                np.savez_compressed(args.weights/f'predictions_{role}_{seed}.npz',**preds)
            result['evaluation'][role].update(mae={k:patient_equal_mae(f,f.truth,v) for k,v in preds.items()},
                comparisons={f'{a}_vs_{b}':paired(f,preds[a],preds[b],f'encoder:{seed}:{role}:{a}:{b}',1000)
                    for a,b in comparisons})
        report['seeds'].append(result)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(seed=seed,evaluation=result['evaluation'])),flush=True)
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
