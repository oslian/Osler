"""Separate auxiliary-task learning from transfer to the glucose task."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from scripts.audit_event_aware_patient_state import patient_partition, paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_auxiliary_encoder import auxiliary_loss
from scripts.train_availability_encoder import batch, configure, teacher_target
from scripts.train_release_masked_abc import CONTEXT_PREFIXES
from whole_body_jepa.availability_inputs import build_arrays, restrict_input_support
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release
from whole_body_joint_jepa import _masked_latent_prediction_loss


def gradient_agreement(model, inputs, truth, valid, indices, primary):
    output=model(*inputs)
    teacher=teacher_target(model,inputs[1],truth[:,primary],primary)
    main=F.smooth_l1_loss(output['value'][:,primary],truth[:,primary])
    main=main+_masked_latent_prediction_loss(output['predicted_target_latent'][:,primary],teacher,
                                           torch.ones(len(truth),dtype=torch.bool))
    extra=auxiliary_loss(model,output,inputs[1],truth,valid,indices)
    params=[p for name,p in model.named_parameters() if name.startswith(CONTEXT_PREFIXES) and p.requires_grad]
    grads=[]
    for i,loss in enumerate((main,extra)):
        gradients=torch.autograd.grad(loss,params,retain_graph=i==0,allow_unused=True)
        grads.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for p,g in zip(params,gradients)]))
    a,b=grads
    denominator=a.norm()*b.norm()
    return dict(cosine=float(a.dot(b)/denominator) if denominator>0 else None,
                weighted_auxiliary_to_primary_norm=float(.25*b.norm()/a.norm()) if a.norm()>0 else None,
                primary_loss=float(main.detach()),auxiliary_loss=float(extra.detach()))


def patient_quantile(values, subjects, quantile):
    values=np.asarray(values,dtype=float)
    subjects=np.asarray(subjects)
    if not len(values) or not np.isfinite(values).all() or not 0<=quantile<=1:
        raise ValueError('invalid weighted quantile input')
    _,inverse,counts=np.unique(subjects,return_inverse=True,return_counts=True)
    weights=1./counts[inverse]
    order=np.argsort(values,kind='stable')
    position=np.searchsorted(np.cumsum(weights[order]),quantile*weights.sum(),side='left')
    return float(values[order[min(position,len(values)-1)]])


def movement_metrics(frame, current, prediction):
    weights=1./frame.groupby('subject_id')['subject_id'].transform('size').to_numpy(float)
    weights=weights/weights.sum()
    actual=frame.truth.to_numpy(float)-current
    predicted=prediction-current
    x=actual-np.dot(weights,actual);y=predicted-np.dot(weights,predicted)
    denom=np.sqrt(np.dot(weights,x*x)*np.dot(weights,y*y))
    return dict(signed_error=float(np.dot(weights,prediction-frame.truth)),
                absolute_movement=float(np.dot(weights,abs(predicted))),
                actual_absolute_movement=float(np.dot(weights,abs(actual))),
                delta_correlation=float(np.dot(weights,x*y)/denom) if denom>1e-12 else None)


def own_features(arrays, rows, col, history):
    indices=arrays['history'][rows] if history else rows[:,None]
    return np.concatenate([arrays[key][indices,col] for key in ('current','mask','ages')],axis=1)


def own_ridge(arrays, rows, col, fit, tune, frame, labels, scale, history):
    x=own_features(arrays,rows,col,history)
    subjects=frame.loc[fit,'subject_id']
    weights=1./subjects.groupby(subjects).transform('size').to_numpy(float)
    weights*=len(weights)/weights.sum()
    delta=(labels['future'][fit,col]-labels['current'][fit,col])/scale
    model=Ridge(alpha=10.).fit(x[fit],delta,sample_weight=weights)
    return labels['current'][tune,col]+scale*model.predict(x[tune])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics',action='store_true')
    parser.add_argument('--own-history',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    assets=load_release(Path('models'))
    report_path=Path('artifacts/encoder_auxiliary_budget_20260910.json')
    source=json.loads(report_path.read_text())
    label_path=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    if sha256(label_path)!=source['binding'][str(label_path)]:raise ValueError('labels changed')
    labels=np.load(label_path)
    frame=pd.DataFrame({'stay_id':labels['stay_id'],'subject_id':labels['subject_id'],'t':labels['query_time']})
    original=pd.read_parquet('/private/tmp/availability_glucose_development.parquet')
    old=frame.subject_id.isin(original.subject_id).to_numpy()
    raw=pd.concat([pd.read_parquet('/private/tmp/availability_jepa_eicu_labs.parquet'),
                   pd.read_parquet('/private/tmp/encoder_source_expansion_20260910/labs.parquet')],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    arrays,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    variables=assets.checkpoint['variables']; index=variables.index('glucose')
    auxiliary=[variables.index(v) for v in source['settings']['auxiliary_targets']]
    medians,scales=np.asarray(assets.scaler.medians),np.asarray(assets.scaler.scales)
    normalized=(labels['future']-medians)/scales
    weights=Path('/private/tmp/encoder_auxiliary_budget_weights_20260910')
    report=dict(status='source_diagnostic_only',runtime_modified=False,external_data_read=False,
                selected_on_same_tune_patients=True,seeds=[],binding={str(report_path):sha256(report_path)})
    for s in source['seeds']:
        seed=s['seed']; old_fit,old_tune=patient_partition(frame.loc[old,'subject_id'],seed)
        fit=np.ones(len(frame),bool);fit[old]=old_fit
        tune=np.zeros(len(frame),bool);tune[old]=old_tune
        predictions={}; item=dict(seed=seed,targets={})
        for arm in ('glucose_only','auxiliary','shuffled_auxiliary'):
            path=weights/f'{arm}_{seed}.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            model=deepcopy(assets.model); model.load_state_dict(checkpoint['state_dict'])
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            configure(model,True)
            view=restrict_input_support(arrays,np.asarray(checkpoint['allowed_variables']),assets.model.variable_module_ids.numpy())
            chunks=[]
            with torch.inference_mode():
                for start in range(0,int(tune.sum()),256):
                    chunk=rows[tune][start:start+256]
                    chunks.append(model(*batch(view,chunk))['value'].numpy()*scales+medians)
            pred=np.concatenate(chunks); predictions[arm]=pred
            query=frame.loc[tune].copy();query['truth']=labels['future'][tune,index]
            mae=patient_equal_mae(query,query.truth,pred[:,index])
            if abs(mae-s['arms'][arm]['tune_mae'])>1e-5:raise ValueError('glucose reproduction failed')
            report['binding'][str(path)]=sha256(path)
            if arm=='auxiliary':
                pools=[np.flatnonzero(fit & frame.subject_id.eq(p).to_numpy()) for p in sorted(set(frame.loc[fit,'subject_id']))]
                rng=np.random.default_rng(seed+991)
                item['gradient_probes']=[]
                for _ in range(3):
                    selected=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    item['gradient_probes'].append(gradient_agreement(model,batch(view,rows[selected]),
                        torch.tensor(normalized[selected],dtype=torch.float32),
                        torch.from_numpy(labels['paired'][selected]),auxiliary,index))
        for col in auxiliary+([index] if args.own_history else []):
            valid=labels['paired'][tune,col]
            q=frame.loc[tune].loc[valid].copy();q['truth']=labels['future'][tune,col][valid]
            points={arm:pred[valid,col] for arm,pred in predictions.items()}
            points['persistence']=labels['current'][tune,col][valid]
            if args.own_history:
                supported=fit & labels['paired'][:,col]
                evaluated=tune & labels['paired'][:,col]
                for history,name in ((False,'own_current_ridge'),(True,'own_history_ridge')):
                    points[name]=own_ridge(arrays,rows,col,supported,evaluated,frame,labels,scales[col],history)
            if args.dynamics:
                supported=fit & labels['paired'][:,col]
                fit_delta=labels['future'][supported,col]-labels['current'][supported,col]
                fit_subjects=frame.loc[supported,'subject_id'].to_numpy()
                center=patient_quantile(fit_delta,fit_subjects,.5)
                high=patient_quantile(abs(fit_delta),fit_subjects,.75)
                points['source_median_delta']=points['persistence']+center
            target=dict(patients=int(q.subject_id.nunique()),rows=len(q),
                        mae={name:patient_equal_mae(q,q.truth,pred) for name,pred in points.items()})
            if target['patients']>=30:
                target['comparisons']={name:paired(q,points['auxiliary'],points[name],
                    f'auxlearning-{seed}-{variables[col]}-{name}',1000)
                    for name in ('glucose_only','shuffled_auxiliary','persistence')+
                    (('source_median_delta',) if args.dynamics else ())+
                    (('own_current_ridge','own_history_ridge') if args.own_history else ())}
            if args.dynamics:
                target['source_median_delta']=center
                target['source_upper_quartile_absolute_delta']=high
                target['movement']={name:movement_metrics(q,points['persistence'],pred) for name,pred in points.items()}
                large=abs(q.truth.to_numpy()-points['persistence'])>high
                target['retrospective_large_change']=dict(patients=int(q.loc[large,'subject_id'].nunique()),
                    rows=int(large.sum()),mae={name:patient_equal_mae(q.loc[large],q.loc[large,'truth'],pred[large])
                    for name,pred in points.items()} if large.any() else {})
                target['retrospective_remaining']=dict(patients=int(q.loc[~large,'subject_id'].nunique()),
                    rows=int((~large).sum()),mae={name:patient_equal_mae(q.loc[~large],q.loc[~large,'truth'],pred[~large])
                    for name,pred in points.items()} if (~large).any() else {})
            item['targets'][variables[col]]=target
        report['seeds'].append(item)
    destination=Path('artifacts/encoder_auxiliary_own_history_20260911.json' if args.own_history else
                     'artifacts/encoder_auxiliary_dynamics_20260911.json' if args.dynamics
                     else 'artifacts/encoder_auxiliary_learning_20260911.json')
    report['binding'][str(Path(__file__))]=sha256(Path(__file__))
    report['dynamics']=args.dynamics
    report['own_history']=args.own_history
    destination.write_text(json.dumps(report,indent=2)+'\n')
    for target in source['settings']['auxiliary_targets']:
        count={name:sum(s['targets'][target].get('comparisons',{}).get(name,{}).get('ci95',[0,0])[1]<0
                        for s in report['seeds']) for name in ('glucose_only','shuffled_auxiliary','persistence')}
        print(target,count)
    print('gradient',[(s['seed'],[p['cosine'] for p in s['gradient_probes']]) for s in report['seeds']])


if __name__=='__main__':
    main()
