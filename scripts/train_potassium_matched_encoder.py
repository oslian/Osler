"""Same-capacity source-only own versus whole potassium supervision."""
import json
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_event_aware_patient_state import patient_partition, paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from scripts.audit_encoder_groups import neutralize_component
from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support
from whole_body_jepa.multisource_intervals import sha256, SEEDS, stable_hash
from whole_body_jepa.runtime import load_release
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_joint_jepa import _masked_latent_prediction_loss


def event_pools(frame, fit, event_ids):
    """Training-only patient -> distinct source outcome -> legal query rows."""
    f=frame.loc[fit,['subject_id','stay_id']].copy()
    f['event']=np.asarray(event_ids)[fit]
    f['row']=np.flatnonzero(fit)
    if f.empty or f.event.isna().any() or f.event.eq('').any():
        raise ValueError('paired training outcomes require event IDs')
    if f.groupby(['stay_id','event']).subject_id.nunique().max()!=1:
        raise ValueError('event spans patients')
    return [[g.row.to_numpy() for _,g in patient.groupby(['stay_id','event'],sort=True)]
            for _,patient in f.groupby('subject_id',sort=True)]


def sample_events(pools, rng, size):
    return np.array([rng.choice(pools[i][rng.integers(len(pools[i]))])
                     for i in rng.integers(len(pools),size=size)])


def three_way_partition(subjects, eligible, seed):
    subjects=pd.Series(np.asarray(subjects)).astype(str)
    eligible=np.asarray(eligible,dtype=bool)
    names=sorted(set(subjects[eligible]),key=lambda s:(stable_hash('potassium-three-way',seed,s),s))
    n=len(names)
    if n<10:raise ValueError('insufficient patients for three-way split')
    cuts=(0,int(.6*n),int(.8*n),n)
    masks=tuple(subjects.isin(names[a:b]).to_numpy() & eligible for a,b in zip(cuts[:-1],cuts[1:]))
    if not np.array_equal(np.sum(masks,axis=0),eligible.astype(int)):
        raise ValueError('partition does not cover eligible rows exactly once')
    return masks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--mae',action='store_true')
    mode.add_argument('--event-balanced',action='store_true')
    mode.add_argument('--three-way',action='store_true')
    mode.add_argument('--support-only',action='store_true')
    args=parser.parse_args()
    if args.support_only:args.three_way=True
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    name='potassium_mae_encoder_20260911' if args.mae else 'potassium_matched_encoder_20260911'
    if args.event_balanced:name='potassium_event_balanced_20260911'
    if args.three_way:name='potassium_three_way_20260911'
    if args.support_only:name='potassium_support_only_20260911'
    destination=Path(f'artifacts/{name}.json')
    weights=Path('/private/tmp')/name
    if destination.exists() or weights.exists():
        raise ValueError('fresh outputs required')
    support=json.loads(Path('artifacts/auxiliary_supervision_20260910.json').read_text())
    label_path=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    raw_paths=[Path('/private/tmp/availability_jepa_eicu_labs.parquet'),
               Path('/private/tmp/encoder_source_expansion_20260910/labs.parquet')]
    for p in [label_path]+raw_paths:
        if sha256(p)!=support['binding'][str(p)]:raise ValueError('source changed')
    labels=np.load(label_path)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    if list(labels['variables'])!=variables:raise ValueError('variable mismatch')
    col=variables.index('potassium')
    frame=pd.DataFrame(dict(stay_id=labels['stay_id'],subject_id=labels['subject_id'],
        t=labels['query_time'],truth=labels['future'][:,col]))
    old=frame.subject_id.isin(pd.read_parquet('/private/tmp/availability_glucose_development.parquet').subject_id).to_numpy()
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    event_ids=None
    if args.event_balanced:
        rebuilt=future_pairs(frame,raw,['potassium'])
        for key in ('current','future','measurement_age','actual_delta_t','paired'):
            np.testing.assert_equal(rebuilt[key][:,0],labels[key][:,col])
        event_ids=rebuilt['future_source_id'][:,0]
    arrays,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    truth=(frame.truth.to_numpy()-median)/scale
    protocol=Path('POTASSIUM_MAE_PROTOCOL_20260911.md' if args.mae else 'POTASSIUM_MATCHED_ENCODER_PROTOCOL_20260911.md')
    if args.event_balanced:protocol=Path('POTASSIUM_EVENT_BALANCED_PROTOCOL_20260911.md')
    if args.three_way:protocol=Path('POTASSIUM_THREE_WAY_PROTOCOL_20260911.md')
    if args.support_only:protocol=Path('POTASSIUM_SUPPORT_ONLY_PROTOCOL_20260911.md')
    report=dict(status='source_diagnostic_only',runtime_modified=False,external_data_read=False,
        promotion_allowed=False,pretraining_patient_overlap_not_excluded=True,seeds=[],
        value_loss='l1' if args.mae else 'smooth_l1',
        sampling='patient_event_query' if args.event_balanced else 'patient_query',
        partition='all_source_60_20_20' if args.three_way else 'original_tune_expanded_fit',
        cross_numeric='removed_at_train_and_evaluation' if args.support_only else 'observed',
        binding={str(p):sha256(p) for p in [Path(__file__),protocol,label_path]+raw_paths})
    weights.mkdir()
    for seed in SEEDS:
        a,b=patient_partition(frame.loc[old,'subject_id'],seed)
        fit=np.ones(len(frame),bool);fit[old]=a;fit&=labels['paired'][:,col]
        tune=np.zeros(len(frame),bool);tune[old]=b;tune&=labels['paired'][:,col]
        evaluation=None
        if args.three_way:
            fit,tune,evaluation=three_way_partition(frame.subject_id,labels['paired'][:,col],seed)
        if set(frame.loc[fit,'subject_id']) & set(frame.loc[tune,'subject_id']):raise ValueError('patient overlap')
        if frame.loc[tune,'subject_id'].nunique()<30:raise ValueError('insufficient tune support')
        allowed=source_patient_support(arrays,rows,frame.subject_id,fit)>0
        pools=[np.flatnonzero(fit & frame.subject_id.eq(s).to_numpy()) for s in sorted(set(frame.loc[fit,'subject_id']))]
        balanced=event_pools(frame,fit,event_ids) if args.event_balanced else None
        item=dict(seed=int(seed),fit_patients=len(pools),tune_patients=int(frame.loc[tune,'subject_id'].nunique()),arms={},comparisons={})
        predictions={'persistence':labels['current'][tune,col]}
        evaluation_predictions={}
        if args.three_way:
            item['evaluation_patients']=int(frame.loc[evaluation,'subject_id'].nunique())
            evaluation_predictions['persistence']=labels['current'][evaluation,col]
            np.savez(weights/f'partition_{seed}.npz',fit=fit,tune=tune,evaluation=evaluation)
        for arm in ('own','whole'):
            channels=allowed.copy()
            if arm=='own':channels[np.arange(len(channels))!=col]=False
            view=restrict_input_support(arrays,channels,assets.model.variable_module_ids.numpy())
            if args.support_only and arm=='whole':
                cross=channels.copy();cross[col]=False
                view=neutralize_component(view,cross,'value')
            torch.manual_seed(seed)
            model=deepcopy(assets.model);model.value_delta_activation=torch.nn.Identity()
            model.belief_input_mode='observed_steps';params=configure(model,True)
            optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            initial=predict(model,view,rows[tune],col,median,scale)
            best_mae=patient_equal_mae(frame.loc[tune],frame.loc[tune,'truth'],initial)
            best=deepcopy(model.state_dict());best_epoch=0;log=[];rng=np.random.default_rng(seed)
            for epoch in range(64):
                for _ in range(16):
                    selected=(sample_events(balanced,rng,32) if args.event_balanced else
                        np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)]))
                    inputs=batch(view,rows[selected]);output=model(*inputs)
                    y=torch.tensor(truth[selected],dtype=torch.float32)
                    teacher=teacher_target(model,inputs[1],y,col)
                    value_loss=F.l1_loss if args.mae else F.smooth_l1_loss
                    loss=value_loss(output['value'][:,col],y)+_masked_latent_prediction_loss(
                        output['predicted_target_latent'][:,col],teacher,torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                pred=predict(model,view,rows[tune],col,median,scale)
                mae=patient_equal_mae(frame.loc[tune],frame.loc[tune,'truth'],pred)
                if mae<best_mae:best_mae,best_epoch,best=mae,epoch+1,deepcopy(model.state_dict())
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best)
            predictions[arm]=predict(model,view,rows[tune],col,median,scale)
            torch.save(dict(state_dict=best,allowed_variables=channels.tolist(),variables=variables,
                value_delta_activation='identity',belief_input_mode='observed_steps',target='potassium',
                selected_epoch=best_epoch,value_loss=report['value_loss'],cross_numeric=report['cross_numeric'],status='candidate_only'),weights/f'{arm}_{seed}.pt')
            np.save(weights/f'{arm}_{seed}_tune.npy',predictions[arm])
            if args.three_way:
                evaluation_predictions[arm]=predict(model,view,rows[evaluation],col,median,scale)
                np.save(weights/f'{arm}_{seed}_evaluation.npy',evaluation_predictions[arm])
            item['arms'][arm]=dict(tune_mae=best_mae,selected_epoch=best_epoch,log=log)
            print(json.dumps(dict(seed=int(seed),arm=arm,mae=best_mae,epoch=best_epoch)),flush=True)
        for candidate,reference in (('whole','own'),('whole','persistence'),('own','persistence')):
            item['comparisons'][f'{candidate}_vs_{reference}']=paired(frame.loc[tune],predictions[candidate],
                predictions[reference],f'potassium-matched-{seed}-{candidate}-{reference}',1000)
        if args.three_way:
            item['evaluation_mae']={a:patient_equal_mae(frame.loc[evaluation],frame.loc[evaluation,'truth'],p)
                                    for a,p in evaluation_predictions.items()}
            item['evaluation_comparisons']={f'{a}_vs_{b}':paired(frame.loc[evaluation],evaluation_predictions[a],
                evaluation_predictions[b],f'potassium-three-way-{seed}-{a}-{b}',1000)
                for a,b in (('whole','own'),('whole','persistence'),('own','persistence'))}
        if args.support_only:
            reference=Path('/private/tmp/potassium_three_way_20260911')
            original=json.loads(Path('artifacts/potassium_three_way_20260911.json').read_text())['seeds']
            old_entry=next(s for s in original if s['seed']==seed)
            if item['arms']['own']!=old_entry['arms']['own']:raise ValueError('own training control changed')
            for role,mask,outputs in (('tune',tune,predictions),('evaluation',evaluation,evaluation_predictions)):
                np.testing.assert_array_equal(outputs['own'],np.load(reference/f'own_{seed}_{role}.npy'))
                full=np.load(reference/f'whole_{seed}_{role}.npy')
                item.setdefault('support_vs_full_numeric',{})[role]=paired(frame.loc[mask],outputs['whole'],full,
                    f'potassium-support-{seed}-{role}',1000)
        if args.mae or args.event_balanced:
            for arm in ('own','whole'):
                baseline=np.load(f'/private/tmp/potassium_matched_encoder_20260911/{arm}_{seed}_tune.npy')
                if baseline.shape!=predictions[arm].shape:raise ValueError('baseline support mismatch')
                item['comparisons'][f'{arm}_vs_smooth_l1']=paired(frame.loc[tune],predictions[arm],baseline,
                    f'potassium-mae-{seed}-{arm}',1000)
        report['seeds'].append(item);destination.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;destination.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
