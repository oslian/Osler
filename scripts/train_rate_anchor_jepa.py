"""Shared JEPA own/whole comparison at fixed rate-record query times."""
import json
import argparse
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
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss
from scripts.audit_encoder_groups import neutralize_component
from whole_body_jepa.history_event_input import prepare_events,append_events,expand_history_projection
from whole_body_jepa.native_event_encoder import NativeEventFilter,native_arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-only',action='store_true')
    parser.add_argument('--event-history',action='store_true')
    parser.add_argument('--native-events',action='store_true')
    parser.add_argument('--repair-clock',action='store_true')
    parser.add_argument('--synthetic-signal',action='store_true')
    parser.add_argument('--projection-fast',action='store_true')
    args=parser.parse_args()
    if args.synthetic_signal and (not args.native_events or args.repair_clock):raise ValueError('synthetic control uses original native clock')
    if args.projection_fast and (not args.native_events or args.repair_clock):raise ValueError('fast projection uses native original clock')
    real_fast=args.projection_fast and not args.synthetic_signal
    if args.repair_clock and not args.native_events:raise ValueError('clock repair requires native events')
    if sum((args.support_only,args.event_history,args.native_events))>1:raise ValueError('choose one contrast')
    output=Path('artifacts/rate_anchor_jepa_support_20260911.json' if args.support_only else 'artifacts/rate_anchor_jepa_20260911.json')
    if args.event_history:output=Path('artifacts/rate_anchor_jepa_events_20260911.json')
    if args.native_events:output=Path('artifacts/rate_anchor_jepa_native_20260911.json')
    if args.repair_clock:output=Path('artifacts/rate_anchor_jepa_clock_20260911.json')
    if args.synthetic_signal:output=Path('artifacts/rate_anchor_jepa_synthetic_20260911.json')
    if args.projection_fast:output=Path('artifacts/rate_anchor_jepa_synthetic_fast_20260911.json')
    if real_fast:output=Path('artifacts/rate_anchor_jepa_real_fast_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    body_path=Path('artifacts/rate_anchor_body_inputs_20260911.json')
    body=json.loads(body_path.read_text());prior=json.loads(Path('artifacts/rate_anchored_potassium_20260911.json').read_text())
    if sha256(root/'body_arrays.npz')!=body['binding'][str(root/'body_arrays.npz')]:raise ValueError('array cache changed')
    for name in ('queries.parquet','labels.npz'):
        if sha256(root/name)!=prior['binding'][str(root/name)]:raise ValueError('label cache changed')
    loaded=dict(np.load(root/'body_arrays.npz'));rows=loaded.pop('query_rows');arrays=loaded
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0];after=labels['bracket_after'][valid]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0];assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('potassium')
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    truth=(q.truth.to_numpy()-median)/scale
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    reference=json.loads(Path('artifacts/rate_jepa_hospitals_20260911.json').read_text()) if args.support_only or args.event_history or args.native_events else None
    event_values=None
    snapshot_arrays,snapshot_rows=arrays,rows
    if args.native_events:
        event_report=json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
        for name,digest in event_report['binding'].items():
            if sha256(Path(name))!=digest:raise ValueError('native inputs changed')
        with np.load(root/'native_tokens.npz') as cache:native={k:cache[k] for k in cache.files}
        original_native=native
        if args.repair_clock:
            clock_report=json.loads(Path('artifacts/native_clock_tokens_20260911.json').read_text())
            for name,digest in clock_report['binding'].items():
                if sha256(Path(name))!=digest:raise ValueError('clock inputs changed')
            with np.load(root/'native_clock_tokens.npz') as cache:native={k:cache[k] for k in cache.files}
        arrays,rows,event_values=native_arrays(dict(arrays,query_rows=rows),native,len(variables),arrays['presence'].shape[1])
        if args.synthetic_signal:
            # Independent artificial event values, never derived from outcomes.
            known=np.isfinite(event_values[:,2]);event_values=event_values.copy()
            event_values[known,2]=np.random.default_rng(20260911).choice([-1.,1.],int(known.sum()))
            shifts=[]
            for row in rows:
                history=arrays['history'][row];seen=history[known[history]]
                shifts.append(.5*event_values[seen[-1],2] if len(seen) else 0.)
            q['truth']=q.truth.to_numpy()+np.asarray(shifts);truth=(q.truth.to_numpy()-median)/scale
        if args.repair_clock or real_fast:
            old_arrays,old_rows,_=native_arrays(dict(snapshot_arrays,query_rows=snapshot_rows),original_native,len(variables),snapshot_arrays['presence'].shape[1])
            old_report=json.loads(Path('artifacts/rate_anchor_jepa_native_20260911.json').read_text())
    if args.event_history:
        event_report=json.loads(Path('artifacts/rate_jepa_event_history_20260911.json').read_text())
        for name,digest in event_report['binding'].items():
            if sha256(Path(name))!=digest:raise ValueError('event inputs changed')
        event_values=np.load(root/'event_history.npz')['values']
    report=dict(complete=False,status='candidate_only',runtime_modified=False,promotion_allowed=False,
                support_only=args.support_only,
                synthetic_signal=args.synthetic_signal,
                epochs=32,steps_per_epoch=16,pretraining_patient_overlap_not_excluded=True,
                treatment_input=args.event_history or args.native_events,native_events=args.native_events,query_type='retrospective_rate_record',seeds=[],
                binding={str(p):sha256(p) for p in [Path(__file__),body_path,Path('RATE_ANCHOR_JEPA_SUPPORT_PROTOCOL_20260911.md' if args.support_only else 'RATE_ANCHOR_JEPA_PROTOCOL_20260911.md')]})
    if args.event_history:
        for p in (Path('RATE_JEPA_EVENTS_PROTOCOL_20260911.md'),Path('artifacts/rate_jepa_event_history_20260911.json'),Path('whole_body_jepa/history_event_input.py')):
            report['binding'][str(p)]=sha256(p)
    if args.native_events:
        report['binding'].update(event_report['binding'])
        for p in map(Path,('NATIVE_RATE_JEPA_TRAIN_PROTOCOL_20260911.md','whole_body_jepa/native_event_encoder.py','whole_body_jepa/history_event_input.py')):
            report['binding'][str(p)]=sha256(p)
    if args.repair_clock:
        report['binding'].update(clock_report['binding']);report['repair_clock']=True
        report['binding']['NATIVE_CLOCK_TRAIN_PROTOCOL_20260911.md']=sha256(Path('NATIVE_CLOCK_TRAIN_PROTOCOL_20260911.md'))
    if args.synthetic_signal:
        report['binding']['NATIVE_SYNTHETIC_PROTOCOL_20260911.md']=sha256(Path('NATIVE_SYNTHETIC_PROTOCOL_20260911.md'))
        report['synthetic_amplitude']=.5
    if args.projection_fast:
        report['binding']['NATIVE_PROJECTION_PROTOCOL_20260911.md']=sha256(Path('NATIVE_PROJECTION_PROTOCOL_20260911.md'))
        report['observation_projection_lr']=.001
    if real_fast:report['binding']['NATIVE_REAL_PROJECTION_PROTOCOL_20260911.md']=sha256(Path('NATIVE_REAL_PROJECTION_PROTOCOL_20260911.md'))
    for entry in prior['seeds']:
        seed=entry['seed'];path=root/f'partition_{seed}.npz'
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        roles={k:v[valid] for k,v in dict(np.load(path)).items()};fit,tune=roles['fit'],roles['tune']
        allowed=source_patient_support(snapshot_arrays,snapshot_rows,q.subject_id,fit)>0
        pools=[np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        points={'persistence':labels['current'][valid,0]};item=dict(seed=seed,arms={},roles={})
        if args.support_only or args.event_history or args.native_events:
            prediction_path=root/f'jepa_predictions_{seed}.npz'
            if sha256(prediction_path)!=reference['binding'][str(prediction_path)]:raise ValueError('reference predictions changed')
            points.update(dict(np.load(prediction_path)))
            report['binding'][str(prediction_path)]=sha256(prediction_path)
        arms=('event_real','event_support') if args.event_history else (('support',) if args.support_only else ('own','whole'))
        if args.native_events:arms=('native_real','native_support')
        if args.synthetic_signal:arms=('synthetic_real','synthetic_support')
        if args.projection_fast:arms=('fast_real','fast_support')
        if real_fast:arms=('real_fast','real_fast_support')
        if args.repair_clock or real_fast:
            if args.repair_clock:arms=('clock_real','clock_support')
            for old_arm in ('native_real','native_support'):
                path=root/f'jepa_{old_arm}_{seed}.pt'
                if sha256(path)!=old_report['binding'][str(path)]:raise ValueError('native checkpoint changed')
                saved=torch.load(path,weights_only=True);model=deepcopy(assets.model)
                expand_history_projection(model,saved['event_width']);model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
                model.load_state_dict(saved['state_dict']);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
                view=restrict_input_support(old_arrays,np.array(saved['allowed_variables']),assets.model.variable_module_ids.numpy())
                numbers=np.where(np.isfinite(event_values),(event_values-np.array(saved['event_center']))/np.array(saved['event_scale']),0.)
                if old_arm=='native_support':numbers[:]=0
                view=append_events(view,np.column_stack([numbers,np.isfinite(event_values)]).astype(np.float32))
                points[old_arm]=predict(model,view,old_rows,col,median,scale)
                ref_item=next(v for v in old_report['seeds'] if v['seed']==seed)
                for role in ('evaluation','hospital'):
                    m=roles[role];score=patient_equal_mae(q.loc[m],q.loc[m,'truth'],points[old_arm][m])
                    np.testing.assert_allclose(score,ref_item['roles'][role]['all']['mae'][old_arm],atol=1e-7,rtol=0)
                report['binding'][str(path)]=sha256(path)
        for arm in arms:
            channels=allowed.copy()
            if arm=='own':channels[np.arange(len(channels))!=col]=False
            view=restrict_input_support(arrays,channels,assets.model.variable_module_ids.numpy())
            if arm=='support':
                cross=channels.copy();cross[col]=False
                view=neutralize_component(view,cross,'value')
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=deepcopy(assets.model);model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
            event_metadata={}
            if args.event_history or args.native_events:
                events,center,event_scale=prepare_events(event_values,np.unique(arrays['history'][rows[fit]]))
                if arm in ('event_support','native_support','clock_support','synthetic_support','fast_support','real_fast_support'):events[:,:event_values.shape[1]]=0
                baseline=predict(model,view,rows[tune][:32],col,median,scale)
                view=append_events(view,events);expand_history_projection(model,events.shape[1])
                initial=predict(model,view,rows[tune][:32],col,median,scale)
                np.testing.assert_allclose(initial,baseline,atol=1e-5,rtol=1e-6)
                event_metadata=dict(event_center=center.tolist(),event_scale=event_scale.tolist(),event_width=events.shape[1],event_mode=arm,lab_supported_steps_only=True)
                if args.native_events:
                    native_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
                    native_filter.load_state_dict(model.belief_filter.state_dict());model.belief_filter=native_filter
                    event_metadata.update(lab_supported_steps_only=False,native_history=True,lookback_hours=48,clock_clipped_hours=168)
                    if args.repair_clock:event_metadata['clock_clipped_hours']=None
            params=configure(model,True)
            if args.projection_fast:
                projection=list(model.belief_filter.observation_projection[0].parameters());ids={id(p) for p in projection}
                groups=[dict(params=[p for p in params if id(p) not in ids],lr=1e-4),dict(params=projection,lr=.001)]
                optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
            else:optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
            pred=predict(model,view,rows[tune],col,median,scale)
            best_mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],pred);best=deepcopy(model.state_dict());selected_epoch=0;log=[]
            for epoch in range(32):
                for _ in range(16):
                    selected=np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                    inputs=batch(view,rows[selected]);result=model(*inputs);y=torch.tensor(truth[selected],dtype=torch.float32)
                    teacher=teacher_target(model,inputs[1],y,col)
                    loss=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(result['predicted_target_latent'][:,col],teacher,torch.ones(len(y),dtype=torch.bool))
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
                pred=predict(model,view,rows[tune],col,median,scale);mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],pred)
                if mae<best_mae:best_mae,best,selected_epoch=mae,deepcopy(model.state_dict()),epoch+1
                log.append(dict(epoch=epoch+1,tune_mae=mae))
            model.load_state_dict(best);points[arm],latent=predict(model,view,rows,col,median,scale,latent=True)
            path=root/f'jepa_{arm}_{seed}.pt';torch.save(dict(state_dict=best,variables=variables,allowed_variables=channels.tolist(),value_delta_activation='identity',belief_input_mode='observed_steps',status='candidate_only',**event_metadata),path)
            report['binding'][str(path)]=sha256(path)
            item['arms'][arm]=dict(selected_epoch=selected_epoch,tune_mae=best_mae,latent=latent,log=log)
            print(f'JEPA seed {seed} {arm} complete epoch {selected_epoch}',flush=True)
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for group,g in [('all',np.ones(len(q),bool)),('bracket_after',after)]:
                mask=roles[role]&g;f=q.loc[mask];p={k:v[mask] for k,v in points.items()}
                comparisons=[('whole','support'),('support','own'),('support','persistence')] if args.support_only else [('whole','own'),('whole','persistence'),('own','persistence')]
                if args.event_history:comparisons=[('event_real','event_support'),('event_real','whole'),('event_real','own'),('event_real','persistence')]
                if args.native_events:comparisons=[('native_real','native_support'),('native_real','whole'),('native_real','own'),('native_real','persistence'),('native_support','whole')]
                if args.repair_clock:comparisons=[('clock_real','native_real'),('clock_support','native_support'),('clock_real','clock_support'),('clock_real','whole'),('clock_real','own'),('clock_real','persistence')]
                if args.synthetic_signal:comparisons=[('synthetic_real','synthetic_support'),('synthetic_real','whole'),('synthetic_real','persistence')]
                if args.projection_fast:comparisons=[('fast_real','fast_support'),('fast_real','whole'),('fast_real','persistence')]
                if real_fast:comparisons=[('real_fast','real_fast_support'),('real_fast','native_real'),('real_fast_support','native_support'),('real_fast','whole'),('real_fast','own'),('real_fast','persistence')]
                item['roles'][role][group]=dict(patients=int(f.subject_id.nunique()),mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},comparisons={f'{a}_vs_{b}':paired(f,p[a],p[b],f'ratejepa-{seed}-{role}-{group}-{a}-{b}',1000) for a,b in comparisons})
        report['seeds'].append(item)
        progress=root/('jepa_real_fast_progress.json' if real_fast else ('jepa_fast_progress.json' if args.projection_fast else ('jepa_synthetic_progress.json' if args.synthetic_signal else ('jepa_clock_progress.json' if args.repair_clock else ('jepa_native_progress.json' if args.native_events else ('jepa_events_progress.json' if args.event_history else ('jepa_support_progress.json' if args.support_only else 'jepa_progress.json')))))));progress.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
