"""Masked raw-history probe independent of neural baseline support selection."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import configure, teacher_target
from whole_body_joint_jepa import _masked_latent_prediction_loss
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from osler_jepa.organ_schema import canonical_organ_adjacency


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/native_body_fit_range_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz')); roles=dict(np.load(root/'partition_7.npz'))
    hours=[np.array([np.nan])]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            hours.append(np.load(root/'unclipped_clock_shards'/p.name)['event_hours'])
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q)); hours=np.concatenate(hours)
    assets=load_release(Path('models')); variables=assets.checkpoint['variables']
    n=len(variables)
    torch.set_num_threads(4)
    rng=np.random.default_rng(20260913); selected={};data={}
    fresh=np.isfinite(labels['current'][:,0])&np.isfinite(labels['measurement_age'][:,0])&(labels['measurement_age'][:,0]>=0)&(labels['measurement_age'][:,0]<=1)
    for role in ('fit','tune','evaluation','hospital'):
        groups=[np.asarray(g.index) for _,g in q.loc[roles[role]&fresh].groupby('subject_id',sort=True)]
        selected[role]=[int(rng.choice(g)) for g in groups]
        for row in selected[role]:
            seq=raw.sequences[row];times=torch.from_numpy(hours[seq])
            tokens=torch.from_numpy(raw.input[seq].copy())
            anchor=float((pd.Timestamp(q.iloc[row]['t'])-pd.Timestamp('2100-01-01')).total_seconds()/3600)
            if not torch.isfinite(times).all() or (times>anchor).any() or (times[1:]<=times[:-1]).any():
                raise ValueError('invalid original times')
            data[row]=((tokens,times,anchor),)

    from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target
    counts={}
    for role,rows in selected.items():
        valid=[]
        for r in rows:
            if not np.isfinite(labels['measurement_age'][r,0]) or labels['measurement_age'][r,0]>1:continue
            try: mask_native_target(data[r][0][0],data[r][0][1],variables,'map')
            except ValueError as e:
                if str(e)=='no independent observed history remains':continue
                raise
            valid.append(r)
        selected[role]=valid;counts[role]=len(valid)
        if len(valid)<2:raise ValueError('insufficient supported role')
    q['truth']=labels['current'][:,0]
    population=float(np.median(q.loc[selected['fit'],'truth']))
    rng=np.random.default_rng(20260913)
    donor={}
    for role,rows in selected.items():
        order=np.array(rows)[rng.permutation(len(rows))]
        donor[role]=dict(zip(order,np.roll(order,1)))
        assert all(k!=v for k,v in donor[role].items())
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    feature={}
    for role,rows in selected.items():
        x=[]
        for r in rows:
            args=data[r][0];tokens,times=mask_native_target(args[0],args[1],variables,'map')
            t=times.numpy();v=tokens.numpy();values=[];support=[]
            for window in (1.,6.,24.,float('inf')):
                for i in range(n):
                    observed=(v[:,n+i]>0)&(t>=args[2]-window)
                    z=v[observed,i];ts=t[observed]
                    if len(z):
                        centered=ts-ts.mean()
                        slope=float(np.dot(centered,z-z.mean())/np.dot(centered,centered)) if np.dot(centered,centered)>0 else 0.
                        values.extend([float(z[-1]),float(z.mean()),float(z.std()),slope])
                        support.extend([1.,float(np.log1p(len(z))),float(args[2]-ts[-1])])
                    else:values.extend([0.]*4);support.extend([0.,0.,0.])
            x.append(values+support)
        feature[role]=np.asarray(x)
    lower=feature['fit'][:,:16*n].min(axis=0)
    upper=feature['fit'][:,:16*n].max(axis=0)
    models={};boards={};preds={}
    for arm in ('raw','support','placebo','incumbent','bounded','bounded_placebo'):
        inputs={role:x.copy() for role,x in feature.items()}
        if arm=='support':
            for x in inputs.values():x[:,:16*n]=0
        if arm=='incumbent':
            inputs={role:np.concatenate([x[:,12*n:16*n],x[:,25*n:28*n]],axis=1)
                    for role,x in inputs.items()}
        if arm in ('placebo','bounded_placebo'):
            for role,rows in selected.items():
                index={r:i for i,r in enumerate(rows)}
                inputs[role]=feature[role][[index[donor[role][r]] for r in rows]]
        if arm in ('bounded','bounded_placebo'):
            for x in inputs.values():
                x[:,:16*n]=np.clip(x[:,:16*n],lower,upper)
        fit_y=q.loc[selected['fit'],'truth'].to_numpy();tune_y=q.loc[selected['tune'],'truth'].to_numpy()
        best=(float(np.mean(abs(population-tune_y))),None);board=[]
        for alpha in (1.,10.,100.,1000.,10000.):
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
            model.fit(inputs['fit'],fit_y)
            mae=float(np.mean(abs(model.predict(inputs['tune'])-tune_y)))
            board.append(dict(alpha=alpha,tune_mae=mae))
            if mae<best[0]:best=(mae,model)
        models[arm]=best[1]
        boards[arm]=dict(board=board,selected='population' if best[1] is None else str(best[1][-1].alpha),tune_mae=best[0])
        preds[arm]={role:np.full(len(selected[role]),population) if best[1] is None else best[1].predict(inputs[role])
                    for role in ('evaluation','hospital')}
    results={}
    for role in ('evaluation','hospital'):
        frame=q.loc[selected[role]];p={a:preds[a][role] for a in preds};p['population']=np.full(len(frame),population)
        results[role]=dict(mae={k:float(np.mean(abs(v-frame.truth.to_numpy()))) for k,v in p.items()},
            comparisons={ref:paired_model_errors(frame,p,ref,seed=20260913,repetitions=20000) for ref in ('population','support','placebo','incumbent','raw','bounded_placebo')})
    tail={}
    names=[f'{v}/{window}h/{stat}' for window in (1,6,24,'all') for v in variables
           for stat in ('latest','mean','std','slope')]
    names += [f'{v}/{window}h/{stat}' for window in (1,6,24,'all') for v in variables
              for stat in ('observed','count','age')]
    rawmodel=models['raw']
    for role in ('evaluation','hospital'):
        errors=np.abs(preds['raw'][role]-q.loc[selected[role],'truth'].to_numpy())
        worst=np.argsort(errors)[-10:]
        contribution=rawmodel[0].transform(feature[role])*rawmodel[-1].coef_
        score=np.mean(np.abs(contribution[worst]),axis=0)
        largest=np.argsort(score)[-10:][::-1]
        tail[role]=dict(error_quantiles=np.quantile(errors,[.5,.9,.99,1.]).tolist(),
            top_one_error_fraction=float(errors.max()/errors.sum()),
            top_ten_error_fraction=float(errors[worst].sum()/errors.sum()),
            top_ten_patients_mean_absolute_feature_contributions=[
                dict(feature=names[i],magnitude=float(score[i])) for i in largest])
    cpk=variables.index('cpk')
    support_diagnostic={}
    for role,rows in selected.items():
        spans=[];slopes=[]
        for row in rows:
            tokens,times,anchor=data[row][0];v=tokens.numpy();t=times.numpy()
            use=(v[:,n+cpk]>0)&(t>=anchor-6)
            if use.sum()>=2:
                ts=t[use];z=v[use,cpk];c=ts-ts.mean()
                spans.append(float(ts.max()-ts.min()))
                slopes.append(float(np.dot(c,z-z.mean())/np.dot(c,c)))
        support_diagnostic[role]=dict(two_event_patients=len(spans),
            span_hour_quantiles=np.quantile(spans,[0,.5,1]).tolist() if spans else [],
            absolute_slope_quantiles=np.quantile(np.abs(slopes),[0,.5,1]).tolist() if slopes else [],
            numeric_out_of_fit_range_patient_fraction=float(
                ((feature[role][:,:16*n]<lower)|(feature[role][:,:16*n]>upper)).any(1).mean()),
            bounded_error_quantiles=np.quantile(np.abs(preds['bounded'][role]-q.loc[rows,'truth'].to_numpy()),[.5,.99,1]).tolist()
                if role in preds['bounded'] else None)
    print(json.dumps(tail),flush=True)
    previous=json.loads(Path('artifacts/native_body_unrestricted_raw_20260913.json').read_text())
    if counts!=previous['patients']:raise ValueError('cohort changed')
    for role in results:
        if abs(results[role]['mae']['incumbent']-previous['roles'][role]['mae']['raw'])>1e-9:
            raise ValueError('incumbent did not reproduce')
    paths=[Path(__file__),Path('NATIVE_BODY_FIT_RANGE_PROTOCOL_20260913.md'),
        root/'queries.parquet',root/'labels.npz',root/'partition_7.npz',
        Path('whole_body_jepa/native_body_reconstruction.py')]
    for directory in ('input_shards','native_query_shards','unclipped_clock_shards'):
        paths.extend(sorted((root/directory).glob('shard_*.npz')))
    output.write_text(json.dumps(dict(complete=True,patients=counts,selection=boards,roles=results,tail=tail,support_diagnostic=support_diagnostic,
        runtime_modified=False,status='diagnostic_only',task='expanded_current_only_masked_raw_history_nowcast',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(dict(selection=boards,roles=results)),flush=True)


if __name__=='__main__':main()






