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
    output=Path('artifacts/native_body_feature_support_20260913.json')
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
    support={}
    table=[]
    for wi,window in enumerate((1,6,24,'all')):
        for i,name in enumerate(variables):
            block=wi*n+i
            counts_by_role={}
            for role,x in feature.items():
                event_counts=np.rint(np.expm1(x[:,16*n+3*block+1])).astype(int)
                counts_by_role[role]=dict(observed_patients=int((event_counts>=1).sum()),
                    repeated_patients=int((event_counts>=2).sum()))
            table.append(dict(variable=name,window=window,roles=counts_by_role))
    for key in ('observed_patients','repeated_patients'):
        fitcounts=np.array([r['roles']['fit'][key] for r in table])
        evalcounts=np.array([r['roles']['evaluation'][key]+r['roles']['hospital'][key] for r in table])
        support[key]=dict(total_variable_windows=len(table),zero_fit=int((fitcounts==0).sum()),
            fit_1_to_4=int(((fitcounts>=1)&(fitcounts<=4)).sum()),
            fit_5_to_19=int(((fitcounts>=5)&(fitcounts<=19)).sum()),
            fit_20_plus=int((fitcounts>=20).sum()),
            present_heldout_absent_fit=int(((fitcounts==0)&(evalcounts>0)).sum()))
    paths=[Path(__file__),Path('whole_body_jepa/native_body_reconstruction.py'),
           root/'queries.parquet',root/'labels.npz',root/'partition_7.npz']
    for d in ('input_shards','native_query_shards','unclipped_clock_shards'):
        paths.extend(sorted((root/d).glob('shard_*.npz')))
    output.write_text(json.dumps(dict(complete=True,patients=counts,summary=support,
        variable_window_support=table,runtime_modified=False,
        interpretation='support counts, not an effective-sample-size or power proof',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(support,indent=2))


if __name__=='__main__':
    main()

