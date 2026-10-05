"""Matched no-gap / explicit derived-gap / derived-support MAP experiments."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from scripts.train_map_native_missing import missing_view
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.derived_anion_events import VARIABLE
from whole_body_jepa.availability_inputs import snapshots
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def gap_view(x,derived,pack,mode):
    if mode not in ('no_gap','derived','derived_support'):
        raise ValueError('unknown gap mode')
    n=len(pack['variables'])
    gap=pack['variables'].index('anion_gap')
    v=missing_view(x,n,pack['target_index'],pack['allowed'],'full').reshape(len(x),3,-1,n)
    if mode=='no_gap':
        v[:,0,:,gap]=np.nan
        v[:,1,:,gap]=0
        v[:,2,:,gap]=7
    else:
        v[:,:,:,gap]=derived
        if mode=='derived_support':
            v[:,0,:,gap]=np.where(derived[:,1]>0,0,np.nan)
    return v.reshape(len(x),-1)


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_missing_training_20260912.json')
    derived_report=Path('artifacts/map_derived_anion_support_20260912.json')
    output=Path('artifacts/map_derived_gap_training_20260912.json')
    dest=root/'tree_derived_gap_trial'
    if dest.exists() or output.exists():
        raise ValueError('fresh outputs required')
    ref=json.loads(parent.read_text())
    for report in (ref,json.loads(derived_report.read_text()),json.loads(Path('artifacts/map_native_anchor_tree_20260912.json').read_text())):
        for p,h in report['bindings'].items():
            assert sha256(Path(p))==h,p
    labels=dict(np.load(root/'labels.npz'))
    valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q=q.rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    current=labels['current'][valid,0]
    x=np.load(root/'tree_trial/features.npy',mmap_mode='r')
    events=pd.read_parquet(root/'derived_anion_events.parquet')
    groups=dict(tuple(events.groupby('stay_id')))
    d=np.empty((len(q),3,len(HISTORY_LAGS_HOURS)),np.float32)
    for stay,f in q.groupby('stay_id',sort=False):
        requests=np.asarray([[t-pd.Timedelta(hours=h) for h in HISTORY_LAGS_HOURS] for t in f.t],dtype='datetime64[ns]')
        times=pd.DatetimeIndex(np.unique(requests))
        values,ages=snapshots(groups.get(stay,events.iloc[:0]),times,[VARIABLE],6)
        indices=np.searchsorted(times.to_numpy(),requests)
        d[f.index,0]=values[indices,0]
        d[f.index,1]=np.isfinite(values[indices,0])
        d[f.index,2]=np.nan_to_num(np.clip(ages[indices,0],0,168)/24,nan=7)
    dest.mkdir()
    dp=dest/'derived_features.npy'
    np.save(dp,d)
    bindings=[parent,derived_report,Path(__file__),dp,Path('scripts/train_map_native_missing.py'),Path('whole_body_jepa/availability_inputs.py')]
    results=[]
    for entry in ref['seeds']:
        seed=entry['seed']
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        fit=roles['fit']
        assert d[fit,1].any()
        subjects=q.loc[fit,'subject_id']
        w=1/subjects.groupby(subjects).transform('size').to_numpy(float)
        w*=len(w)/w.sum()
        pack=joblib.load(root/f'tree_trial/full_{seed}.joblib')
        scale=pack['target_scale']
        previous=dict(np.load(root/f'tree_missing_trial/predictions_{seed}.npz'))
        points={k:previous[k] for k in ('own','full','persistence')}
        for mode in ('no_gap','derived','derived_support'):
            model=clone(pack['model'])
            train=gap_view(x[fit],d[fit],pack,mode)
            model.fit(train,(q.truth.to_numpy()[fit]-current[fit])/scale,sample_weight=w)
            del train
            pred=np.empty(len(q))
            for start in range(0,len(q),4096):
                end=min(len(q),start+4096)
                pred[start:end]=current[start:end]+scale*model.predict(gap_view(x[start:end],d[start:end],pack,mode))
            points[mode]=pred
            path=dest/f'{mode}_{seed}.joblib'
            joblib.dump(dict(model=model,base_contract=pack,gap_mode=mode,
                replacement_variable=VARIABLE,replacement_units='mmol/L',replacement_is_derived=True),path)
            bindings.append(path)
            print(f'Seed {seed} {mode} complete',flush=True)
        scores={}
        for role in ('evaluation','hospital'):
            m=roles[role]
            frame=q.loc[m]
            p={k:v[m] for k,v in points.items()}
            for name in ('own','full','persistence'):
                np.testing.assert_allclose(patient_equal_mae(frame,frame.truth,p[name]),entry['roles'][role]['mae'][name],rtol=1e-12)
            scores[role]=dict(mae={k:patient_equal_mae(frame,frame.truth,v) for k,v in p.items()},
                comparisons={k:paired_model_errors(frame,p,k,seed=20260912) for k in ('own','full','no_gap','derived_support','persistence')})
        path=dest/f'predictions_{seed}.npz'
        np.savez_compressed(path,**points)
        bindings.append(path)
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,promotion_allowed=False,
        seeds=results,bindings={str(p):sha256(p) for p in bindings}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        rs=[r['roles'][role] for r in results]
        print(role,{k:float(np.median([r['mae'][k] for r in rs])) for k in points})
        for b in ('own','full','no_gap','derived_support','persistence'):
            cis=[r['comparisons'][b]['derived']['ci95'] for r in rs]
            print('derived vs',b,sum(c[1]<0 for c in cis),sum(c[0]>0 for c in cis))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
