"""Descriptive common-outcome comparison; no model or routing selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def describe(values, seed):
    values=np.asarray(values,float)
    if len(values)<2:
        return dict(patients=len(values),supported=False)
    rng=np.random.default_rng(seed)
    boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(4000)])
    loo=(values.sum()-values)/(len(values)-1)
    return dict(patients=len(values),supported=len(values)>=30,mean=float(values.mean()),
                ci95=np.quantile(boot,[.025,.975]).tolist(),benefit_fraction=float(np.mean(values<0)),
                leave_one_out_range=[float(loo.min()),float(loo.max())])


def errors(q, labels, column, p, own, full):
    frame=q[['stay_id','uniquepid']].copy()
    frame['future_id']=labels['future_source_id'][:,column]
    frame['truth']=labels['future'][:,column]
    for name,key in [('own',own),('full',full)]:
        frame[name]=np.abs(p[key]-frame.truth)
    # Reused future outcomes receive one weight before patient equalization.
    return frame.groupby(['stay_id','uniquepid','future_id'],as_index=False).agg(
        truth=('truth','first'),own=('own','mean'),full=('full','mean'))


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parents=[Path('artifacts/magnesium_native_anchor_tree_20260911.json'),Path('artifacts/magnesium_event_history_20260911.json')]
    for path in parents:
        for name,digest in json.loads(path.read_text())['binding'].items():
            assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_anchor_overlap_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    original=pd.read_parquet(root/'queries.parquet')
    old=dict(np.load(root/'multitarget_labels.npz'));v=list(old['variables']);k=v.index('magnesium')
    ok=old['paired'][:,k]&old['paired'][:,v.index('glucose')]
    oldq=original.loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    private=root/'magnesium_native_anchor_support'
    new=dict(np.load(private/'labels.npz'));valid=new['paired'][:,0]
    nq=pd.read_parquet(private/'queries.parquet').loc[valid].reset_index(drop=True)
    new={key:val[valid] for key,val in new.items()}
    items=[];bindings={str(p):sha256(p) for p in parents+[Path(__file__)]}
    for seed in (7,19,31,43,59,71,83):
        oldpath=root/f'magnesium_event_history/predictions_{seed}.npz'
        newpath=private/f'tree_predictions_{seed}.npz'
        op=dict(np.load(oldpath));npred=dict(np.load(newpath))
        oe=errors(oldq.loc[ok].reset_index(drop=True),{a:b[ok] for a,b in old.items() if a!='variables'},k,
                  {a:b[ok] for a,b in op.items()},'lag_own','lag_full')
        ne=errors(nq,new,0,npred,'own','full')
        joined=oe.merge(ne,on=['stay_id','uniquepid','future_id'],suffixes=('_old','_new'),validate='one_to_one')
        np.testing.assert_array_equal(joined.truth_old,joined.truth_new)
        roles=dict(np.load(root/f'partition_{seed}.npz'));entry=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            subjects=set(original.loc[roles[role],'uniquepid'])
            j=joined.loc[joined.uniquepid.isin(subjects)]
            by=j.groupby('uniquepid')[['own_old','full_old','own_new','full_new']].mean()
            allnew=ne.loc[ne.uniquepid.isin(subjects)].groupby('uniquepid')[['own','full']].mean()
            entry['roles'][role]=dict(common_events=len(j),common_patient_mae=by.mean().to_dict(),
                old_increment=describe(by.full_old-by.own_old,seed),
                new_increment=describe(by.full_new-by.own_new,seed),
                increment_change=describe((by.full_new-by.own_new)-(by.full_old-by.own_old),seed),
                all_native_event_equal_increment=describe(allnew.full-allnew.own,seed))
        for path in (oldpath,newpath,root/f'partition_{seed}.npz'):bindings[str(path)]=sha256(path)
        items.append(entry)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        status='descriptive_shared_future_outcomes_not_causal_timing_effect',seeds=items,binding=bindings),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,[(s['seed'],s['roles'][role]) for s in items])


if __name__=='__main__':main()
