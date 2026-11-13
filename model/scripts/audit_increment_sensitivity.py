"""Positive-control sensitivity; injected labels are never clinical evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/increment_sensitivity_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/insulin_potassium_pairs_20260911')
    prior=json.loads(Path('artifacts/insulin_potassium_partition_20260911.json').read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(root/name)!=prior['binding'][str(root/name)]:raise ValueError('cache changed')
    y=np.load(root/'labels.npz');valid=y['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    current=y['current'][valid,0]
    # Artificial independent input and additive label signal, fixed before fitting.
    rng=np.random.default_rng(911)
    signal=rng.choice([-1.,1.],len(q));nuisance=rng.choice([-1.,1.],len(q))
    q['truth']=y['future'][valid,0]+.5*signal
    own=np.column_stack([current,y['measurement_age'][valid,0]])
    report=dict(complete=False,status='artificial_positive_control',injection=.5,runtime_modified=False,promotion_allowed=False,
                limitation='two-column base, not full history or JEPA; no inference of real clinical signal',seeds=[])
    for entry in prior['seeds']:
        seed=entry['seed'];path=root/f'partition_{seed}.npz'
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        roles=dict(np.load(path));fit=roles['fit'];ids=q.loc[fit,'subject_id']
        w=1/ids.groupby(ids).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={}
        for name,extra in [('signal',signal),('nuisance',nuisance)]:
            x=np.column_stack([own,extra])
            m=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            m.fit(x[fit],(q.truth.to_numpy()-current)[fit],sample_weight=w)
            points[name]=current+m.predict(x)
        result=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            mask=roles[role];f=q.loc[mask]
            result['roles'][role]=dict(mae={k:patient_equal_mae(f,f.truth,p[mask]) for k,p in points.items()},comparison=paired(f,points['signal'][mask],points['nuisance'][mask],f'positive-{seed}-{role}',1000))
        report['seeds'].append(result)
    report['complete']=True
    report['binding']={str(p):sha256(p) for p in [Path(__file__),root/'labels.npz',root/'queries.parquet']}
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
