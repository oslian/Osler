"""Inventory reconstruction truth at native events, not MAP query anchors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    out=Path('artifacts/native_multitarget_support_20260913.json')
    if out.exists():raise ValueError('fresh output required')
    paths=[root/'queries.parquet',root/'partition_7.npz',Path(__file__),
           Path('whole_body_jepa/nowcast.py')]
    q=pd.read_parquet(paths[0]);roles=dict(np.load(paths[1]));hours=[np.array([np.nan])]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            clock=root/'unclipped_clock_shards'/p.name
            native=root/'native_query_shards'/p.name
            paths.extend([p,clock,native])
            with np.load(clock) as c:hours.append(c['event_hours'])
            with np.load(p) as a,np.load(native) as b:yield dict(a),dict(b)
    raw=RaggedNativeBatch(shards(),len(q));hours=np.concatenate(hours)
    variables=load_release(Path('models')).checkpoint['variables'];n=len(variables)
    hidden=[np.array([j for j,v in enumerate(variables) if v in {t}|related_siblings(t)]) for t in variables]
    counts={role:dict(patients=0,observed_events=np.zeros(n,int),
        reconstructable_events=np.zeros(n,int),supported_patients=np.zeros(n,int)) for role in roles}
    owner=np.full(len(raw.input),-1,int);cache_copies=0
    for patient,(_,g) in enumerate(q.groupby('uniquepid',sort=True)):
        membership=[r for r,m in roles.items() if m[g.index].any()]
        if len(membership)!=1:raise ValueError('patient role overlap')
        role=membership[0]
        if not roles[role][g.index].all():raise ValueError('partial patient role')
        c=counts[role];c['patients']+=1;patient_supported=np.zeros(n,bool)
        for _,stay in g.groupby('stay_id',sort=True):
            seq=np.unique(np.concatenate([raw.sequences[i] for i in stay.index]))
            if (owner[seq]>=0).any():raise ValueError('event token shared across encounters')
            owner[seq]=patient
            order=np.argsort(hours[seq],kind='stable');seq=seq[order];t=hours[seq]
            if not np.isfinite(t).all():raise ValueError('nonfinite time')
            keep=np.r_[True,np.diff(t)>0]
            for j in np.flatnonzero(~keep):
                if not np.array_equal(raw.input[seq[j]],raw.input[seq[j-1]],equal_nan=True):
                    different=~np.isclose(raw.input[seq[j]],raw.input[seq[j-1]],rtol=0,atol=0,equal_nan=True)
                    raise ValueError(f'same-time cache copies disagree at columns {np.flatnonzero(different).tolist()} of {raw.input.shape[1]}')
            cache_copies+=int((~keep).sum());seq=seq[keep]
            mask=raw.input[seq,n:2*n]>0
            if not np.isfinite(raw.input[seq,:n][mask]).all():raise ValueError('invalid measured value')
            prefix=np.cumsum(mask,axis=0);total=prefix.sum(axis=1)
            c['observed_events']+=mask.sum(axis=0)
            for i in range(n):
                # Same event's independent variables are known; later events are not.
                independent=total-prefix[:,hidden[i]].sum(axis=1)
                valid=mask[:,i]&(independent>0)
                c['reconstructable_events'][i]+=valid.sum()
                patient_supported[i]|=valid.any()
        c['supported_patients']+=patient_supported
    table=[]
    for i,v in enumerate(variables):
        table.append(dict(variable=v,roles={r:{k:int(a[i]) for k,a in c.items() if k!='patients'}
            for r,c in counts.items()}))
    report=dict(complete=True,cohort='existing MAP-cohort native event cache, not all eICU',
        native_tokens=int((owner>=0).sum()),identical_same_time_cache_copies=cache_copies,
        patients={r:c['patients'] for r,c in counts.items()},
        targets=table,task='current native-event masked reconstruction support only',
        runtime_modified=False,bindings={str(p):sha256(p) for p in paths})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(patients=report['patients'],native_tokens=report['native_tokens'],
        supported_targets_all_roles=sum(all(t['roles'][r]['supported_patients']>=20 for r in counts) for t in table),
        by_target={t['variable']:{r:t['roles'][r]['supported_patients'] for r in counts} for t in table}),indent=2))


if __name__=='__main__':main()
