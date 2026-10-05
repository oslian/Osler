"""Outcome-free observation support from exact masked reconstruction histories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.reconstruction_cache import ReconstructionCache, unpack_example
from whole_body_jepa.multisource_intervals import sha256


def support(example,n):
    x=example['tokens'].numpy();times=example['event_hours'].numpy()
    anchor=example['anchor_hour']
    if len(x)==0 or not np.isfinite(times).all() or (times>anchor+1e-8).any():raise ValueError('Invalid known-time history')
    m=x[:,n:2*n]
    if not np.isin(m,[0,1]).all():raise ValueError('Invalid mask')
    mask=m.astype(bool)
    if mask[:,example['target_index']].any():raise ValueError('Target visible')
    ages=anchor-times[:,None]+24*x[:,2*n:3*n]
    observed=mask.any(0)
    if not observed.any() or not np.isfinite(ages[mask]).all() or (ages[mask]<-1e-6).any():raise ValueError('Invalid measured age')
    latest=np.where(mask,ages,np.inf).min(0)[observed]
    return dict(event_steps=len(times),observed_events=int(mask.sum()),variables_observed=int(observed.sum()),
        fresh_1h_variables=int((latest<=1+1e-6).sum()),fresh_6h_variables=int((latest<=6+1e-6).sum()),
        median_latest_age_hours=float(np.median(latest)),max_latest_age_hours=float(latest.max()),
        history_span_hours=float(anchor-times.min()),latest_available_age_hours=float(anchor-times.max()))


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    cp=root/'full_histories_20260922.sqlite';sp=root/'scaling_20260922.pt'
    rp=Path('artifacts/native_reconstruction_additional_full_cache_20260922.json')
    report=json.loads(rp.read_text())
    if sha256(cp)!=report['private_sha256']:raise ValueError('Changed source cache')
    variables=torch.load(sp,weights_only=False)['variables'];cache=ReconstructionCache(cp)
    dest=root/'observation_support_20260925.parquet'
    out=Path('artifacts/native_reconstruction_additional_observation_support_20260925.json')
    if dest.exists() or out.exists():raise ValueError('Existing support extraction')
    rows=[]
    try:
        for i,(patient,c,payload,digest) in enumerate(cache.db.execute('SELECT patient,target,payload,digest FROM examples ORDER BY patient,target'),1):
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Corrupt history')
            e=unpack_example(payload)
            if e['target_index']!=c:raise ValueError('Target mismatch')
            rows.append(dict(patient=patient,target=variables[c],**support(e,len(variables))))
            if i%100000==0:print('Support extraction',i,flush=True)
    finally:cache.close()
    frame=pd.DataFrame(rows)
    if len(frame)!=1800214 or frame.duplicated(['patient','target']).any():raise ValueError('Support identity mismatch')
    tmp=dest.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(dest)
    result=dict(pairs=len(frame),patients=int(frame.patient.nunique()),targets=int(frame.target.nunique()),
        prediction_targets_used_as_features=False,private_sha256=sha256(dest),
        bindings={str(p):sha256(p) for p in (Path(__file__),cp,sp,rp,Path('whole_body_jepa/reconstruction_cache.py'))},
        runtime_modified=False,promotion_allowed=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'}),flush=True)


if __name__=='__main__':main()
