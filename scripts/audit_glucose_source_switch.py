"""Trace measured glucose source labels; retrospective strata are not routes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/glucose_source_switch_20260911.json')
    if out.exists(): raise ValueError('fresh output required')
    reference=Path('artifacts/rate_glucose_jepa_20260911.json')
    ref=json.loads(reference.read_text())
    for name,digest in ref['binding'].items():
        if sha256(Path(name))!=digest: raise ValueError('input changed')
    meta=pd.read_parquet(root/'glucose_source_metadata.parquet')
    if meta.source_id.duplicated().any(): raise ValueError('duplicate source identity')
    lookup=meta.set_index('source_id').labname
    raw=pd.read_parquet(root/'body_labs.parquet')
    raw=raw.loc[raw.variable.eq('glucose')].merge(meta[['source_id','labname']],on='source_id',validate='one_to_one')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    l=np.load(root/'multitarget_labels.npz');k=list(l['variables']).index('glucose');eligible=l['paired'][:,k]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=l['future'][:,k]
    current=np.full(len(q),'unknown',object)
    future=pd.Series(l['future_source_id'][:,k]).map(lookup).fillna('unknown').to_numpy()
    groups={s:g for s,g in raw.groupby('stay_id')}
    for i in np.flatnonzero(eligible):
        row=q.iloc[i];g=groups[row.stay_id]
        measured=row.t-pd.Timedelta(hours=float(l['measurement_age'][i,k]))
        match=g.loc[(g.available_at<=row.t)&((g.measured_at-measured).dt.total_seconds().abs()<.01)&np.isclose(g.value,l['current'][i,k],atol=1e-5,rtol=0)]
        names=match.labname.unique()
        if len(names)==1: current[i]=names[0]
    known=(current!='unknown')&(future!='unknown')
    cases={'same_source':known&(current==future),'switch_source':known&(current!=future),'unresolved':~known}
    results=[]
    for e in ref['seeds']:
        seed=e['seed'];p=np.load(root/f'glucose_native_predictions_{seed}.npz');roles=np.load(root/f'partition_{seed}.npz');item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for name,condition in cases.items():
                mask=roles[role][valid]&eligible&condition;f=q.loc[mask]
                item['roles'][role][name]=dict(rows=int(mask.sum()),patients=int(f.subject_id.nunique()),
                    comparison=paired(f,p['real'][mask],p['own'][mask],f'glucose-source:{seed}:{role}:{name}',2000))
        results.append(item)
    pairs=pd.DataFrame({'current':current[eligible],'future':future[eligible]}).value_counts().reset_index(name='rows').to_dict('records')
    paths=(reference,root/'glucose_source_metadata.parquet',Path(__file__))
    out.write_text(json.dumps(dict(complete=True,status='retrospective_diagnostic',runtime_modified=False,
        future_source_used_for_routing=False,pair_counts=pairs,binding={str(p):sha256(p) for p in paths},seeds=results),indent=2)+'\n')
    print(pairs)
    for role in ('evaluation','hospital'):
        for name in cases:
            z=[s['roles'][role][name] for s in results]
            print(role,name,'patients',[g['patients'] for g in z],'delta',[g['comparison']['mean'] for g in z],
                'gain/harm',sum(g['comparison']['ci95'] is not None and g['comparison']['ci95'][1]<0 for g in z),
                sum(g['comparison']['ci95'] is not None and g['comparison']['ci95'][0]>0 for g in z))


if __name__=='__main__': main()
