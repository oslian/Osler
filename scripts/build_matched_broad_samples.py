"""Outcome-blind equal-patient/site source comparison and masked samples."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def matched_people(old, new):
    """Match role and deterministic representative site, not observed outcomes."""
    def people(frame):
        if frame.groupby('uniquepid').role.nunique().max()!=1:raise ValueError('Role overlap')
        return frame.sort_values('hospitalid').drop_duplicates('uniquepid')
    old,new=people(old),people(new)
    if set(old.uniquepid)&set(new.uniquepid):raise ValueError('Patient overlap')
    selected=[]
    for (role,site), group in old.groupby(['role','hospitalid']):
        pool=new.loc[new.role.eq(role)&new.hospitalid.eq(site),'uniquepid'].tolist()
        if len(pool)<len(group):raise ValueError('Insufficient matched site support')
        pool.sort(key=lambda p:hashlib.sha256(('20260917:'+str(p)).encode()).hexdigest())
        selected.extend(pool[:len(group)])
    return selected


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    dest=root/'matched_samples';out=Path('artifacts/native_reconstruction_matched_broad_samples_20260917.json')
    if dest.exists() or out.exists():raise ValueError('Fresh build required')
    report_path=Path('artifacts/native_reconstruction_broad_source_events_20260917.json')
    report=json.loads(report_path.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed source')
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    roles=torch.load(sp,weights_only=False)['roles']
    qp=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    old=pd.read_parquet(qp,columns=['uniquepid','hospitalid']).drop_duplicates();old['role']=old.uniquepid.map(roles)
    old=old.loc[old.role.isin(['fit','encoder_tune'])].copy();old['hospitalid']=old.hospitalid.astype(int)
    provenance=pd.read_parquet(root/'provenance.parquet');provenance['hospitalid']=provenance.hospitalid.astype(int)
    selected=set(matched_people(old,provenance))
    selected_provenance=provenance.loc[provenance.uniquepid.isin(selected)]
    if selected&set(roles):raise ValueError('Current patient leaked')
    dest.mkdir();selected_provenance.to_parquet(dest/'provenance.parquet',index=False)
    index_path=root/'event_index.pt';shards=torch.load(index_path,weights_only=False);pieces=[]
    for shard in shards:
        path=Path(shard['path'])
        if sha256(path)!=shard['sha256']:raise ValueError('Changed events')
        ids=selected_provenance.loc[selected_provenance.hospitalid.eq(int(shard['hospitalid'])),'stay_id'].tolist()
        if ids:pieces.append(pd.read_parquet(path,filters=[('stay_id','in',ids)]))
    events=pd.concat(pieces,ignore_index=True).drop(columns=['role','hospitalid'])
    events=events.merge(selected_provenance[['stay_id','uniquepid','role','hospitalid']],on='stay_id',validate='many_to_one')
    catalog_path=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    catalog=list(json.loads(catalog_path.read_text())['target_counts'])
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];columns={v:i for i,v in enumerate(variables)}
    age=(events.event_time-events.measured_at).dt.total_seconds()/3600
    events['eligible']=age.between(0,1)&events.quality_valid.fillna(False).astype(bool)&events.variable.isin(catalog)&np.isfinite(events.value)
    counts={r:dict.fromkeys(catalog,0) for r in ('fit','encoder_tune')};proposed={r:dict.fromkeys(catalog,0) for r in counts}
    rng=np.random.default_rng(20260917);index=[];torch.set_num_threads(4)
    for i,(patient,history) in enumerate(events.groupby('uniquepid',sort=True)):
        role=history.role.iloc[0]
        candidates=history.loc[history.eligible,['stay_id','event_time','variable']].drop_duplicates().sort_values(['variable','stay_id','event_time'])
        chosen=[]
        for target,part in candidates.groupby('variable',sort=True):
            chosen.append(part.iloc[int(rng.integers(len(part)))]);proposed[role][target]+=1
        samples={}
        if chosen:
            chosen=pd.DataFrame(chosen)
            anchors=chosen[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
            requested={(r.stay_id,pd.Timestamp(r.event_time),columns[r.variable]) for r in chosen.itertuples()}
            for e in reconstruction_examples(history.drop(columns=['eligible','uniquepid','role','hospitalid']),anchors,variables,assets.scaler.medians,assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
                anchor=anchors.iloc[e['query_position']];c=e['target_index']
                if (anchor.stay_id,pd.Timestamp(anchor.t),c) not in requested:continue
                if c in samples:raise ValueError('Duplicate target')
                samples[c]=e;counts[role][variables[c]]+=1
        path=dest/f'patient_{i:05d}.pt';torch.save(dict(patient=patient,role=role,samples=samples),path)
        index.append(dict(path=str(path),patient=patient,role=role,sha256=sha256(path)))
        if (i+1)%100==0:print('Matched patients',i+1,flush=True)
    torch.save(dict(shards=index,variables=variables),dest/'index.pt')
    paths=[Path(__file__),report_path,sp,qp,catalog_path,index_path,dest/'index.pt',dest/'provenance.parquet']
    out.write_text(json.dumps(dict(complete=True,selected_patients=selected_provenance.groupby('role').uniquepid.nunique().to_dict(),
        patients_with_events=len(index),proposed=proposed,accepted=counts,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Equal role and representative-site patient counts only; age/diagnosis/target support not matched. One fresh anchor per patient-target, no redraw. Full broad-source pool retained for later expansion. No training or accuracy claim.'),indent=2)+'\n')
    print('Matched sample build complete',len(index),flush=True)


if __name__=='__main__':main()
