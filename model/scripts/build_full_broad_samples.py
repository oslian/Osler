"""Full source sample expansion; old matched examples retained byte-for-byte."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917');dest=root/'full_samples'
    out=Path('artifacts/native_reconstruction_full_broad_samples_20260918.json')
    if out.exists() or dest.exists():raise ValueError('Fresh build required')
    rp=Path('artifacts/native_reconstruction_broad_source_events_20260917.json')
    mp=Path('artifacts/native_reconstruction_matched_broad_samples_20260917.json')
    for report in (json.loads(rp.read_text()),json.loads(mp.read_text())):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Source changed')
    pp=root/'provenance.parquet';provenance=pd.read_parquet(pp)
    if provenance.groupby('uniquepid').role.nunique().max()!=1:raise ValueError('Patient roles overlap')
    oldip=root/'matched_samples/index.pt';old=torch.load(oldip,weights_only=False)
    reused={s['patient']:s for s in old['shards']}
    ep=root/'event_index.pt';shards=torch.load(ep,weights_only=False)
    for s in shards:
        if sha256(Path(s['path']))!=s['sha256']:raise ValueError('Events changed')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];columns={v:i for i,v in enumerate(variables)}
    cp=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    catalog=list(json.loads(cp.read_text())['target_counts'])
    counts={r:dict.fromkeys(catalog,0)for r in ('fit','encoder_tune')}
    proposed_new={r:dict.fromkeys(catalog,0)for r in counts}
    patients=sorted(provenance.uniquepid.unique());index=[];dest.mkdir();torch.set_num_threads(4)
    for start in range(0,len(patients),256):
        batch=patients[start:start+256];newpeople=[p for p in batch if p not in reused]
        subset=provenance.loc[provenance.uniquepid.isin(newpeople)]
        pieces=[]
        for s in shards:
            ids=subset.loc[subset.hospitalid.astype(int).eq(int(s['hospitalid'])),'stay_id'].tolist()
            if ids:pieces.append(pd.read_parquet(s['path'],filters=[('stay_id','in',ids)]))
        histories={}
        if pieces:
            events=pd.concat(pieces,ignore_index=True).drop(columns=['role','hospitalid'])
            events=events.merge(subset[['stay_id','uniquepid','role','hospitalid']],on='stay_id',validate='many_to_one')
            age=(events.event_time-events.measured_at).dt.total_seconds()/3600
            events['eligible']=age.between(0,1)&events.quality_valid.fillna(False).astype(bool)&events.variable.isin(catalog)&np.isfinite(events.value)
            histories=dict(tuple(events.groupby('uniquepid',sort=False)))
        for offset,patient in enumerate(batch):
            role=provenance.loc[provenance.uniquepid.eq(patient),'role'].iloc[0]
            if patient in reused:
                s=reused[patient];p=Path(s['path'])
                if sha256(p)!=s['sha256']or s['role']!=role:raise ValueError('Reused sample changed')
                samples=torch.load(p,weights_only=False)['samples'];index.append(dict(s,reused=True))
            else:
                samples={};history=histories.get(patient)
                if history is not None:
                    candidates=history.loc[history.eligible,['stay_id','event_time','variable']].drop_duplicates().sort_values(['variable','stay_id','event_time'])
                    seed=int.from_bytes(hashlib.sha256(('20260918:'+str(patient)).encode()).digest()[:8],'little')
                    rng=np.random.default_rng(seed);chosen=[]
                    for target,part in candidates.groupby('variable',sort=True):
                        chosen.append(part.iloc[int(rng.integers(len(part)))]);proposed_new[role][target]+=1
                    if chosen:
                        chosen=pd.DataFrame(chosen)
                        anchors=chosen[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
                        requested={(r.stay_id,pd.Timestamp(r.event_time),columns[r.variable])for r in chosen.itertuples()}
                        for e in reconstruction_examples(history.drop(columns=['eligible','uniquepid','role','hospitalid']),anchors,variables,assets.scaler.medians,assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
                            anchor=anchors.iloc[e['query_position']];c=e['target_index']
                            if (anchor.stay_id,pd.Timestamp(anchor.t),c)not in requested:continue
                            if c in samples:raise ValueError('Duplicate target')
                            samples[c]=e
                p=dest/f'patient_{start+offset:05d}.pt';torch.save(dict(patient=patient,role=role,samples=samples),p)
                index.append(dict(patient=patient,role=role,path=str(p),sha256=sha256(p),reused=False))
            for c in samples:counts[role][variables[c]]+=1
        print('Full-source patients',len(index),'/',len(patients),flush=True)
    torch.save(dict(shards=index,variables=variables),dest/'index.pt')
    paths=[Path(__file__),rp,mp,pp,oldip,ep,cp,dest/'index.pt',Path('whole_body_jepa/native_reconstruction_examples.py'),Path('whole_body_jepa/native_body_reconstruction.py')]
    out.write_text(json.dumps(dict(complete=True,patients=provenance.groupby('role').uniquepid.nunique().to_dict(),
        reused_patients=len(reused),accepted=counts,proposed_new=proposed_new,bindings={str(p):sha256(p)for p in paths},
        runtime_modified=False,promotion_allowed=False,limitation='Full existing source roles only. Old matched samples unchanged; new anchors outcome-blind hash-seeded, no redraw. No training or accuracy result.'),indent=2)+'\n')


if __name__=='__main__':main()
