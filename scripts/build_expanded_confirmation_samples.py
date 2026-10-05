"""One frozen fresh anchor per new confirmation patient-target, no redraw."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    source=Path('artifacts/native_reconstruction_expanded_confirmation_events_20260917.json')
    output=Path('artifacts/native_reconstruction_expanded_confirmation_samples_20260917.json')
    root=Path('data/research_recovery/expanded_confirmation_20260917')
    dest=root/'samples'
    if output.exists() or dest.exists():raise ValueError('Fresh sample build required')
    report=json.loads(source.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed event cache')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];columns={v:i for i,v in enumerate(variables)}
    catalog_path=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    catalog=list(json.loads(catalog_path.read_text())['target_counts'])
    provenance=pd.read_parquet(root/'provenance.parquet')
    events=pd.read_parquet(root/'events.parquet').merge(provenance[['stay_id','uniquepid','hospitalid']],on='stay_id',validate='many_to_one')
    age=(pd.to_datetime(events.event_time)-pd.to_datetime(events.measured_at)).dt.total_seconds()/3600
    events['eligible']=age.between(0,1)&events.quality_valid.fillna(False).astype(bool)&events.variable.isin(catalog)&np.isfinite(events.value)
    proposed=dict.fromkeys(catalog,0);accepted=dict.fromkeys(catalog,0);sites={t:{} for t in catalog}
    dest.mkdir();index=[];rng=np.random.default_rng(20260917);torch.set_num_threads(4)
    for number,(patient,history) in enumerate(events.groupby('uniquepid',sort=True)):
        candidates=history.loc[history.eligible,['stay_id','event_time','variable','hospitalid']].drop_duplicates().sort_values(['variable','stay_id','event_time'])
        chosen=[]
        for target,part in candidates.groupby('variable',sort=True):
            chosen.append(part.iloc[int(rng.integers(len(part)))]);proposed[target]+=1
        if not chosen:continue
        chosen=pd.DataFrame(chosen)
        anchors=chosen[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
        requested={(r.stay_id,pd.Timestamp(r.event_time),columns[r.variable]):str(r.hospitalid) for r in chosen.itertuples()}
        samples={};sample_sites={}
        for e in reconstruction_examples(history.drop(columns=['eligible','uniquepid','hospitalid']),anchors,variables,assets.scaler.medians,assets.scaler.scales,
                assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
            anchor=anchors.iloc[e['query_position']];c=e['target_index'];key=(anchor.stay_id,pd.Timestamp(anchor.t),c)
            if key not in requested:continue
            if c in samples:raise ValueError('Repeated patient-target')
            if bool((e['event_hours']>e['anchor_hour']).any()):raise ValueError('Future history')
            samples[c]=e;sample_sites[c]=requested[key];target=variables[c];accepted[target]+=1
            sites[target][requested[key]]=sites[target].get(requested[key],0)+1
        path=dest/f'patient_{number:05d}.pt'
        torch.save(dict(patient=patient,role='expanded_readout_confirmation',samples=samples,sample_sites=sample_sites),path)
        index.append(dict(path=str(path),patient=patient,sha256=sha256(path)))
        if (number+1)%100==0:print('Processed patients',number+1,flush=True)
    index_path=dest/'index.pt';torch.save(dict(shards=index,variables=variables),index_path)
    paths=[Path(__file__),source,catalog_path,root/'events.parquet',root/'provenance.parquet',index_path,
        Path('whole_body_jepa/native_reconstruction_examples.py'),Path('whole_body_jepa/native_history_schedule.py'),
        Path('whole_body_jepa/native_event_tokens.py'),Path('whole_body_jepa/native_body_reconstruction.py'),Path('models/whole_body_joint_jepa.pt')]
    result=dict(complete=True,patients_with_shards=len(index),proposed=proposed,accepted=accepted,
        per_target_site_counts={t:sorted(s.values()) for t,s in sites.items()},
        sampling='One uniformly selected fresh raw anchor per patient-target with fixed seed; no redraw after rejection; existing conflict, target/sibling mask and 48h history rules.',
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Expanded reused source-confirmation samples only. No accuracy scoring or training; prior project exposure not ruled out.')
    output.write_text(json.dumps(result,indent=2)+'\n');print('Completed',sum(accepted.values()),'pairs',flush=True)


if __name__=='__main__':main()
