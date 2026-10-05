"""Patient-equal raw-event reconstruction cache, with no outcome-based resampling."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    cache=root/'map_native_anchor_support'
    private=cache/'expanded_reconstruction_20260913'
    output=Path('artifacts/native_reconstruction_expanded_20260913.json')
    if private.exists() or output.exists():raise ValueError('fresh build required')
    private.mkdir()
    torch.set_num_threads(4)
    q=pd.read_parquet(cache/'queries.parquet');partition=dict(np.load(cache/'partition_7.npz'))
    roles={r:set(q.loc[partition[r],'uniquepid']) for r in ('fit','tune','evaluation','hospital')}
    if sum(map(len,roles.values()))!=len(set.union(*roles.values())):raise ValueError('patient overlap')
    link=q[['stay_id','uniquepid']].drop_duplicates()
    if link.stay_id.duplicated().any():raise ValueError('ambiguous stay')
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    events=pd.concat([original,bedside],ignore_index=True)
    events=events.loc[events.source_kind.isin(['lab_available','bedside_available'])]
    events=events.merge(link,on='stay_id',how='inner',validate='many_to_one')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];columns={v:i for i,v in enumerate(variables)}
    lag=(pd.to_datetime(events.event_time)-pd.to_datetime(events.measured_at)).dt.total_seconds()/3600
    events['candidate']=lag.between(0,1)&events.quality_valid.fillna(False).astype(bool)&np.isfinite(events.value)&events.variable.isin(variables)
    by_patient=events.groupby('uniquepid',sort=True)
    rng=np.random.default_rng(20260913);index=[];counts={};proposed={}
    for role,people in roles.items():
        counts[role]={v:0 for v in variables};proposed[role]={v:0 for v in variables}
        for number,patient in enumerate(sorted(people)):
            if patient not in by_patient.groups:continue
            history=by_patient.get_group(patient)
            candidates=history.loc[history.candidate,['stay_id','event_time','variable']].drop_duplicates().sort_values(['variable','stay_id','event_time'])
            selected=[]
            for variable,part in candidates.groupby('variable',sort=True):
                selected.append(part.iloc[int(rng.integers(len(part)))])
                proposed[role][variable]+=1
            if not selected:continue
            chosen=pd.DataFrame(selected)
            anchors=chosen[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
            requested={(row.stay_id,pd.Timestamp(row.event_time),columns[row.variable]) for row in chosen.itertuples()}
            samples={}
            for e in reconstruction_examples(history.drop(columns=['candidate','uniquepid']),anchors,variables,assets.scaler.medians,assets.scaler.scales,
                    assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
                anchor=anchors.iloc[e['query_position']];c=e['target_index']
                if (anchor.stay_id,pd.Timestamp(anchor.t),c) not in requested:continue
                if c in samples:raise ValueError('duplicate selected target')
                samples[c]=e;counts[role][variables[c]]+=1
            path=private/f'{role}_{number:05d}.pt'
            torch.save(dict(patient=patient,role=role,samples=samples),path)
            index.append(dict(path=str(path),role=role,patient=patient,sha256=sha256(path)))
            if (number+1)%50==0:print(role,number+1,'patients',flush=True)
        print('completed',role,sum(counts[role].values()),flush=True)
    torch.save(dict(shards=index,variables=variables,roles=roles),private/'index.pt')
    paths=[Path(__file__),Path('whole_body_jepa/native_reconstruction_examples.py'),Path('whole_body_jepa/native_history_schedule.py'),
           Path('whole_body_jepa/native_event_tokens.py'),Path('whole_body_jepa/native_body_reconstruction.py'),
           root/'native_events.parquet',root/'qualified_bedside_events.parquet',cache/'queries.parquet',cache/'partition_7.npz',
           Path('models/whole_body_joint_jepa.pt'),private/'index.pt']
    report=dict(complete=True,role_patients={r:len(p) for r,p in roles.items()},proposed_patient_targets=proposed,
                accepted_patient_targets=counts,all_role_minimum_30=[v for v in variables if all(counts[r][v]>=30 for r in roles)],
                source_eligible=[v for v in variables if counts['fit'][v]>=30 and counts['tune'][v]>=30],
                sampling='one raw-fresh availability anchor per patient-target; no redraw after rejection',
                runtime_modified=False,promotion_allowed=False,bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print('complete',flush=True)


if __name__=='__main__':main()
