"""Broad source cohorts, bounded per-hospital canonicalization, no test inputs."""
import json
from pathlib import Path
import duckdb
import pandas as pd
import torch
from extractors.eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from extractors.mimiciv_observation_transition_extract import EXTRA_PLAUSIBLE
from whole_body_jepa.availability_inputs import UNITS, filter_units
from whole_body_jepa.bedside_events import LABELS, nursing_events
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_broad_source_events_20260917.json')
    private=Path('data/research_recovery/broad_source_20260917')
    if out.exists() or private.exists():raise ValueError('Fresh source extraction required')
    inventory_path=Path('artifacts/native_reconstruction_expansion_inventory_20260917.json')
    inventory=json.loads(inventory_path.read_text())
    for f,h in inventory['bindings'].items():
        if sha256(Path(f))!=h:raise ValueError('Changed provenance')
    root=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    patients=pd.read_csv(root/'patient.csv.gz',usecols=['patientunitstayid','uniquepid','hospitalid'],dtype=str)
    split=torch.load('data/research_recovery/reconstruction_nested_selection_20260917.pt',weights_only=False)
    q=pd.read_parquet('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet',columns=['uniquepid','hospitalid'])
    site_roles={str(int(r.hospitalid)):split['roles'][r.uniquepid] for r in q.itertuples()}
    a=pd.read_parquet('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet',columns=['subject_id','hospitalid'])
    excluded=set(split['roles'])|set(a.subject_id.astype(str));ancestral_sites=set(a.hospitalid.astype(int).astype(str))
    roles={}
    for person,sites in patients.groupby('uniquepid').hospitalid.agg(set).items():
        known={site_roles.get(s) for s in sites}
        if person not in excluded and not sites&ancestral_sites and len(known)==1:
            role=next(iter(known))
            if role in ('fit','encoder_tune'):roles[person]=role
    selected=patients.loc[patients.uniquepid.isin(roles)].copy()
    selected['role']=selected.uniquepid.map(roles);selected['stay_id']='eicu:'+selected.patientunitstayid
    for role in ('fit','encoder_tune'):
        if selected.loc[selected.role==role,'uniquepid'].nunique()!=inventory['role_candidates'][role]['new_patients_with_full_history_inside_role']:
            raise ValueError('Inventory mismatch')
    private.mkdir();selected.to_parquet(private/'provenance.parquet',index=False)
    variables=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)['variables']
    mapping=pd.DataFrame([(k,v) for k,v in LAB_TO_STATE.items() if v in variables and v in UNITS],columns=['name','variable'])
    labels=pd.DataFrame(LABELS,columns=['label','name','variable','unit'])
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='2GB'")
    db.register('selected',selected);db.register('mapping',mapping);db.register('labels',labels)
    print('Scanning broad source lab events',flush=True)
    labs=private/'raw_labs.parquet'
    db.read_csv(str(root/'lab.csv.gz'),header=True).create_view('source_labs')
    db.sql('''SELECT s.stay_id,s.role,s.hospitalid,cast(l.labid as varchar) source_id,m.variable,
        l.labname source_test_name,
        timestamp '2100-01-01'+l.labresultoffset*interval '1 minute' measured_at,
        timestamp '2100-01-01'+greatest(l.labresultoffset,l.labresultrevisedoffset)*interval '1 minute' available_at,
        cast(l.labresult as double) AS value,coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface) unit
        FROM source_labs l JOIN selected s ON cast(l.patientunitstayid as varchar)=s.patientunitstayid
        JOIN mapping m ON l.labname=m.name WHERE l.labresultrevisedoffset IS NOT NULL''').write_parquet(str(labs))
    print('Scanning broad source bedside events',flush=True)
    nursing=private/'raw_nursing.parquet'
    db.read_csv(str(root/'nurseCharting.csv.gz'),header=True,all_varchar=True).create_view('nursing')
    db.sql('''SELECT n.*,s.role,s.hospitalid FROM nursing n JOIN selected s ON n.patientunitstayid=s.patientunitstayid
        JOIN labels m ON n.nursingchartcelltypevallabel=m.label AND n.nursingchartcelltypevalname=m.name
        WHERE n.nursingchartcelltypecat='Vital Signs' AND try_cast(n.nursingchartvalue AS DOUBLE) IS NOT NULL''').write_parquet(str(nursing))
    db.close();shards=[];summary={r:dict(patients=int(selected.loc[selected.role==r,'uniquepid'].nunique()),events=0,quality_events=0) for r in ('fit','encoder_tune')}
    for number,((role,site),_) in enumerate(selected.groupby(['role','hospitalid'],sort=True)):
        l=pd.read_parquet(labs,filters=[('role','==',role),('hospitalid','==',site)])
        l=filter_units(l,{**PLAUSIBLE,**EXTRA_PLAUSIBLE});l['source_kind']='lab_available';l['quality_valid']=True
        n=pd.read_parquet(nursing,filters=[('role','==',role),('hospitalid','==',site)])
        n=nursing_events(n);n['quality_valid']=False
        for v,g in n.groupby('variable'):n.loc[g.index,'quality_valid']=g.value.between(*PLAUSIBLE[v])
        n['role']=role;n['hospitalid']=site
        events=pd.concat([l,n],ignore_index=True);events['event_time']=events.available_at
        path=private/f'events_{number:03d}.parquet';events.to_parquet(path,index=False)
        shards.append(dict(path=str(path),role=role,hospitalid=site,sha256=sha256(path)))
        summary[role]['events']+=len(events);summary[role]['quality_events']+=int(events.quality_valid.sum())
        print('Canonicalized source shard',number+1,role,len(events),flush=True)
    torch.save(shards,private/'event_index.pt')
    paths=[Path(__file__),inventory_path,root/'lab.csv.gz',root/'nurseCharting.csv.gz',private/'provenance.parquet',
        labs,nursing,private/'event_index.pt',Path('whole_body_jepa/availability_inputs.py'),Path('whole_body_jepa/bedside_events.py'),
        Path('whole_body_jepa/raw_lab_replay.py'),Path('extractors/eicu_sepsis_transition_extract.py'),Path('extractors/mimiciv_observation_transition_extract.py'),Path('models/whole_body_joint_jepa.pt')]
    out.write_text(json.dumps(dict(complete=True,summary=summary,shards=len(shards),bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Source-only event extraction, not sample acceptance or training. No insulin-rate inclusion criterion; no confirmation/evaluation/reserved patients. Prior project exposure not exhaustively ruled out.'),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
