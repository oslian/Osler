"""Durable source-confirmation event cache with frozen provenance and extractors."""
import json
from pathlib import Path
import duckdb
import pandas as pd
import torch
from scripts.audit_availability_jepa import extract
from whole_body_jepa.bedside_events import nursing_events, LABELS
from extractors.eicu_sepsis_transition_extract import PLAUSIBLE
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_expanded_confirmation_events_20260917.json')
    private=Path('data/research_recovery/expanded_confirmation_20260917')
    if out.exists() or private.exists():raise ValueError('Fresh cache required')
    inventory_path=Path('artifacts/native_reconstruction_expansion_inventory_20260917.json')
    inventory=json.loads(inventory_path.read_text())
    for p,h in inventory['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed provenance')
    raw=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    p=pd.read_csv(raw/'patient.csv.gz',usecols=['patientunitstayid','uniquepid','hospitalid'],dtype=str)
    roles=torch.load('data/research_recovery/reconstruction_nested_selection_20260917.pt',weights_only=False)['roles']
    q=pd.read_parquet('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet',columns=['uniquepid','hospitalid'])
    site_roles={str(int(r.hospitalid)):roles[r.uniquepid] for r in q.itertuples()}
    a=pd.read_parquet('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet',columns=['subject_id','hospitalid'])
    excluded=set(roles)|set(a.subject_id.astype(str)); ancestral_sites=set(a.hospitalid.astype(int).astype(str))
    people={person for person,sites in p.groupby('uniquepid').hospitalid.agg(set).items()
            if person not in excluded and not sites&ancestral_sites and {site_roles.get(s) for s in sites}=={'readout_tune'}}
    if len(people)!=inventory['role_candidates']['readout_tune']['new_patients_with_full_history_inside_role']:
        raise ValueError('Candidate count changed')
    selected=p.loc[p.uniquepid.isin(people)].copy()
    selected['stay_id']='eicu:'+selected.patientunitstayid
    private.mkdir()
    selected.to_parquet(private/'provenance.parquet',index=False)
    checkpoint=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)
    variables=checkpoint['variables']
    print('Extracting qualified labs',flush=True)
    labs,_,audit=extract('eicu',set(selected.stay_id),variables,output=private/'labs.parquet')
    labs['source_kind']='lab_available';labs['quality_valid']=True;labs['event_time']=labs.available_at
    print('Extracting mapped nursing vitals',flush=True)
    db=duckdb.connect();db.execute('SET threads=2');db.execute("SET memory_limit='2GB'")
    db.register('selected',selected)
    mapping=pd.DataFrame(LABELS,columns=['label','name','variable','unit']);db.register('mapping',mapping)
    db.read_csv(str(raw/'nurseCharting.csv.gz'),header=True,all_varchar=True).create_view('nursing')
    db.sql('''SELECT n.* FROM nursing n JOIN selected s ON n.patientunitstayid=s.patientunitstayid
        JOIN mapping m ON n.nursingchartcelltypevallabel=m.label AND n.nursingchartcelltypevalname=m.name
        WHERE n.nursingchartcelltypecat='Vital Signs' AND try_cast(n.nursingchartvalue AS DOUBLE) IS NOT NULL''').write_parquet(str(private/'nursing_raw.parquet'))
    db.close()
    bedside=nursing_events(pd.read_parquet(private/'nursing_raw.parquet'))
    bedside['quality_valid']=False
    for v,g in bedside.groupby('variable'):
        bedside.loc[g.index,'quality_valid']=g.value.between(*PLAUSIBLE[v])
    bedside['event_time']=bedside.available_at
    bedside.to_parquet(private/'bedside.parquet',index=False)
    events=pd.concat([labs,bedside],ignore_index=True)
    events.to_parquet(private/'events.parquet',index=False)
    counts=events.groupby(['source_kind','variable']).agg(rows=('value','size'),quality_rows=('quality_valid','sum'),stays=('stay_id','nunique')).reset_index().to_dict('records')
    paths=[Path(__file__),inventory_path,raw/'lab.csv.gz',raw/'nurseCharting.csv.gz',
        Path('scripts/audit_availability_jepa.py'),Path('whole_body_jepa/bedside_events.py'),
        Path('whole_body_jepa/availability_inputs.py'),Path('whole_body_jepa/raw_lab_replay.py'),
        Path('extractors/eicu_sepsis_transition_extract.py'),Path('extractors/mimiciv_observation_transition_extract.py'),Path('models/whole_body_joint_jepa.pt')]
    paths+=list(private.glob('*.parquet'))
    result=dict(complete=True,patients=len(people),stays=len(selected),events=len(events),counts=counts,lab_audit=audit,
        bindings={str(f):sha256(f) for f in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Event cache only. Reconstruction examples, conflicting simultaneous measurements and block-mask acceptance are not yet evaluated. New cohort from reused source hospitals, not untouched external data. No fit/evaluation/reserved patient data included.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('Completed events:',len(events),'patients:',len(people),flush=True)


if __name__=='__main__':main()
