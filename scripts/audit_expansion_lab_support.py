"""Source-only aggregate laboratory eligibility preflight; no model fitting."""
import json
from pathlib import Path
import duckdb
import pandas as pd
import torch
from extractors.eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from extractors.mimiciv_observation_transition_extract import EXTRA_PLAUSIBLE
from whole_body_jepa.availability_inputs import UNITS
from whole_body_jepa.multisource_intervals import sha256


def main():
    out = Path('artifacts/native_reconstruction_expansion_labs_20260917.json')
    if out.exists():
        raise ValueError('Existing preflight')
    prior = Path('artifacts/native_reconstruction_expansion_inventory_20260917.json')
    inventory = json.loads(prior.read_text())
    for path, digest in inventory['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed source')
    root = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    patients = pd.read_csv(root/'patient.csv.gz', usecols=['patientunitstayid','uniquepid','hospitalid'], dtype=str)
    split = torch.load('data/research_recovery/reconstruction_nested_selection_20260917.pt', weights_only=False)
    q = pd.read_parquet('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet', columns=['uniquepid','hospitalid'])
    hospital_roles = {str(int(r.hospitalid)):split['roles'][r.uniquepid] for r in q.itertuples()}
    source = pd.read_parquet('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet', columns=['subject_id','hospitalid'])
    excluded = set(split['roles']) | set(source.subject_id.astype(str))
    excluded_sites = set(source.hospitalid.astype(int).astype(str))
    roles = {}
    for patient, sites in patients.groupby('uniquepid').hospitalid.agg(set).items():
        mapped = {hospital_roles.get(s,'unassigned') for s in sites}
        if patient not in excluded and not sites & excluded_sites and len(mapped)==1:
            role=next(iter(mapped))
            if role in ('fit','encoder_tune','readout_tune'):
                roles[patient]=role
    wanted = patients.loc[patients.uniquepid.isin(roles)].copy()
    wanted['role']=wanted.uniquepid.map(roles)
    for role in ('fit','encoder_tune','readout_tune'):
        if wanted.loc[wanted.role==role,'uniquepid'].nunique()!=inventory['role_candidates'][role]['new_patients_with_full_history_inside_role']:
            raise ValueError('Inventory mismatch')
    catalog_path=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    catalog=list(json.loads(catalog_path.read_text())['target_counts'])
    bounds={**PLAUSIBLE,**EXTRA_PLAUSIBLE}
    mapping=pd.DataFrame([(name,var,unit,*bounds[var]) for name,var in LAB_TO_STATE.items()
        if var in catalog and var in UNITS and var in bounds for unit in UNITS[var]],
        columns=['name','variable','unit','low','high'])
    db=duckdb.connect();db.execute('SET threads=4')
    db.register('wanted',wanted);db.register('mapping',mapping)
    print('Scanning source-only laboratory support',flush=True)
    counts=db.execute('''SELECT w.role,m.variable,w.hospitalid,
        count(distinct w.uniquepid) patients,count(distinct l.labid) events
        FROM read_csv_auto(?) l JOIN wanted w ON cast(l.patientunitstayid as varchar)=w.patientunitstayid
        JOIN mapping m ON l.labname=m.name
        AND lower(trim(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,'')))=m.unit
        WHERE isfinite(try_cast(l.labresult as double))
        AND try_cast(l.labresult as double) BETWEEN m.low AND m.high
        AND l.labresultoffset IS NOT NULL AND l.labresultrevisedoffset IS NOT NULL
        AND greatest(l.labresultoffset,l.labresultrevisedoffset)-l.labresultoffset BETWEEN 0 AND 60
        GROUP BY w.role,m.variable,w.hospitalid''',[str(root/'lab.csv.gz')]).df()
    db.close()
    result={}
    for target in catalog:
        result[target]={}
        for role in ('fit','encoder_tune','readout_tune'):
            rows=counts.loc[(counts.variable==target)&(counts.role==role)]
            result[target][role]=dict(site_patient_counts=sorted(rows.patients.astype(int).tolist()),
                sites_at_least_30=int((rows.patients>=30).sum()),events=int(rows.events.sum()),
                mapped_lab=target in set(mapping.variable))
    paths=(Path(__file__),prior,catalog_path,root/'lab.csv.gz',Path('whole_body_jepa/availability_inputs.py'),
           Path('extractors/eicu_sepsis_transition_extract.py'),Path('extractors/mimiciv_observation_transition_extract.py'))
    payload=dict(complete=True,targets=result,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Patient counts are per site, not deduplicated across sites. Screening only: source IDs/conflicting events, history, block masks and complete reconstruction acceptance remain unverified. Bedside-only targets explicitly unmapped. No evaluation or reserved lab outcomes loaded.')
    out.write_text(json.dumps(payload,indent=2)+'\n')
    print('Readout targets with >=3 sites of >=30 screened patients:',sum(r['readout_tune']['sites_at_least_30']>=3 for r in result.values()))
    for t in ('bun','bicarbonate','phosphate'):print(t,result[t]['readout_tune'])


if __name__=='__main__':main()
