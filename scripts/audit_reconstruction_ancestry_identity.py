"""Resolve provenance through original patient/stay table, no identifiers output."""
import json
from pathlib import Path
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_ancestry_identity_20260917.json')
    if out.exists():raise ValueError('existing evidence')
    raw=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    source=Path('/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet')
    qp=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/queries.parquet')
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    patients=pd.read_csv(raw,usecols=['patientunitstayid','uniquepid','hospitalid'],dtype=str)
    if patients.patientunitstayid.duplicated().any():raise ValueError('nonunique stay crosswalk')
    patients=patients.set_index('patientunitstayid')
    a=pd.read_parquet(source,columns=['subject_id','stay_id','hospitalid']).drop_duplicates()
    b=pd.read_parquet(qp,columns=['uniquepid','stay_id','hospitalid']).drop_duplicates()
    if not b.stay_id.str.startswith('eicu:').all():raise ValueError('unexpected query namespace')
    for frame,col,stays in ((a,'subject_id',a.stay_id.astype(str)),(b,'uniquepid',b.stay_id.str.removeprefix('eicu:'))):
        identity=stays.map(patients.uniquepid)
        hospital=stays.map(patients.hospitalid)
        if identity.isna().any() or hospital.isna().any():raise ValueError('unmapped stay')
        if not identity.eq(frame[col].astype(str)).all():raise ValueError('patient identity mismatch')
        if not pd.to_numeric(hospital).eq(pd.to_numeric(frame.hospitalid)).all():raise ValueError('hospital mismatch')
    roles=torch.load(sp,weights_only=False)['roles']
    if not set(roles).issubset(set(b.uniquepid)):raise ValueError('role patient missing')
    source_patients=set(a.subject_id.astype(str));source_sites=set(pd.to_numeric(a.hospitalid))
    role_results={}
    for role in sorted(set(roles.values())):
        selected={p for p,r in roles.items() if r==role}
        part=b.loc[b.uniquepid.isin(selected)]
        sites=set(pd.to_numeric(part.hospitalid))
        role_results[role]=dict(patients=len(selected),patient_overlap=len(selected&source_patients),
            sites=len(sites),site_overlap=len(sites&source_sites))
    paths=[Path(__file__),raw,source,qp,sp]
    result=dict(complete=True,raw_crosswalk_verified=True,
        source_stays=int(a.stay_id.nunique()),source_patients=len(source_patients),source_sites=len(source_sites),
        query_stays=int(b.stay_id.nunique()),query_patients=int(b.uniquepid.nunique()),
        patient_overlap=len(set(b.uniquepid)&source_patients),
        stay_overlap=len(set(a.stay_id.astype(str))&set(b.stay_id.str.removeprefix('eicu:'))),
        roles=role_results,bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Verified overlap with entire CURRENT ancestral source file, not exact historical optimizer exposure. Historical source counts/hash still unresolved. No blanket external-release or accuracy claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('bindings','limitation')})


if __name__=='__main__':main()
