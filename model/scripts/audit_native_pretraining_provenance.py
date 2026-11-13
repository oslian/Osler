"""Verify current pretraining source identities without inventing historical hashes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_pretraining_provenance_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    checkpoint=Path('models/whole_body_joint_jepa.pt')
    payload=torch.load(checkpoint,weights_only=True)
    source=Path(payload['training']['event_examples'])
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    raw=pd.read_parquet(source,columns=['subject_id','stay_id'])
    unique=raw.drop_duplicates()
    patient_path=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    patients=pd.read_csv(patient_path,usecols=['patientunitstayid','uniquepid','hospitalid'],dtype=str)
    matched=unique.assign(patientunitstayid=unique.stay_id.astype(str)).merge(patients,on='patientunitstayid',validate='many_to_one')
    if len(matched)!=len(unique) or not matched.subject_id.astype(str).eq(matched.uniquepid).all():
        raise ValueError('pretraining identity mapping unresolved')
    q=pd.read_parquet(root/'queries.parquet')
    valid=np.load(root/'labels.npz')['paired'][:,0];q=q.loc[valid]
    overlap=dict(subjects=len(set(matched.uniquepid)&set(q.uniquepid.astype(str))),
        stays=len(set(matched.patientunitstayid)&set(q.stay_id.str.removeprefix('eicu:'))),
        hospitals=len(set(matched.hospitalid)&set(q.hospitalid.astype(str))))
    report=dict(complete=True,status='current_file_identity_audit',runtime_modified=False,
        current_source_rows=len(raw),current_source_subjects=int(raw.subject_id.nunique()),
        current_source_stays=int(raw.stay_id.nunique()),
        checkpoint_training_rows=payload['training']['rows'],checkpoint_training_subjects=payload['training']['subjects'],
        overlap_with_current_source_file=overlap,historical_training_membership_certified=False,
        reason='checkpoint lacks immutable training-source hash and exact membership; current file counts differ',
        binding={str(p):sha256(p) for p in (checkpoint,source,patient_path,root/'queries.parquet',root/'labels.npz',Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='binding'})


if __name__=='__main__':main()
