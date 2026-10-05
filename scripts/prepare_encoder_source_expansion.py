"""Build a new patient-disjoint source pool without overwriting old inputs."""
import hashlib
import json
from pathlib import Path
import pandas as pd

from scripts.audit_availability_jepa import extract
from whole_body_jepa.availability_anchors import rebuild
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def select_patients(metadata, development, heldout, limit=4000):
    dev=metadata.loc[metadata.stay_id.isin(development.stay_id)]
    test=metadata.loc[metadata.stay_id.isin(heldout.stay_id)]
    if len(dev)!=development.stay_id.nunique() or len(test)!=heldout.stay_id.nunique():
        raise ValueError('missing role metadata')
    source_hospitals=set(dev.hospitalid);heldout_hospitals=set(test.hospitalid)
    if source_hospitals & heldout_hospitals:raise ValueError('overlapping hospitals')
    blocked=set(dev.uniquepid)|set(test.uniquepid)|set(metadata.loc[metadata.hospitalid.isin(heldout_hospitals),'uniquepid'])
    eligible=metadata.loc[metadata.hospitalid.isin(source_hospitals)&~metadata.uniquepid.isin(blocked)].copy()
    patients=sorted(set(eligible.uniquepid),key=lambda s:hashlib.sha256(('source-expansion-v1:'+s).encode()).hexdigest())
    chosen=eligible.loc[eligible.uniquepid.isin(patients[:limit])].copy()
    if set(chosen.uniquepid)&blocked:raise ValueError('patient overlap')
    return chosen,dict(source_hospitals=len(source_hospitals),heldout_hospitals=len(heldout_hospitals),
        eligible_patients=len(patients),selected_patients=int(chosen.uniquepid.nunique()),selected_stays=len(chosen),
        patient_overlap=0,hospital_overlap=0)


def main():
    output=Path('artifacts/encoder_source_expansion_20260910.json')
    directory=Path('/private/tmp/encoder_source_expansion_20260910')
    if output.exists() or directory.exists():raise ValueError('use fresh output')
    metadata_path=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    metadata=pd.read_csv(metadata_path,usecols=['patientunitstayid','uniquepid','hospitalid'],dtype={'uniquepid':str})
    if metadata[['uniquepid','hospitalid']].isna().any().any():raise ValueError('missing identity')
    metadata['stay_id']='eicu:'+metadata.patientunitstayid.astype(str)
    rolepaths=[Path('/private/tmp/availability_glucose_'+r+'.parquet') for r in ['development','eicu_hospital']]
    dev,test=[pd.read_parquet(p)[['stay_id','subject_id']].drop_duplicates() for p in rolepaths]
    chosen,support=select_patients(metadata,dev,test)
    directory.mkdir()
    chosen['subject_id']='eicu:'+chosen.uniquepid
    chosen[['stay_id','subject_id']].to_parquet(directory/'subjects.parquet',index=False)
    assets=load_release(Path('models'))
    print(json.dumps(support),flush=True)
    raw,_,audit=extract('eicu',set(chosen.stay_id),assets.checkpoint['variables'],output=directory/'labs.parquet')
    queries=rebuild(raw.loc[raw.variable.eq('glucose')])
    queries=queries.merge(chosen[['stay_id','subject_id']],on='stay_id',validate='many_to_one')
    queries.to_parquet(directory/'queries.parquet',index=False)
    valid=queries.fresh & queries.truth.notna()
    support.update(queries=len(queries),scored_rows=int(valid.sum()),scored_patients=int(queries.loc[valid,'subject_id'].nunique()))
    report=dict(status='source_data_prepared',accuracy_tested=False,runtime_modified=False,
        support=support,extraction=audit,binding={str(p):sha256(p) for p in [Path(__file__),
        Path('ENCODER_SOURCE_EXPANSION_PROTOCOL_20260910.md'),metadata_path,*rolepaths,
        directory/'labs.parquet',directory/'queries.parquet',directory/'subjects.parquet']})
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(support),flush=True)


if __name__=='__main__':main()
