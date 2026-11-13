"""Inventory matched potassium labels; no outcome-driven patient selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from scripts.audit_availability_jepa import extract
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.potassium_treatment_context import rate_history_features
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/insulin_potassium_pairs_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    root=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    private=Path('/private/tmp/insulin_potassium_pairs_20260911');private.mkdir(exist_ok=True)
    rates=read_insulin_rates(root/'infusionDrug.csv.gz')
    rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    wanted=set(rates.stay_id)
    result=extract('eicu',wanted,['potassium'],output=private/'labs.parquet')
    raw=pd.read_parquet(private/'labs.parquet')
    q=raw[['stay_id','available_at']].rename(columns={'available_at':'t'}).drop_duplicates().reset_index(drop=True)
    labels=future_pairs(q,raw,['potassium'],horizon=6,tolerance=1,max_age=6)
    q['offset']=(q.t-pd.Timestamp('2100-01-01')).dt.total_seconds()/60
    rates=rates.rename(columns={'label':'original_label'})
    rates['action']='insulin';rates['dimension']='drug_units_per_hour'
    features,cols=rate_history_features(q,rates)
    sequence=np.isfinite(features[:,cols.index(('insulin','drug_units_per_hour','change'))])
    paired=labels['paired'][:,0]
    meta=pd.read_csv(root/'patient.csv.gz',usecols=['patientunitstayid','uniquepid','hospitalid'])
    meta['stay_id']='eicu:'+meta.patientunitstayid.astype(str)
    q=q.merge(meta[['stay_id','uniquepid','hospitalid']],on='stay_id',how='left',validate='many_to_one')
    def support(mask):
        sub=q.loc[mask]
        return dict(rows=len(sub),stays=int(sub.stay_id.nunique()),patients=int(sub.uniquepid.nunique()),hospitals=int(sub.hospitalid.nunique()))
    report=dict(complete=True,status='pair_support_not_forecast_validation',all_queries=support(np.ones(len(q),bool)),
        paired=support(paired),sequence=support(sequence),paired_sequence=support(paired&sequence),
        runtime_modified=False,pristine_lockbox=False,treatment_availability_verified=False,
        binding={str(p):sha256(p) for p in [Path(__file__),root/'infusionDrug.csv.gz',root/'lab.csv.gz',private/'labs.parquet']})
    np.savez_compressed(private/'labels.npz',**labels,sequence=sequence)
    q.to_parquet(private/'queries.parquet',index=False)
    report['private_cache_hashes']={str(p):sha256(p) for p in [private/'labels.npz',private/'queries.parquet']}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
