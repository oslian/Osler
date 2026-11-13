"""Quantify a verified analyte mapping repair using fixed source-supported models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import duckdb
from whole_body_jepa.runtime import load_release
from whole_body_jepa.availability_inputs import build_arrays,restrict_input_support
from whole_body_jepa.multisource_intervals import sha256
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    torch.set_num_threads(4)
    assets=load_release(Path('models'))
    variables=assets.checkpoint['variables'];idx=variables.index('glucose')
    median=float(assets.scaler.medians[idx]);scale=float(assets.scaler.scales[idx])
    dictionary=pd.read_csv('/Users/chunyouchang/Downloads/mimic-iv-3.1/hosp/d_labitems.csv.gz')
    identity=dictionary.loc[dictionary.itemid.eq(50963)].iloc[0]
    if identity.label!='NTproBNP' or identity.fluid!='Blood':raise ValueError('unverified identity')
    frame=pd.read_parquet('/private/tmp/availability_glucose_mimic.parquet')
    valid=(frame.fresh & frame.truth.notna()).to_numpy()
    f=frame.loc[valid].reset_index(drop=True)
    active_cache=Path('/private/tmp/availability_jepa_mimic_labs.parquet')
    baseline_cache=Path('/private/tmp/availability_jepa_mimic_before_bnp_identity.parquet')
    if not baseline_cache.exists():
        pd.read_parquet(active_cache).to_parquet(baseline_cache,index=False)
    raw=pd.read_parquet(baseline_cache)
    # This historical cache was built with the single BNP item50963; no new
    # rows or units are invented. Preserve the original cache as audit evidence.
    corrected=raw.copy()
    corrected.loc[corrected.variable.eq('bnp'),'variable']='nt_pro_bnp'
    corrected.to_parquet('/private/tmp/availability_jepa_mimic_analyte_repaired.parquet',index=False)
    corrected.to_parquet(active_cache,index=False)
    c=duckdb.connect()
    starts=c.execute("SELECT stay_id,intime FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df();c.close()
    origins=dict(zip('mimic:'+starts.stay_id.astype(str),starts.intime))
    arrays={}
    for name,events in [('before',raw),('after',corrected)]:
        a,r=build_arrays(frame,events,assets,origins,truthful=True)
        arrays[name]=a
        print('built '+name,flush=True)
    rows=r[valid]
    report=dict(status='candidate_only',runtime_modified=False,causal_claim_allowed=False,
                corrected_events=int(raw.variable.eq('bnp').sum()),itemid=50963,
                dictionary_label=str(identity.label),seeds=[],
                binding={str(p):sha256(p) for p in (Path(__file__),Path('extractors/mimiciv_observation_transition_extract.py'),Path('BNP_IDENTITY_PROTOCOL_20260910.md'))})
    for seed in (7,19,31):
        result=dict(seed=seed,arms={})
        for arm in ('whole_unbounded','whole_unbounded_balanced'):
            p=Path(f'/private/tmp/encoder_supported_weights_20260910/{arm}_{seed}.pt')
            cp=torch.load(p,map_location='cpu',weights_only=False)
            if cp['variables']!=variables or cp['value_delta_activation']!='identity':raise ValueError('candidate schema')
            model=assets.model;model.value_delta_activation=torch.nn.Identity();model.load_state_dict(cp['state_dict'])
            pred={name:predict(model,restrict_input_support(a,cp['allowed_variables'],model.variable_module_ids.numpy()),rows,idx,median,scale)
                  for name,a in arrays.items()}
            result['arms'][arm]=dict(checkpoint_sha256=sha256(p),
                mae={name:patient_equal_mae(f,f.truth,y) for name,y in pred.items()},
                comparison=paired(f,pred['after'],pred['before'],f'bnp:{seed}:{arm}',1000),
                changed_predictions=int(np.sum(np.abs(pred['after']-pred['before'])>1e-5)))
        report['seeds'].append(result)
    report['complete']=True
    Path('artifacts/bnp_identity_repair_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
