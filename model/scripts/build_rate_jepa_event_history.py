"""Separate recorded insulin context aligned to every JEPA history snapshot."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.potassium_treatment_context import context_features
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/rate_jepa_event_history_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    body=json.loads(Path('artifacts/rate_anchor_body_inputs_20260911.json').read_text())
    if sha256(root/'body_arrays.npz')!=body['binding'][str(root/'body_arrays.npz')]:raise ValueError('body changed')
    labels=np.load(root/'labels.npz');q=pd.read_parquet(root/'queries.parquet').loc[labels['paired'][:,0]].reset_index(drop=True)
    arrays=np.load(root/'body_arrays.npz');rows=arrays['query_rows'];history=arrays['history'][rows]
    mapping={}
    for i,query in enumerate(q.itertuples()):
        for slot,lag in zip(history[i],HISTORY_LAGS_HOURS):
            key=(query.stay_id,pd.Timestamp(query.t)-pd.Timedelta(hours=lag))
            if int(slot) in mapping and mapping[int(slot)]!=key:raise ValueError('history row identity collision')
            mapping[int(slot)]=key
    if set(mapping)!=set(range(len(arrays['input']))):raise ValueError('unmapped history snapshots')
    frame=pd.DataFrame([mapping[i] for i in range(len(mapping))],columns=['stay_id','t'])
    frame['offset']=(frame.t-pd.Timestamp('2100-01-01')).dt.total_seconds()/60
    raw=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz')
    events=read_insulin_rates(raw);events['stay_id']='eicu:'+events.stay_id.astype(str)
    events=events.loc[events.stay_id.isin(frame.stay_id)].rename(columns={'label':'original_label'})
    events['action']='insulin';events['dimension']='drug_units_per_hour'
    values,columns=context_features(frame,events)
    keep=[i for i,c in enumerate(columns) if c in [('insulin','count'),('insulin','age'),('insulin','drug_units_per_hour')]]
    values=values[:,keep];columns=[columns[i] for i in keep]
    rate=values[:,columns.index(('insulin','drug_units_per_hour'))]
    path=root/'event_history.npz';np.savez_compressed(path,values=values,observed=np.isfinite(values),query_rows=rows)
    report=dict(complete=True,runtime_modified=False,training_performed=False,
                independent_availability_verified=False,status='record_time_history_candidate',
                history_snapshots=len(frame),query_count=len(q),columns=columns,
                supported_rate_snapshots=int(np.isfinite(rate).sum()),
                zero_rate_snapshots=int((rate==0).sum()),
                query_rate_support=int(np.isfinite(rate[rows]).sum()),
                binding={str(p):sha256(p) for p in [path,root/'body_arrays.npz',root/'queries.parquet',root/'labels.npz',Path(__file__)]})
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
