"""Inventory native event groups on the unchanged rate-query cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.native_history_schedule import native_history_schedule
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/native_rate_jepa_schedule_20260911_v2.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    labs=pd.read_parquet(root/'body_labs.parquet')
    labs=labs.rename(columns={'available_at':'event_time'});labs['source_kind']='lab_available'
    rates=read_insulin_rates(Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz'))
    rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    rates=rates.loc[rates.stay_id.isin(q.stay_id)].copy()
    rates['event_time']=pd.Timestamp('2100-01-01')+pd.to_timedelta(rates.offset,unit='m')
    rates['source_kind']='insulin_record';rates['variable']='insulin_rate'
    rates['measured_at']=pd.NaT
    rates['unit']='units/hour'
    columns=['stay_id','source_kind','source_id','event_time','measured_at','value','variable','unit']
    combined=pd.concat([labs[columns],rates[columns]],ignore_index=True)
    combined['source_id']=combined.source_id.astype(str)
    events,schedules=native_history_schedule(combined,q)
    events_path=root/'native_events.parquet';events.to_parquet(events_path,index=False)
    query_ptr=[0];group_ptr=[0];indices=[]
    for schedule in schedules:
        for group in schedule:
            indices.extend(group.tolist());group_ptr.append(len(indices))
        query_ptr.append(len(group_ptr)-1)
    index_path=root/'native_schedule.npz'
    np.savez_compressed(index_path,query_ptr=np.asarray(query_ptr,np.int64),group_ptr=np.asarray(group_ptr,np.int64),event_indices=np.asarray(indices,np.int64))
    with np.load(index_path) as archive:
        saved={name:archive[name] for name in archive.files}
    for i,schedule in enumerate(schedules):
        start,stop=saved['query_ptr'][i:i+2]
        if stop-start!=len(schedule):raise ValueError('query grouping serialization mismatch')
        for j,group in enumerate(schedule,start):
            a,b=saved['group_ptr'][j:j+2]
            np.testing.assert_array_equal(group,saved['event_indices'][a:b])
    lengths=np.array([len(s) for s in schedules]);counts=np.array([sum(len(g) for g in s) for s in schedules])
    report=dict(complete=True,training_performed=False,runtime_modified=False,
                queries=len(q),canonical_events=len(events),
                group_length_quantiles=np.quantile(lengths,[0,.5,.9,.99,1]).tolist(),
                event_count_quantiles=np.quantile(counts,[0,.5,.9,.99,1]).tolist(),
                queries_over_ten_groups=int((lengths>10).sum()),
                empty_queries=int((lengths==0).sum()),
                timing='lab availability plus retrospective insulin record time; not verified treatment availability',
                serialization_roundtrip=True,
                binding={str(p):sha256(p) for p in [events_path,index_path,root/'queries.parquet',root/'labels.npz',root/'body_labs.parquet',Path(__file__)]})
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
