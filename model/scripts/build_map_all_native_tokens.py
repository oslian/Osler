"""Native histories for MAP queries independent of potassium future labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_history_schedule import conflict_checked_history_schedule
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=root/'map_all_native_tokens.npz';out=Path('artifacts/map_all_native_tokens_20260912.json')
    if path.exists() or out.exists():raise ValueError('fresh output required')
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    q=pd.read_parquet(root/'queries.parquet')
    events,schedules=conflict_checked_history_schedule(pd.concat([original,bedside],ignore_index=True),q)
    print('schedules complete',len(schedules),flush=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    tokens=[];treatments=[];ids=[];qptr=[0];lookup={}
    for groups in schedules:
        for group in groups:
            key=tuple(group.tolist())
            if key not in lookup:
                token,treatment=encode_group(events.iloc[group],variables,assets.scaler.medians,
                    assets.scaler.scales,assets.model.variable_module_ids.numpy(),
                    len(assets.checkpoint['modules']),assets.max_age_hours)
                lookup[key]=len(tokens);tokens.append(token);treatments.append(treatment)
            ids.append(lookup[key])
        qptr.append(len(ids))
    matrix=np.asarray(tokens,np.float32);values=np.asarray(treatments,np.float64)
    ids=np.asarray(ids,np.int64);qptr=np.asarray(qptr,np.int64)
    if not np.isfinite(matrix).all():raise ValueError('nonfinite input')
    old=np.load(root/'native_tokens_bedside.npz');selected=np.flatnonzero(np.load(root/'labels.npz')['paired'][:,0])
    for j,i in enumerate(selected):
        new_ids=ids[qptr[i]:qptr[i+1]]
        old_ids=old['history_indices'][old['query_ptr'][j]:old['query_ptr'][j+1]]
        np.testing.assert_array_equal(matrix[new_ids],old['input'][old_ids])
        np.testing.assert_array_equal(values[new_ids],old['event_values'][old_ids])
    np.savez_compressed(path,input=matrix,event_values=values,query_ptr=qptr,history_indices=ids)
    report=dict(complete=True,training_performed=False,runtime_modified=False,queries=len(q),
        old_histories_exact=True,unique_groups=len(matrix),max_history=int(np.diff(qptr).max()),
        median_history=float(np.median(np.diff(qptr))),bindings={str(p):sha256(p) for p in (
            path,Path(__file__),root/'queries.parquet',root/'labels.npz',root/'native_events.parquet',
            root/'qualified_bedside_events.parquet',root/'native_tokens_bedside.npz',
            Path('whole_body_jepa/native_history_schedule.py'),Path('whole_body_jepa/native_event_tokens.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='bindings'})


if __name__=='__main__':main()
