"""Build expanded query-as-of histories; keep labels and old caches unchanged."""
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
    path=root/'native_tokens_bedside.npz'
    output=Path('artifacts/native_bedside_tokens_20260911.json')
    if path.exists() or output.exists():raise ValueError('fresh outputs required')
    original=pd.read_parquet(root/'native_events.parquet')
    original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet')
    bedside['event_time']=bedside.available_at
    events=pd.concat([original,bedside],ignore_index=True)
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    events,schedules=conflict_checked_history_schedule(events,q)
    print('canonical schedule complete',len(events),len(schedules),flush=True)
    assets=load_release(Path('models')); variables=assets.checkpoint['variables']
    tokens=[]; treatments=[]; ids=[]; qptr=[0]; lookup={}
    for groups in schedules:
        for group in groups:
            key=tuple(group.tolist())
            if key not in lookup:
                token,treatment=encode_group(events.iloc[group],variables,assets.scaler.medians,
                    assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules']),
                    assets.max_age_hours)
                lookup[key]=len(tokens);tokens.append(token);treatments.append(treatment)
            ids.append(lookup[key])
        qptr.append(len(ids))
    matrix=np.asarray(tokens,np.float32)
    if not np.isfinite(matrix).all():raise ValueError('nonfinite tokens')
    np.savez_compressed(path,input=matrix,event_values=np.asarray(treatments,np.float64),
        query_ptr=np.asarray(qptr,np.int64),history_indices=np.asarray(ids,np.int64))
    report=dict(complete=True,training_performed=False,runtime_modified=False,queries=len(q),
        unique_groups=len(matrix),history_length_max=int(np.diff(qptr).max()),
        history_length_median=float(np.median(np.diff(qptr))),
        supported_tokens={v:int((matrix[:,len(variables)+variables.index(v)]>0).sum()) for v in sorted(set(bedside.variable))},
        bindings={str(p):sha256(p) for p in [path,root/'native_events.parquet',root/'qualified_bedside_events.parquet',
            root/'queries.parquet',root/'labels.npz',Path(__file__),Path('whole_body_jepa/native_event_tokens.py'),
            Path('whole_body_jepa/native_history_schedule.py')]})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
