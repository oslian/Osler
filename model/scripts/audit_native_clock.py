"""Quantify lost elapsed time, without refitting or selecting patients."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_event_tokens import encode_group


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_clock_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_rate_jepa_tokens_20260911.json').read_text())
    for name in ('native_events.parquet','native_schedule.npz','native_tokens.npz'):
        if sha256(root/name)!=ref['binding'][str(root/name)]:raise ValueError('cache changed')
    events=pd.read_parquet(root/'native_events.parquet')
    with np.load(root/'native_schedule.npz') as f:s={k:f[k] for k in f.files}
    with np.load(root/'native_tokens.npz') as f:t={k:f[k] for k in f.files}
    first=s['event_indices'][s['group_ptr'][:-1]]
    hours=(pd.to_datetime(events.event_time.iloc[first])-pd.Timestamp('2100-01-01')).dt.total_seconds().to_numpy()/3600
    ids=t['history_indices'];x=t['input'];v=t['event_values']
    active=x[ids,53:106].any(axis=1)|np.isfinite(v[ids,2])
    physical=[];encoded=[];affected=[]
    for a,b in zip(s['query_ptr'][:-1],s['query_ptr'][1:]):
        m=active[a:b];h=hours[a:b][m];clock=x[ids[a:b],-1][m]
        d=np.diff(h);e=np.diff(clock)*168
        if (d<=0).any():raise ValueError('nonincreasing native groups')
        physical.extend(d);encoded.extend(e);affected.append(bool(((d>0)&(e==0)).any()))
    physical=np.asarray(physical);encoded=np.asarray(encoded);lost=(physical>0)&(encoded==0)
    # Identical observations at two late timestamps produce identical tokens.
    g=pd.DataFrame(dict(stay_id=['a'],source_kind=['insulin_record'],
                        event_time=[pd.Timestamp('2100-01-09')],measured_at=[pd.NaT],
                        variable=['insulin_rate'],value=[1.]))
    a,_=encode_group(g,['potassium'],[4.],[1.],[0],1,6.)
    g.event_time+=pd.Timedelta(hours=12)
    b,_=encode_group(g,['potassium'],[4.],[1.],[0],1,6.)
    np.testing.assert_array_equal(a,b)
    report=dict(complete=True,runtime_modified=False,training_performed=False,
                active_transitions=len(physical),zero_encoded_positive_time=int(lost.sum()),
                affected_queries=int(sum(affected)),queries=len(affected),
                lost_gap_median_hours=float(np.median(physical[lost])) if lost.any() else None,
                synthetic_12h_late_clock_collision=True,
                binding={str(p):sha256(p) for p in (Path(__file__),root/'native_tokens.npz',root/'native_schedule.npz',root/'native_events.parquet')})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
