"""Separate simultaneous rejection from later-arriving conflicting evidence."""
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from whole_body_jepa.raw_lab_replay import canonical_events
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=root/'body_labs.parquet';out=Path('artifacts/native_conflict_arrivals_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    raw=canonical_events(pd.read_parquet(path))
    counts=raw.groupby(['stay_id','variable','measured_at']).value.nunique()
    keys=counts[counts>1].index
    chosen=raw.set_index(['stay_id','variable','measured_at']).loc[keys].reset_index()
    stats=Counter();variables=Counter()
    for key,g in chosen.groupby(['stay_id','variable','measured_at']):
        known={};any_conflict=False;later_conflict=False;age_eligible=False
        for time,arrival in g.groupby('available_at',sort=True):
            previous=len(known)>0
            for r in arrival.itertuples():known[str(r.source_id)]=float(r.value)
            if len(set(known.values()))>1:
                any_conflict=True
                if previous and arrival.value.nunique()==1:
                    later_conflict=True
                    if 0<=(time-key[2]).total_seconds()/3600<=6:age_eligible=True
        if any_conflict:stats['asof_conflicting_groups']+=1
        if later_conflict:stats['single_value_later_arrival_groups']+=1
        if age_eligible:
            stats['later_arrival_within_6h_groups']+=1;variables[key[1]]+=1
    report=dict(complete=True,runtime_modified=False,training_performed=False,
        scope='raw-cache candidate ingestion risk; not exact encoded schedule exposure',counts=dict(stats),
        variables=dict(variables),binding={str(p):sha256(p) for p in [path,Path(__file__),Path('whole_body_jepa/native_event_tokens.py'),Path('whole_body_jepa/raw_lab_replay.py')]})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'counts':dict(stats),'variables':dict(variables)},indent=2))


if __name__=='__main__':main()
