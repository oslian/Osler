"""Count actual as-of conflicting event groups in summary windows."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from whole_body_jepa.raw_lab_replay import canonical_events
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/event_summary_conflicts_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    raw=canonical_events(pd.read_parquet(root/'body_labs.parquet'))
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    groups=dict(tuple(raw.groupby('stay_id',sort=False)));counts={h:Counter() for h in (6,24)};anyrows={h:set() for h in counts}
    for stay,queries in q.groupby('stay_id',sort=False):
        events=groups.get(stay)
        if events is None:continue
        for row,t in zip(queries.index,queries.t):
            known=events.loc[events.available_at<=t].drop_duplicates('source_id',keep='last')
            known=known.loc[known.measured_at>=t-pd.Timedelta(hours=24)]
            c=known.groupby(['variable','measured_at']).value.nunique()
            for h in counts:
                conflict=c[(c>1)&(c.index.get_level_values('measured_at')>=t-pd.Timedelta(hours=h))]
                for variable in set(conflict.index.get_level_values('variable')):counts[h][variable]+=1
                if len(conflict):anyrows[h].add(row)
    report=dict(complete=True,runtime_modified=False,training_performed=False,query_count=len(q),
        windows={str(h):dict(affected_queries=len(anyrows[h]),affected_query_fraction=len(anyrows[h])/len(q),
            variables=dict(counts[h])) for h in counts},
        binding={str(p):sha256(p) for p in [Path(__file__),root/'body_labs.parquet',root/'queries.parquet',root/'labels.npz',Path('whole_body_jepa/event_window_history.py')]})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['windows'],indent=2))


if __name__=='__main__':main()
