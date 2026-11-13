"""Remove unrelated future-pair filtering from MAP label construction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/map_label_selection_20260912.json')
    cache=root/'map_all_query_labels.npz'
    if out.exists() or cache.exists():raise ValueError('fresh outputs required')
    q=pd.read_parquet(root/'queries.parquet')
    events=pd.read_parquet(root/'qualified_bedside_events.parquet')
    events=events.loc[events.variable.eq('map')]
    old_filter=np.load(root/'labels.npz')['paired'][:,0]
    labels=future_pairs(q,events,['map'],horizon=6.,tolerance=1.,max_age=6.)
    old=np.load(root/'bedside_future_labels.npz');j=old['variables'].tolist().index('map')
    for field in ('current','measurement_age','future','actual_delta_t','paired','future_source_id'):
        np.testing.assert_array_equal(labels[field][old_filter,0],old[field][:,j])
    paired=labels['paired'][:,0]
    def summary(mask):
        return dict(rows=int(mask.sum()),patients=int(q.loc[mask,'uniquepid'].nunique()),
                    stays=int(q.loc[mask,'stay_id'].nunique()))
    roles=[]
    for seed in (7,19,31,43,59,71,83):
        p=dict(np.load(root/f'partition_{seed}.npz'))
        roles.append(dict(seed=seed,roles={k:dict(old=summary(v&paired&old_filter),
            recovered=summary(v&paired&~old_filter),all=summary(v&paired)) for k,v in p.items()}))
    np.savez_compressed(cache,**labels,variables=np.asarray(['map']))
    report=dict(complete=True,training_performed=False,runtime_modified=False,
        queries=len(q),old=summary(paired&old_filter),recovered=summary(paired&~old_filter),
        all=summary(paired),old_overlap_exact=True,seeds=roles,
        limitation='Original query and bedside cohort remain restricted; not all eICU or MAP-native anchors',
        bindings={str(p):sha256(p) for p in [Path(__file__),cache,root/'queries.parquet',root/'labels.npz',
            root/'bedside_future_labels.npz',root/'qualified_bedside_events.parquet',
            Path('whole_body_jepa/availability_supervision.py')]+[root/f'partition_{s}.npz' for s in (7,19,31,43,59,71,83)]})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print({k:report[k] for k in ('queries','old','recovered','all','old_overlap_exact')})
    print(roles[0])


if __name__=='__main__':main()
