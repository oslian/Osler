"""Quantify target-native label support without selecting a new model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.raw_lab_replay import canonical_events


def summary(q, labels, column, eligible):
    return dict(rows=int(eligible.sum()), patients=int(q.loc[eligible, 'uniquepid'].nunique()),
                stays=int(q.loc[eligible, 'stay_id'].nunique()),
                age_quantiles=np.quantile(labels['measurement_age'][eligible, column], [0, .25, .5, .75, 1]).tolist(),
                delta_quantiles=np.quantile(labels['actual_delta_t'][eligible, column], [0, .25, .5, .75, 1]).tolist())


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/magnesium_anchor_support_20260911.json')
    if out.exists():
        raise ValueError('fresh output required')
    parent=Path('artifacts/rate_multitarget_labels_20260911.json')
    ref=json.loads(parent.read_text())
    for path, digest in ref['binding'].items():
        assert sha256(Path(path)) == digest, path
    queries=pd.read_parquet(root/'queries.parquet')
    mapping=queries[['stay_id', 'uniquepid', 'hospitalid']].drop_duplicates()
    assert not mapping.stay_id.duplicated().any()
    raw=canonical_events(pd.read_parquet(root/'body_labs.parquet'))
    mg=raw.loc[raw.variable.eq('magnesium') & raw.stay_id.isin(mapping.stay_id)]
    q=mg[['stay_id', 'available_at']].rename(columns={'available_at':'t'}).drop_duplicates()
    q=q.merge(mapping, on='stay_id', validate='many_to_one').sort_values(['stay_id', 't']).reset_index(drop=True)
    labels=future_pairs(q, raw.loc[raw.variable.eq('magnesium')], ['magnesium'], horizon=6, tolerance=1, max_age=6)
    paired=labels['paired'][:, 0]
    old=dict(np.load(root/'multitarget_labels.npz'))
    variables=list(old['variables']);col=variables.index('magnesium')
    oldq=queries.loc[np.load(root/'labels.npz')['paired'][:, 0]].reset_index(drop=True)
    old_ok=old['paired'][:, col] & old['paired'][:, variables.index('glucose')]
    old_ids=set(zip(oldq.loc[old_ok, 'stay_id'].astype(str), old['future_source_id'][old_ok, col]))
    new_ids=set(zip(q.loc[paired, 'stay_id'].astype(str), labels['future_source_id'][paired, 0]))
    role_counts=[]
    for seed in (7,19,31,43,59,71,83):
        roles=dict(np.load(root/f'partition_{seed}.npz'))
        counts={}
        all_patients=[]
        for role, mask in roles.items():
            patients=set(queries.loc[mask, 'uniquepid'])
            all_patients.append(patients)
            selected=paired & q.uniquepid.isin(patients).to_numpy()
            counts[role]=dict(rows=int(selected.sum()),patients=int(q.loc[selected,'uniquepid'].nunique()))
        for i,a in enumerate(all_patients):
            assert all(not a.intersection(b) for b in all_patients[i+1:])
        role_counts.append(dict(seed=seed,roles=counts))
    private=root/'magnesium_native_anchor_support'
    private.mkdir(exist_ok=True)
    q.to_parquet(private/'queries.parquet',index=False)
    np.savez_compressed(private/'labels.npz',**labels)
    report=dict(complete=True,runtime_modified=False,training_performed=False,promotion_allowed=False,
                scope='same available stay universe, not full eICU; changed prediction task',
                old=summary(oldq,old,col,old_ok),native=summary(q,labels,0,paired),
                old_unique_future_events=len(old_ids),native_unique_future_events=len(new_ids),
                common_future_events=len(old_ids & new_ids),roles=role_counts,
                binding={str(p):sha256(p) for p in [parent,Path(__file__),Path('whole_body_jepa/availability_supervision.py'),
                    private/'queries.parquet',private/'labels.npz']})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('binding','roles')},indent=2))
    print('Role patient counts', [r['roles'] for r in role_counts])


if __name__=='__main__':
    main()
