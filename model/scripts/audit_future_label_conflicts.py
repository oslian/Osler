"""Measure conservative conflict rejection on the unchanged query cache."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.raw_lab_replay import canonical_events


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/rate_multitarget_labels_20260911.json');ref=json.loads(parent.read_text())
    for path,digest in ref['binding'].items():assert sha256(Path(path))==digest,path
    out=Path('artifacts/future_label_conflicts_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    old=dict(np.load(root/'multitarget_labels.npz'));variables=list(old['variables'])
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    raw=canonical_events(pd.read_parquet(root/'body_labs.parquet'))
    new=future_pairs(q,raw,variables,horizon=6,tolerance=1,max_age=6)
    np.testing.assert_equal(new['current'],old['current'])
    np.testing.assert_equal(new['measurement_age'],old['measurement_age'])
    assert not (new['paired'] & ~old['paired']).any()
    final=raw.drop_duplicates(['stay_id','source_id'],keep='last')
    counts=final.groupby(['stay_id','variable','measured_at']).value.nunique()
    conflicts=counts[counts>1]
    table={}
    for j,v in enumerate(variables):
        good=new['paired'][:,j]
        for name in ('future','actual_delta_t','future_source_id'):
            np.testing.assert_equal(new[name][good,j],old[name][good,j])
        removed=old['paired'][:,j]&~good
        if removed.any():
            table[v]=dict(old_pairs=int(old['paired'][:,j].sum()),removed=int(removed.sum()),
                          patients=int(q.loc[removed,'uniquepid'].nunique()))
    dest=root/'multitarget_labels_conflict_checked.npz'
    if dest.exists():raise ValueError('cache overwrite blocked')
    np.savez_compressed(dest,**new,variables=np.asarray(variables))
    report=dict(complete=True,runtime_modified=False,training_performed=False,total_conflicting_draw_groups=len(conflicts),
                removed_pairs=table,unchanged_retained_truth_verified=True,current_contract_unchanged=True,
                binding={str(p):sha256(p) for p in [parent,Path(__file__),Path('whole_body_jepa/availability_supervision.py'),dest]})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(table,indent=2))


if __name__=='__main__':main()
