"""Measure joint input/label conflict rejection on the unchanged query cache."""
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
    out=Path('artifacts/joint_conflict_contract_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    old=dict(np.load(root/'multitarget_labels.npz'));variables=list(old['variables'])
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    raw=canonical_events(pd.read_parquet(root/'body_labs.parquet'))
    new=future_pairs(q,raw,variables,horizon=6,tolerance=1,max_age=6)
    retained_current=np.isfinite(new['current'])
    np.testing.assert_equal(new['current'][retained_current],old['current'][retained_current])
    arrays=dict(np.load(root/'body_arrays_conflict_checked.npz'))
    from whole_body_jepa.runtime import load_release
    assets=load_release(Path('models'));rows=arrays['query_rows']
    np.testing.assert_array_equal(retained_current,arrays['mask'][rows]>0)
    reconstructed=arrays['current'][rows]*assets.scaler.scales+assets.scaler.medians
    np.testing.assert_allclose(new['current'][retained_current],reconstructed[retained_current],atol=1e-4,rtol=1e-6)
    future_only=dict(np.load(root/'multitarget_labels_conflict_checked.npz'))
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
                          patients=int(q.loc[removed,'uniquepid'].nunique()),additional_current_rejections=int((future_only['paired'][:,j]&~good).sum()))
    dest=root/'multitarget_labels_joint_conflict_checked.npz'
    if dest.exists():raise ValueError('cache overwrite blocked')
    np.savez_compressed(dest,**new,variables=np.asarray(variables))
    report=dict(complete=True,runtime_modified=False,training_performed=False,total_conflicting_draw_groups=len(conflicts),
                removed_pairs=table,unchanged_retained_truth_verified=True,input_label_current_agreement_verified=True,
                binding={str(p):sha256(p) for p in [parent,Path(__file__),Path('whole_body_jepa/availability_supervision.py'),Path('whole_body_jepa/availability_inputs.py'),root/'body_arrays_conflict_checked.npz',dest]})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(table,indent=2))


if __name__=='__main__':main()

