"""Rebuild conflict-safe snapshots on the exact old query/history grid."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/rate_anchor_body_inputs_20260911.json');ref=json.loads(parent.read_text())
    for path in (root/'body_labs.parquet',root/'body_arrays.npz'):
        assert sha256(path)==ref['binding'][str(path)]
    out=Path('artifacts/snapshot_conflicts_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    assets=load_release(Path('models'));raw=pd.read_parquet(root/'body_labs.parquet')
    new,rows=build_arrays(q,raw,assets,{s:pd.Timestamp('2100-01-01') for s in q.stay_id.unique()})
    old=dict(np.load(root/'body_arrays.npz'))
    np.testing.assert_array_equal(rows,old['query_rows'])
    np.testing.assert_array_equal(new['history'],old['history'])
    retained=(new['mask']>0)&(old['mask']>0)
    np.testing.assert_array_equal(new['current'][retained],old['current'][retained])
    assert not ((new['mask']>0)&(old['mask']==0)).any()
    dropped=(old['mask']>0)&(new['mask']==0)
    table={}
    for j,v in enumerate(assets.checkpoint['variables']):
        if dropped[:,j].any():
            touched=dropped[old['history'][rows],j].any(axis=1)
            table[v]=dict(history_snapshot_cells=int(dropped[:,j].sum()),query_cells=int(dropped[rows,j].sum()),
                          affected_histories=int(touched.sum()),patients=int(q.loc[touched,'uniquepid'].nunique()))
    dest=root/'body_arrays_conflict_checked.npz'
    if dest.exists():raise ValueError('cache overwrite blocked')
    np.savez_compressed(dest,**new,query_rows=rows)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        retained_values_exact=True,only_observation_removal=True,affected=table,
        binding={str(p):sha256(p) for p in [parent,Path(__file__),Path('whole_body_jepa/availability_inputs.py'),dest]}),indent=2)+'\n')
    print(json.dumps(table,indent=2))


if __name__=='__main__':main()
