"""Expand the same query inputs without changing any laboratory label."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=root/'body_arrays_bedside.npz'
    output=Path('artifacts/native_bedside_inputs_20260911.json')
    if path.exists() or output.exists():raise ValueError('fresh outputs required')
    labs=pd.read_parquet(root/'body_labs.parquet');labs['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet')
    raw=pd.concat([labs,bedside],ignore_index=True)
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]].reset_index(drop=True)
    assets=load_release(Path('models'))
    arrays,rows=build_arrays(q,raw,assets,{s:pd.Timestamp('2100-01-01') for s in q.stay_id.unique()})
    old=dict(np.load(root/'body_arrays_conflict_checked.npz'))
    variables=assets.checkpoint['variables']
    added=set(bedside.variable)
    unchanged=[i for i,v in enumerate(variables) if v not in added]
    for field in ('current','mask','ages'):
        np.testing.assert_array_equal(arrays[field][rows][:,unchanged],old[field][old['query_rows']][:,unchanged])
    np.savez_compressed(path,**arrays,query_rows=rows)
    summary={v:int((arrays['mask'][rows,variables.index(v)]>0).sum()) for v in sorted(added)}
    report=dict(complete=True,training_performed=False,runtime_modified=False,queries=len(q),
        unchanged_lab_inputs_exact=True,support=summary,
        bindings={str(p):sha256(p) for p in [path,root/'body_labs.parquet',root/'qualified_bedside_events.parquet',
            root/'body_arrays_conflict_checked.npz',root/'queries.parquet',root/'labels.npz',Path(__file__),
            Path('whole_body_jepa/availability_inputs.py')]})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
