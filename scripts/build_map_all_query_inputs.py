"""Availability-safe inputs without an unrelated potassium future filter."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=root/'map_all_query_inputs.npz';out=Path('artifacts/map_all_query_inputs_20260912.json')
    parent=Path('artifacts/map_label_selection_20260912.json')
    if path.exists() or out.exists():raise ValueError('fresh output required')
    for p,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('label dependency changed')
    q=pd.read_parquet(root/'queries.parquet')
    labs=pd.read_parquet(root/'body_labs.parquet');labs['quality_valid']=True
    raw=pd.concat([labs,pd.read_parquet(root/'qualified_bedside_events.parquet')],ignore_index=True)
    assets=load_release(Path('models'))
    arrays,rows=build_arrays(q,raw,assets,{s:pd.Timestamp('2100-01-01') for s in q.stay_id.unique()})
    old=np.load(root/'body_arrays_bedside.npz');old_filter=np.load(root/'labels.npz')['paired'][:,0]
    for field in ('input','current','mask','ages','presence','horizon'):
        np.testing.assert_array_equal(arrays[field][rows[old_filter]],old[field][old['query_rows']])
        np.testing.assert_array_equal(arrays[field][arrays['history'][rows[old_filter]]],
                                      old[field][old['history'][old['query_rows']]])
    label=np.load(root/'map_all_query_labels.npz');eligible=label['paired'][:,0]
    col=assets.checkpoint['variables'].index('map')
    np.testing.assert_allclose(arrays['current'][rows[eligible],col]*assets.scaler.scales[col]+assets.scaler.medians[col],
                               label['current'][eligible,0],atol=1e-4,rtol=1e-6)
    np.savez_compressed(path,**arrays,query_rows=rows)
    report=dict(complete=True,training_performed=False,runtime_modified=False,queries=len(q),
        eligible_map_pairs=int(eligible.sum()),old_query_and_history_exact=True,
        bindings={str(p):sha256(p) for p in (parent,path,root/'body_labs.parquet',root/'qualified_bedside_events.parquet',
            root/'body_arrays_bedside.npz',Path(__file__),Path('whole_body_jepa/availability_inputs.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='bindings'})


if __name__=='__main__':main()
