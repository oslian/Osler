"""Verify old source transforms under the expanded native index layout."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_event_encoder import native_arrays
from whole_body_jepa.availability_inputs import source_patient_support, fit_numeric_rescale
from whole_body_jepa.history_event_input import prepare_events
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/map_expanded_transforms_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    assets=load_release(Path('models'));v=assets.checkpoint['variables'];col=v.index('map')
    q=pd.read_parquet(root/'queries.parquet');old=np.load(root/'labels.npz')['paired'][:,0]
    eligible=np.load(root/'map_all_query_labels.npz')['paired'][:,0]
    results={};paths=[]
    for tag,b,n in (('old','body_arrays_bedside.npz','native_tokens_bedside.npz'),
                    ('expanded','map_all_query_inputs.npz','map_all_native_tokens.npz')):
        base=dict(np.load(root/b));native=dict(np.load(root/n));paths.extend([root/b,root/n])
        arrays,rows,values=native_arrays(base,native,len(v),base['presence'].shape[1])
        subset=old if tag=='old' else np.ones(len(q),bool)
        results[tag]=[]
        for seed in (7,19,31,43,59,71,83):
            fit=np.load(root/f'partition_{seed}.npz')['fit']&eligible&old
            fit=fit[subset]
            allowed=source_patient_support(base,base['query_rows'],q.loc[subset,'uniquepid'],fit)>0
            center,scale=fit_numeric_rescale(arrays,rows,fit,col)
            _,ec,es=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
            results[tag].append((allowed,center,scale,ec,es))
        del arrays,rows,values,base,native
    for left,right in zip(results['old'],results['expanded']):
        for a,b in zip(left,right):np.testing.assert_array_equal(a,b)
    out.write_text(json.dumps(dict(complete=True,training_performed=False,runtime_modified=False,
        exact_source_transform_seeds=7,fields=['allowed','numeric_center','numeric_scale','event_center','event_scale'],
        bindings={str(p):sha256(p) for p in paths+[Path(__file__),root/'queries.parquet',root/'labels.npz',
            root/'map_all_query_labels.npz',Path('whole_body_jepa/native_event_encoder.py'),
            Path('whole_body_jepa/availability_inputs.py'),Path('whole_body_jepa/history_event_input.py')]
            +[root/f'partition_{s}.npz' for s in (7,19,31,43,59,71,83)]}),indent=2)+'\n')
    print('All seven source transforms are bit-exact')


if __name__=='__main__':main()
