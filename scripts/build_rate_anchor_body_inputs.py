"""Build truthful whole-lab history for existing record-anchor labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_availability_jepa import extract
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/rate_anchor_body_inputs_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    report_path=Path('artifacts/rate_anchored_potassium_20260911.json')
    reference=json.loads(report_path.read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(root/name)!=reference['binding'][str(root/name)]:raise ValueError('cache changed')
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    rawpath=root/'body_labs.parquet'
    extract('eicu',set(q.stay_id),variables,output=rawpath)
    raw=pd.read_parquet(rawpath)
    arrays,rows=build_arrays(q,raw,assets,{s:pd.Timestamp('2100-01-01') for s in q.stay_id.unique()})
    col=variables.index('potassium')
    current=arrays['current'][rows,col]*assets.scaler.scales[col]+assets.scaler.medians[col]
    np.testing.assert_allclose(current,labels['current'][valid,0],atol=1e-5,rtol=1e-6)
    p=root/'body_arrays.npz';np.savez_compressed(p,**arrays,query_rows=rows)
    support={v:int(arrays['mask'][rows,i].sum()) for i,v in enumerate(variables)}
    report=dict(complete=True,status='whole_lab_inputs_not_training',runtime_modified=False,rows=len(q),
                subjects=int(q.uniquepid.nunique()),observed_current_support=support,
                current_potassium_reproduced=True,causal_claim_allowed=False,
                limitation='labs only; treatment retained separately, no verified infusion availability, not full bedside body state',
                binding={str(f):sha256(f) for f in [Path(__file__),report_path,rawpath,p]})
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
