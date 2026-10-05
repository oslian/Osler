"""Build private derived gap stream and audit pre-query support only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.derived_anion_events import derive_events,VARIABLE
from whole_body_jepa.availability_inputs import snapshots
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/map_native_anchor_inputs_20260912.json')
    output=Path('artifacts/map_derived_anion_support_20260912.json')
    dest=root/'map_native_anchor_support/derived_anion_events.parquet'
    if output.exists() or dest.exists():
        raise ValueError('fresh outputs required')
    path=root/'body_labs.parquet'
    assert sha256(path)==json.loads(parent.read_text())['bindings'][str(path)]
    events=derive_events(pd.read_parquet(path))
    events.to_parquet(dest,index=False)
    q=pd.read_parquet(root/'map_native_anchor_support/queries.parquet')
    values=np.full(len(q),np.nan)
    groups=dict(tuple(events.groupby('stay_id')))
    for stay,frame in q.groupby('stay_id',sort=False):
        a,_=snapshots(groups.get(stay,events.iloc[:0]),pd.DatetimeIndex(frame.t),[VARIABLE],6)
        values[frame.index]=a[:,0]
    valid=np.load(root/'map_native_anchor_support/labels.npz')['paired'][:,0]
    roles=dict(np.load(root/'map_native_anchor_support/partition_7.npz'))
    summary={}
    for role,mask in roles.items():
        m=mask&valid
        observed=np.isfinite(values[m])
        patients=q.loc[m,'uniquepid'].to_numpy()
        summary[role]=dict(rows=int(m.sum()),observed_rows=int(observed.sum()),
            observed_patients=int(pd.Series(patients[observed]).nunique()),
            patient_equal_observed_fraction=float(pd.Series(observed).groupby(patients).mean().mean()))
    report=dict(complete=True,training_performed=False,runtime_modified=False,
        events=len(events),valid_versions=int(events.quality_valid.sum()),roles=summary,
        limitation='Same timestamp not specimen ID; saved qualified components only; no forecast gain measured.',
        bindings={str(p):sha256(p) for p in (parent,path,dest,Path(__file__),Path('whole_body_jepa/derived_anion_events.py'))})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='bindings'})


if __name__=='__main__':
    main()
