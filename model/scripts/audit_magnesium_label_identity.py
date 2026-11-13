"""Validate future event identity and measurement-time separation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');parent=Path('artifacts/rate_multitarget_labels_20260911.json')
    ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_label_identity_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    l=dict(np.load(root/'multitarget_labels.npz'));v=list(l['variables']);k=v.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];ok=l['paired'][:,k]&l['paired'][:,v.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    ev=pd.read_parquet(root/'body_labs.parquet');ev=ev.loc[ev.variable.eq('magnesium')].sort_values('available_at')
    lookup={(str(r.stay_id),str(r.source_id)):r for r in ev.itertuples()}
    for i in np.flatnonzero(ok):
        event=lookup[(str(q.stay_id.iloc[i]),str(l['future_source_id'][i,k]))]
        delta=(event.measured_at-q.t.iloc[i]).total_seconds()/3600
        np.testing.assert_allclose(delta,l['actual_delta_t'][i,k],rtol=0,atol=1e-9)
        np.testing.assert_allclose(event.value,l['future'][i,k],rtol=0,atol=1e-9)
        assert event.measured_at>q.t.iloc[i]
        assert l['measurement_age'][i,k]>=0
    age=l['measurement_age'][ok,k];dt=l['actual_delta_t'][ok,k];change=l['future'][ok,k]-l['current'][ok,k]
    report=dict(complete=True,runtime_modified=False,paired_rows=int(ok.sum()),future_identity_and_time_verified=True,
        exact_same_value_fraction=float(np.mean(change==0)),
        quantiles={name:np.quantile(a,[0,.25,.5,.75,1]).tolist() for name,a in [('current_age',age),('anchor_to_future',dt),('draw_to_draw',age+dt),('signed_change',change)]},
        binding={str(p):sha256(p) for p in (parent,Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='binding'},indent=2))


if __name__=='__main__':main()
