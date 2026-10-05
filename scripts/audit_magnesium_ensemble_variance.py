"""Common-hospital fixed ensemble and exact squared-error variance identity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');parent=Path('artifacts/native_magnesium_matched_20260911.json')
    ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_ensemble_variance_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    labels=dict(np.load(root/'multitarget_labels.npz'));v=list(labels['variables']);k=v.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=labels['paired'][:,k]&labels['paired'][:,v.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,k];mask=None;predictions={'own':[],'real':[]}
    for e in ref['seeds']:
        seed=e['seed'];m=np.load(root/f'partition_{seed}.npz')['hospital'][valid]&eligible
        if mask is not None:np.testing.assert_array_equal(mask,m)
        mask=m;p=dict(np.load(root/f'magnesium_matched_predictions_{seed}.npz'))
        for a in predictions:predictions[a].append(p[a][m])
    frame=q.loc[mask];truth=frame.truth.to_numpy();ensemble={};stats={}
    def equal(a):return float(pd.Series(a,index=frame.subject_id.to_numpy()).groupby(level=0).mean().mean())
    for arm,values in predictions.items():
        values=np.array(values,dtype=np.float64);mean=values.mean(axis=0);ensemble[arm]=mean
        member_mse=((values-truth)**2).mean(axis=0);variance=values.var(axis=0);ensemble_mse=(mean-truth)**2
        np.testing.assert_allclose(member_mse,variance+ensemble_mse,rtol=1e-10,atol=1e-12)
        stats[arm]=dict(mean_member_mse=equal(member_mse),prediction_variance=equal(variance),ensemble_mse=equal(ensemble_mse),
            ensemble_mae=patient_equal_mae(frame,truth,mean))
    persistence=labels['current'][mask,k]
    report=dict(complete=True,runtime_modified=False,fitting_performed=False,patients=int(frame.subject_id.nunique()),stats=stats,
        comparisons=dict(whole_vs_own=paired(frame,ensemble['real'],ensemble['own'],'mg-fixed-ensemble-own',5000),
        whole_vs_persistence=paired(frame,ensemble['real'],persistence,'mg-fixed-ensemble-persistence',5000)),
        binding={str(p):sha256(p) for p in (parent,Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='binding'},indent=2))


if __name__=='__main__':main()
