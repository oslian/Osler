"""Audit whether residual fitting uses out-of-patient baseline predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/native_residual_training_exposure_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    parent=Path('artifacts/map_native_query_training_7_20260912.json')
    report=json.loads(parent.read_text())
    for name,h in report['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed baseline dependency')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz'));roles=dict(np.load(root/'partition_7.npz'))
    pred=np.load(root/'native_query_predictions_7.npz')['own']
    eligible=labels['paired'][:,0]
    pools=[np.asarray(g.index) for _,g in q.loc[roles['fit']&eligible].groupby('subject_id',sort=True)]
    rng=np.random.default_rng(7); seen=set(); seen_patients=set()
    for _ in range(report['arms']['own']['selected_epoch']):
        for _ in range(16):
            chosen=[int(rng.choice(pools[i])) for i in rng.integers(len(pools),size=32)]
            seen.update(chosen); seen_patients.update(q.loc[chosen,'subject_id'])
    rng=np.random.default_rng(20260913);results={}
    for role,count in (('fit',64),('tune',32),('evaluation',64),('hospital',64)):
        groups=[np.asarray(g.index) for _,g in q.loc[roles[role]&eligible].groupby('subject_id',sort=True)]
        rows=np.array([int(rng.choice(groups[i])) for i in rng.choice(len(groups),count,replace=False)])
        x=labels['current'][rows,0];e=labels['future'][rows,0]-pred[rows]
        coef=np.linalg.lstsq(np.column_stack((np.ones(len(x)),x)),e,rcond=None)[0]
        results[role]=dict(patients=count, baseline_fit_exposed_patients=int(q.loc[rows,'subject_id'].isin(seen_patients).sum()),
            exact_queries_seen=int(sum(int(r) in seen for r in rows)),
            baseline_mae=float(np.mean(abs(e))),mean_residual=float(e.mean()),median_residual=float(np.median(e)),
            residual_current_slope=float(coef[1]),residual_current_correlation=float(np.corrcoef(x,e)[0,1]))
    output.write_text(json.dumps(dict(complete=True,selected_baseline_epoch=report['arms']['own']['selected_epoch'],
        results=results,training_performed=False,runtime_modified=False,
        caveat='Descriptive exposure, not proof of causal overfit; tune also used for baseline epoch selection.',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),root/'partition_7.npz')}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
