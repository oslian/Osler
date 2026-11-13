"""Frozen identity-only source monitoring, separate from optimization and gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256


def monitor_keys(frame, roles, per_component=32):
    keys=frame[['patient','target','component']].copy()
    if keys.duplicated(['patient','target']).any() or not all(roles[p]=='selection' for p in keys.patient):
        raise ValueError('Unique selection-only identities required')
    keys['_order']=[hashlib.sha256(('fixed-monitor-v1:'+str(p)+':'+str(t)).encode()).hexdigest() for p,t in zip(keys.patient,keys.target)]
    return keys.sort_values(['target','component','_order','patient']).groupby(['target','component'],sort=False).head(per_component).drop(columns='_order').reset_index(drop=True)


def monitor_scores(keys, predictions):
    if predictions.duplicated(['patient','target']).any():raise ValueError('Duplicate predictions')
    data=keys.merge(predictions,on=['patient','target','component'],how='left',validate='one_to_one',indicator=True)
    if not data._merge.eq('both').all():raise ValueError('Incomplete fixed monitor')
    if not np.isfinite(data[['truth','fit_scale','prediction']]).all().all() or not data.fit_scale.gt(0).all():raise ValueError('Invalid numeric output')
    error=abs(data.prediction-data.truth)/data.fit_scale
    data=data.assign(error=error,loss=np.where(error<1,.5*error**2,error-.5))
    return {target:dict(patients=len(part),components=int(part.component.nunique()),
        normalized_patient_mae=float(part.error.mean()),
        normalized_component_mae=float(part.groupby('component').error.mean().mean()),
        smooth_l1=float(part.loss.mean())) for target,part in data.groupby('target',sort=True)}


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    folder=root/'fixed_monitor_20260929';folder.mkdir(exist_ok=True)
    output=Path('artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json')
    if output.exists():raise ValueError('Existing monitor baseline')
    fp=root/'partitions_20260922.pt';plan=torch.load(fp,weights_only=False)
    results={};bindings={str(fp):sha256(fp),str(Path(__file__)):sha256(Path(__file__))}
    for fold in range(5):
        path=root/f'coverage_selection_20260926/fold{fold}.parquet'
        meta=json.loads(path.with_suffix('.json').read_text())
        if sha256(path)!=meta['prediction_sha256']:raise ValueError('Changed predictions')
        data=pd.read_parquet(path);keys=monitor_keys(data,plan['folds'][fold])
        kp=folder/f'fold{fold}_keys.parquet'
        if kp.exists():pd.testing.assert_frame_equal(pd.read_parquet(kp),keys)
        else:keys.to_parquet(kp,index=False)
        bindings[str(path)]=sha256(path);bindings[str(kp)]=sha256(kp)
        results[str(fold)]={arm:monitor_scores(keys,data[['patient','target','component','truth','fit_scale',arm]].rename(columns={arm:'prediction'})) for arm in ['real','process','replacement_real','replacement_process','population']}
    output.write_text(json.dumps(dict(epoch=8,folds=results,bindings=bindings,
        promotion_allowed=False,runtime_modified=False,
        limitation='Fixed source-only diagnostic monitor, established after epoch8. Cannot retrospectively establish a learning curve or certify new performance; no future checkpoint selection policy yet.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps({fold:dict(targets=len(v['real']),pairs=sum(r['patients'] for r in v['real'].values())) for fold,v in results.items()},indent=2))


if __name__=='__main__':main()
