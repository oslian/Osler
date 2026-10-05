"""Verify completed source inference without interpreting partial-fold efficacy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.fixed_reconstruction_monitor import monitor_scores
from whole_body_jepa.multisource_intervals import sha256


def verify_frames(base, fresh, meta, roles, fold, expected):
    if (meta['fold'],meta['source_fold'],meta['role'],meta['epoch'])!=(fold,0 if fold==2 else fold,'selection',10):
        raise ValueError('Wrong inference contract')
    if meta['candidate_only'] is not True or meta['promotion_allowed'] is not False:
        raise ValueError('Unexpected promotion metadata')
    pd.testing.assert_frame_equal(base[['patient','target','fold']],fresh[['patient','target','fold']])
    if fresh.duplicated(['patient','target']).any() or not fresh.fold.eq(fold).all():
        raise ValueError('Invalid identity')
    if not all(roles[p]=='selection' for p in fresh.patient):raise ValueError('Nonselection query')
    if set(zip(fresh.patient,fresh.target))!=expected:raise ValueError('Incomplete eligibility grid')
    if len(fresh)!=meta['examples'] or fresh.patient.nunique()!=meta['patients'] or fresh.target.nunique()!=meta['targets']:
        raise ValueError('Incorrect support counts')
    if not np.isfinite(fresh[['real10','process10']]).all().all():raise ValueError('Nonfinite prediction')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    args=parser.parse_args();fold=args.fold
    root=Path('data/research_recovery/additional_development_domains_20260921')
    path=root/f'continuation_selection_20260930/fold{fold}.parquet'
    meta_path=path.with_suffix('.json');meta=json.loads(meta_path.read_text())
    if sha256(path)!=meta['prediction_sha256']:raise ValueError('Changed prediction payload')
    for name,digest in meta['bindings'].items():
        if sha256(Path(name))!=digest:raise ValueError('Changed dependency: '+name)
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt'
    roles=torch.load(fp,weights_only=False)['folds'][fold]
    scaling=torch.load(sp,weights_only=False)
    expected={(patient,scaling['variables'][c]) for i,patient in enumerate(scaling['people'])
              if roles[patient]=='selection' for c in np.flatnonzero(np.isfinite(scaling['centers'][i]))}
    old_path=root/f'coverage_selection_20260926/fold{fold}.parquet'
    old_meta=json.loads(old_path.with_suffix('.json').read_text())
    if sha256(old_path)!=old_meta['prediction_sha256']:raise ValueError('Changed source baseline')
    old=pd.read_parquet(old_path);new=pd.read_parquet(path)
    verify_frames(old,new,meta,roles,fold,expected)
    baseline_path=Path('artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json')
    baseline=json.loads(baseline_path.read_text())
    kp=root/f'fixed_monitor_20260929/fold{fold}_keys.parquet'
    if sha256(kp)!=baseline['bindings'][str(kp)]:raise ValueError('Changed monitor identities')
    keys=pd.read_parquet(kp);diffs=[]
    summary_path=Path('artifacts/native_reconstruction_monitored_continuation_summary_20260929.json')
    summary=json.loads(summary_path.read_text())
    paths=[Path(__file__),Path('scripts/fixed_reconstruction_monitor.py'),path,meta_path,
           fp,sp,old_path,old_path.with_suffix('.json'),baseline_path,kp,summary_path]
    for arm in ('real','process'):
        rows=old[['patient','target','component','truth','fit_scale']].copy()
        rows['prediction']=new[arm+'10'].to_numpy()
        scores=monitor_scores(keys,rows)
        monitor_path=root/f'monitored_continuation_20260929/fold{fold}_{arm}_epoch10_monitor.json'
        if sha256(monitor_path)!=summary['bindings'][str(monitor_path)]:raise ValueError('Changed reference monitor')
        reference=json.loads(monitor_path.read_text())['targets'];paths.append(monitor_path)
        if scores.keys()!=reference.keys():raise ValueError('Incomplete monitor targets')
        for target in scores:
            for metric in ('normalized_patient_mae','normalized_component_mae','smooth_l1'):
                a,b=scores[target][metric],reference[target][metric]
                if not np.isclose(a,b,rtol=1e-10,atol=1e-10):raise ValueError('Monitor parity failure')
                diffs.append(abs(a-b))
    output=Path(f'artifacts/native_reconstruction_continuation_selection_fold{fold}_verification_20260930.json')
    if output.exists():raise ValueError('Existing verification')
    report=dict(fold=fold,pairs=len(new),patients=int(new.patient.nunique()),targets=int(new.target.nunique()),
        prediction_sha256=sha256(path),fixed_monitor_max_abs_difference=max(diffs),
        equivalent_inference_reuse=meta['equivalent_inference_reuse'],
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Integrity and monitor parity only; no new efficacy, independent replication or promotion.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bindings'},indent=2))


if __name__=='__main__':main()
