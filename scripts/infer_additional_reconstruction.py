"""Frozen epoch8 predictions on complete outer patient-target folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import transform_example, inverse_value
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    args=parser.parse_args();root=Path('data/research_recovery/additional_development_domains_20260921')
    folder=root/'outer_20260924';folder.mkdir(exist_ok=True)
    dest=folder/f'fold{args.fold}.parquet';meta=folder/f'fold{args.fold}.json'
    if dest.exists() or meta.exists():raise ValueError('Existing outer predictions')
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt';cp=Path('models/whole_body_joint_jepa.pt')
    paths=[Path(__file__),fp,sp,cp,Path('scripts/encoder_fold_scaling.py'),Path('scripts/train_encoder_crossfit.py'),
           Path('RECONSTRUCTION_ADDITIONAL_EVALUATION_PROTOCOL_20260924.md')]
    checked={};checkpoints={}
    certificate=Path('artifacts/native_reconstruction_additional_fold_reuse_20260923.json')
    reuse=json.loads(certificate.read_text());paths.append(certificate)
    # Certify all training complete before viewing any outer predictions.
    for fold in range(5):
        for arm in ('real','process'):
            path=root/f'training_20260923/fold{fold}_{arm}.pt';paths.append(path)
            if fold==2:
                digest=reuse['destinations'][arm]['sha256']
            else:
                report_path=Path(f'artifacts/native_reconstruction_additional_training_fold{fold}_{arm}_20260923.json')
                report=json.loads(report_path.read_text());paths.append(report_path)
                if report['epoch']!=8:raise ValueError('Training incomplete')
                digest=report['private_sha256']
            if sha256(path)!=digest:raise ValueError('Changed trained checkpoint')
            ck=torch.load(path,weights_only=False)
            if ck['epoch']!=8 or ck['fold']!=fold or ck['arm']!=arm:raise ValueError('Wrong checkpoint role')
            for name,h in ck['bindings'].items():
                if name not in checked:checked[name]=sha256(Path(name))
                if checked[name]!=h:raise ValueError('Changed training dependency')
            if fold==args.fold:checkpoints[arm]=ck
    cache_report_path=Path('artifacts/native_reconstruction_additional_full_cache_20260922.json')
    cache_report=json.loads(cache_report_path.read_text());paths.append(cache_report_path)
    cache_path=root/'full_histories_20260922.sqlite'
    if sha256(cache_path)!=cache_report['private_sha256']:raise ValueError('Changed histories')
    plan=torch.load(fp,weights_only=False);roles=plan['folds'][args.fold]
    scaling=torch.load(sp,weights_only=False);scaler=scaling['scalers'][args.fold]
    variables=scaling['variables'];n=len(variables);schema=torch.load(cp,weights_only=False)
    if schema['variables']!=variables:raise ValueError('Schema mismatch')
    provenance=root/'provenance.parquet';paths.append(provenance)
    links=pd.read_parquet(provenance)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    components=dict(links.itertuples(index=False,name=None))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    models={}
    for arm,ck in checkpoints.items():
        model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
        model.load_state_dict(ck['state_dict'],strict=True);models[arm]=model
    cache=ReconstructionCache(cache_path);rows=[]
    try:
        keys=[(p,int(c)) for p,c in cache.db.execute('SELECT patient,target FROM examples ORDER BY patient,target') if roles[p]=='test']
        if not keys:raise ValueError('Empty outer fold')
        with torch.inference_mode():
            for number,(patient,c) in enumerate(keys,1):
                original=cache.get(patient,c);e=transform_example(original,scaler)
                row=dict(patient=patient,target=variables[c],fold=args.fold,component=components[patient],
                    truth=original['truth_normalized'],fit_scale=float(scaler['scale'][c]),
                    population=float(inverse_value(checkpoints['real']['medians'][c],c,scaler)))
                for arm,model in models.items():
                    x=e['tokens']
                    if arm=='process':x=x.clone();x[:,:n]=0
                    value=float(model(x,e['event_hours'],e['anchor_hour'],variables[c])[0])
                    row[arm]=float(inverse_value(value,c,scaler))
                    if not np.isfinite(row[arm]):raise ValueError('Nonfinite outer prediction')
                rows.append(row)
                if number%10000==0:print('Outer progress fold',args.fold,number,'of',len(keys),flush=True)
    finally:cache.close()
    frame=pd.DataFrame(rows)
    if len(frame)!=len(keys) or frame.duplicated(['patient','target']).any():raise ValueError('Outer identity mismatch')
    temp=dest.with_suffix('.tmp');frame.to_parquet(temp,index=False);temp.replace(dest)
    result=dict(fold=args.fold,examples=len(frame),patients=int(frame.patient.nunique()),
        bindings={str(p):sha256(p) for p in paths},prediction_sha256=sha256(dest),
        runtime_modified=False,promotion_allowed=False,
        limitation='Frozen outer inference only. Aggregate all folds before interpretation; no selection on these predictions.')
    meta.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'},indent=2),flush=True)


if __name__=='__main__':main()
