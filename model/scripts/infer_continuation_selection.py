"""Full source-selection inference for the frozen epoch-10 candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import inverse_value, transform_example
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=(1,2,3,4),required=True)
    args=p.parse_args()
    root=Path('data/research_recovery/additional_development_domains_20260921')
    folder=root/'continuation_selection_20260930';folder.mkdir(exist_ok=True)
    output=folder/f'fold{args.fold}.parquet'
    if output.exists() or output.with_suffix('.json').exists():raise ValueError('Existing source predictions')
    original_path=root/f'coverage_selection_20260926/fold{args.fold}.parquet'
    meta=json.loads(original_path.with_suffix('.json').read_text())
    if sha256(original_path)!=meta['prediction_sha256']:raise ValueError('Changed original queries')
    original=pd.read_parquet(original_path)
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt';schema_path=Path('models/whole_body_joint_jepa.pt')
    roles=torch.load(fp,weights_only=False)['folds'][args.fold]
    scaling=torch.load(sp,weights_only=False);scaler=scaling['scalers'][args.fold]
    variables=scaling['variables'];source_fold=0 if args.fold==2 else args.fold
    if original.duplicated(['patient','target']).any() or not original.fold.eq(args.fold).all():
        raise ValueError('Invalid source identity')
    if not all(roles[p]=='selection' for p in original.patient):raise ValueError('Wrong query role')
    assessed=json.loads(Path('artifacts/native_reconstruction_continuation_outer_20260929.json').read_text())
    # Partition, transforms and checkpoints must be those of the completed assessment.
    for path in (fp,sp):
        if sha256(path)!=assessed['bindings'][str(path)]:raise ValueError('Changed assessment contract')
    outer_meta_path=root/f'continuation_outer_20260929/fold{args.fold}.json'
    if sha256(outer_meta_path)!=assessed['bindings'][str(outer_meta_path)]:raise ValueError('Changed weight provenance')
    outer_meta=json.loads(outer_meta_path.read_text())
    paths=[Path(__file__),original_path,original_path.with_suffix('.json'),fp,sp,schema_path,
           outer_meta_path,Path('artifacts/native_reconstruction_continuation_outer_20260929.json'),
           Path('scripts/encoder_fold_scaling.py'),Path('scripts/train_encoder_crossfit.py')]
    for name,digest in outer_meta['bindings'].items():
        if sha256(Path(name))!=digest:raise ValueError('Changed inference dependency: '+name)
    schema=torch.load(schema_path,weights_only=False)
    if schema['variables']!=variables:raise ValueError('Changed schema')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    models={}
    for arm in ('real','process'):
        path=root/f'monitored_continuation_20260929/fold{source_fold}_{arm}_epoch10.pt'
        if sha256(path)!=outer_meta['bindings'][str(path)]:raise ValueError('Changed model weights')
        ck=torch.load(path,weights_only=False)
        if (ck['epoch'],ck['fold'],ck['arm'])!=(10,source_fold,arm):raise ValueError('Wrong model identity')
        model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
        model.load_state_dict(ck['state_dict'],strict=True);models[arm]=model;paths.append(path)
    cache_path=root/'full_histories_20260922.sqlite';paths.append(cache_path)
    cache=ReconstructionCache(cache_path);rows=[]
    try:
        keys=[(p,int(c)) for p,c in cache.db.execute('SELECT patient,target FROM examples ORDER BY patient,target')
              if roles[p]=='selection']
        if [(p,variables[c]) for p,c in keys]!=list(zip(original.patient,original.target)):
            raise ValueError('Changed complete query grid')
        print('Verified full selection',args.fold,len(keys),'queries',flush=True)
        with torch.inference_mode():
            for number,(patient,c) in enumerate(keys,1):
                e=transform_example(cache.get(patient,c),scaler)
                row=dict(patient=patient,target=variables[c],fold=args.fold)
                for arm,model in models.items():
                    tokens=e['tokens']
                    if arm=='process':tokens=tokens.clone();tokens[:,:len(variables)]=0
                    value=float(model(tokens,e['event_hours'],e['anchor_hour'],variables[c])[0])
                    row[arm+'10']=float(inverse_value(value,c,scaler))
                    if not np.isfinite(row[arm+'10']):raise ValueError('Nonfinite prediction')
                rows.append(row)
                if number%10000==0:print('Selection progress',args.fold,number,len(keys),flush=True)
    finally:cache.close()
    frame=pd.DataFrame(rows)
    pd.testing.assert_frame_equal(original[['patient','target','fold']],frame[['patient','target','fold']])
    temp=output.with_suffix('.tmp');frame.to_parquet(temp,index=False);temp.replace(output)
    report=dict(fold=args.fold,source_fold=source_fold,role='selection',epoch=10,
        examples=len(frame),patients=int(frame.patient.nunique()),targets=int(frame.target.nunique()),
        prediction_sha256=sha256(output),bindings={str(path):sha256(path) for path in paths},
        equivalent_inference_reuse=False,candidate_only=True,promotion_allowed=False,runtime_modified=False,
        limitation='Frozen full source-selection development inference; no model fitting, epoch selection or independent replication claim.')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed source selection',args.fold,len(frame),flush=True)


if __name__=='__main__':main()
