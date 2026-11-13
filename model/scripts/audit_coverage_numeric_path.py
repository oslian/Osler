"""Fixed hash diagnostic: matched original and neutralized numeric inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_expanded_reconstruction_numeric_path import neutralize_numbers
from scripts.encoder_fold_scaling import transform_example, inverse_value
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    output=Path('artifacts/native_reconstruction_coverage_numeric_path_20260929.json')
    if output.exists(): raise ValueError('Existing evidence')
    scaling=torch.load(root/'scaling_20260922.pt',weights_only=False)
    plan=torch.load(root/'partitions_20260922.pt',weights_only=False)
    schema=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)
    variables=scaling['variables']; n=len(variables)
    cache_path=root/'full_histories_20260922.sqlite'
    cache_report=json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    if sha256(cache_path)!=cache_report['private_sha256']: raise ValueError('Changed cache')
    cache=ReconstructionCache(cache_path)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results={};checked={};bindings={str(Path(__file__)):sha256(Path(__file__))}
    try:
        for fold in range(5):
            p=root/f'coverage_selection_20260926/fold{fold}.parquet'
            meta=json.loads(p.with_suffix('.json').read_text())
            if sha256(p)!=meta['prediction_sha256']:raise ValueError('Changed predictions')
            bindings[str(p)]=sha256(p)
            data=pd.read_parquet(p)
            if not all(plan['folds'][fold][x]=='selection' for x in data.patient.unique()):raise ValueError('Invalid role')
            models={}
            for arm,directory in [('real','coverage_training_20260925'),('replacement_real','training_20260923')]:
                path=root/directory/f'fold{fold}_real.pt'
                if sha256(path)!=meta['bindings'][str(path)]:raise ValueError('Changed weights')
                bindings[str(path)]=sha256(path)
                ck=torch.load(path,weights_only=False)
                for name,digest in ck['bindings'].items():
                    if name not in checked:checked[name]=sha256(Path(name))
                    if checked[name]!=digest:raise ValueError('Changed training dependency')
                model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
                model.load_state_dict(ck['state_dict'],strict=True);models[arm]=model
            results[str(fold)]={}
            with torch.inference_mode():
                for target,part in data.groupby('target',sort=True):
                    # Selection fixed by identity, not truth, error, or predictions.
                    order=part.patient.map(lambda x:hashlib.sha256(('coverage-numeric-v1:'+str(x)+':'+target).encode()).hexdigest())
                    selected=part.loc[order.sort_values().index[:128]]
                    c=variables.index(target);rows=[]
                    for row in selected.itertuples():
                        e=transform_example(cache.get(row.patient,c),scaling['scalers'][fold])
                        neutral=neutralize_numbers(e['tokens'],n)
                        assert torch.equal(neutral[:,n:],e['tokens'][:,n:])
                        values={}
                        for arm,model in models.items():
                            for suffix,x in [('',e['tokens']),('_neutral',neutral)]:
                                value=float(model(x,e['event_hours'],e['anchor_hour'],target)[0])
                                value=float(inverse_value(value,c,scaling['scalers'][fold]))
                                if not np.isfinite(value):raise ValueError('Invalid prediction')
                                if not suffix and value!=getattr(row,arm):raise ValueError('Original prediction mismatch')
                                values[arm+suffix]=value
                        rows.append({k:abs(v-row.truth)/row.fit_scale for k,v in values.items()})
                    errors=pd.DataFrame(rows)
                    means=errors.mean().to_dict()
                    results[str(fold)][target]=dict(patients=len(rows),normalized_mae=means,
                        coverage_neutralization_cost=means['real_neutral']-means['real'],
                        original_neutralization_cost=means['replacement_real_neutral']-means['replacement_real'])
                    print('Numeric path',fold,target,len(rows),flush=True)
    finally:cache.close()
    records=[v for fold in results.values() for v in fold.values()]
    summary=dict(target_folds=len(records),query_pairs=sum(v['patients'] for v in records),
        coverage_harmed_by_neutralization=sum(v['coverage_neutralization_cost']>0 for v in records),
        original_harmed_by_neutralization=sum(v['original_neutralization_cost']>0 for v in records),
        median_coverage_cost=float(np.median([v['coverage_neutralization_cost'] for v in records])),
        median_original_cost=float(np.median([v['original_neutralization_cost'] for v in records])))
    output.write_text(json.dumps(dict(summary=summary,folds=results,bindings=bindings,
        promotion_allowed=False,runtime_modified=False,
        limitation='128 identity-hash selected source queries per target-fold at most. Frozen out-of-distribution neutralization, not causal or full-cohort evidence; overlapping patients/fits. No retuning.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
