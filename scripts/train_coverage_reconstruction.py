"""Frozen-protocol coverage-sampling arm on the broad raw history cache."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.coverage_sampling import coverage_indices
from scripts.encoder_fold_scaling import transform_example
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--arm',choices=('real','process'),required=True)
    p.add_argument('--through-epoch',type=int,choices=range(1,9),required=True)
    args=p.parse_args();root=Path('data/research_recovery/additional_development_domains_20260921')
    cr=Path('artifacts/native_reconstruction_additional_full_cache_20260922.json')
    report=json.loads(cr.read_text());cache_path=root/'full_histories_20260922.sqlite'
    for name,h in report['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('Changed cache dependency')
    if sha256(cache_path)!=report['private_sha256']:raise ValueError('Changed cache')
    if any(report['unsupported_input_examples_by_fold']):raise ValueError('Unsupported inputs')
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt'
    plan=torch.load(fp,weights_only=False);scaling=torch.load(sp,weights_only=False)
    for path,h in scaling['bindings'].items():
        if sha256(Path(path))!=h:raise ValueError('Changed scaling/partition binding')
    roles=plan['folds'][args.fold];variables=scaling['variables'];n=len(variables)
    scaler=scaling['scalers'][args.fold];centers=scaling['centers'];people=scaling['people']
    fit=np.array([roles[x]=='fit' for x in people]);cols=np.flatnonzero(scaler['counts'])
    provenance=root/'provenance.parquet'
    components=pd.read_parquet(provenance)[['uniquepid','component']].drop_duplicates()
    if components.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    component=dict(components.itertuples(index=False,name=None))
    pools={int(c):np.flatnonzero(fit & np.isfinite(centers[:,c])) for c in cols}
    weights={c:bounded_weights([component[people[i]] for i in ids]) for c,ids in pools.items()}
    medians={c:float(np.median((centers[ids,c]-scaler['location'][c])/scaler['scale'][c])) for c,ids in pools.items()}
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    cp=Path('models/whole_body_joint_jepa.pt');schema=torch.load(cp,weights_only=False)
    if schema['variables']!=variables:raise ValueError('Schema mismatch')
    model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
    with torch.no_grad():
        model.readout.weight.zero_();model.readout.bias.zero_()
        for c,value in medians.items():model.readout.bias[c]=value
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);rng=np.random.default_rng(19)
    paths=[Path(__file__),cr,fp,sp,cp,provenance,Path('RECONSTRUCTION_COVERAGE_TRAINING_PROTOCOL_20260925.md'),Path('scripts/coverage_sampling.py'),
        Path('scripts/train_encoder_crossfit.py'),Path('scripts/encoder_fold_scaling.py'),Path('scripts/bounded_site_sampling.py'),Path('whole_body_joint_jepa.py')]
    paths+=list(Path('whole_body_jepa').glob('*.py'));paths+=list(Path('osler_jepa').glob('*.py'))
    bindings={str(p):sha256(p) for p in paths}
    destdir=root/'coverage_training_20260925';destdir.mkdir(exist_ok=True)
    dest=destdir/f'fold{args.fold}_{args.arm}.pt'
    ck=dict(epoch=0,log=[],bindings=bindings,fold=args.fold,arm=args.arm,
            variables=variables,targets=[variables[c] for c in cols],medians=medians,seen=set(),pending={c:np.array([],dtype=np.int64) for c in pools})
    if dest.exists():
        ck=torch.load(dest,weights_only=False)
        if ck['bindings']!=bindings:raise ValueError('Changed resume dependencies')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
    cache=ReconstructionCache(cache_path)
    try:
        for epoch in range(ck['epoch']+1,args.through_epoch+1):
            schedule=[]
            for c,ids in pools.items():
                draws,ck['pending'][c]=coverage_indices(weights[c],rng,1072,ck['pending'][c])
                schedule.extend((int(ids[i]),c) for i in draws)
            rng.shuffle(schedule);loss_sum=0.
            for step,(row,c) in enumerate(schedule,1):
                patient=people[row]
                if roles[patient]!='fit':raise ValueError('Non-fit gradient input')
                e=transform_example(cache.get(patient,c),scaler)
                x=e['tokens']
                if args.arm=='process':x=x.clone();x[:,:n]=0
                optimizer.zero_grad(set_to_none=True)
                pred=model(x,e['event_hours'],e['anchor_hour'],variables[c])
                loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach());ck['seen'].add((row,c))
                if step%5000==0:print('Progress',args.fold,args.arm,epoch,step,len(schedule),flush=True)
            ck['log'].append(dict(epoch=epoch,updates=len(schedule),fit_draw_loss=loss_sum/len(schedule),unique_fit_pairs_seen=len(ck['seen'])))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            tmp=dest.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(dest)
            print('Completed',args.fold,args.arm,ck['log'][-1],flush=True)
    finally:cache.close()
    out=Path(f'artifacts/native_reconstruction_coverage_training_fold{args.fold}_{args.arm}_20260925.json')
    out.write_text(json.dumps(dict(fold=args.fold,arm=args.arm,epoch=ck['epoch'],planned_epochs=8,
        targets=ck['targets'],fit_people=int(fit.sum()),available_fit_pairs=sum(map(len,pools.values())),
        log=ck['log'],bindings=bindings,private_sha256=sha256(dest),runtime_modified=False,promotion_allowed=False,
        limitation='Staged fixed-budget current reconstruction training only. No outer evaluation or promotion.'),indent=2)+'\n')


if __name__=='__main__':main()

