"""Exact epoch8 continuation with immutable checkpoints and fixed source monitor."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.bounded_site_sampling import bounded_weights
from scripts.coverage_sampling import coverage_indices
from scripts.encoder_fold_scaling import transform_example,inverse_value
from scripts.fixed_reconstruction_monitor import monitor_scores
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=(0,1,3,4),required=True)
    p.add_argument('--arm',choices=('real','process'),required=True)
    p.add_argument('--through-epoch',type=int,choices=(8,9,10),required=True)
    args=p.parse_args();root=Path('data/research_recovery/additional_development_domains_20260921')
    parent=root/f'coverage_training_20260925/fold{args.fold}_{args.arm}.pt'
    report=json.loads(Path(f'artifacts/native_reconstruction_coverage_training_fold{args.fold}_{args.arm}_20260925.json').read_text())
    if sha256(parent)!=report['private_sha256']:raise ValueError('Changed parent')
    original=torch.load(parent,weights_only=False)
    for name,digest in original['bindings'].items():
        if sha256(Path(name))!=digest:raise ValueError('Changed training dependency')
    folder=root/'monitored_continuation_20260929';folder.mkdir(exist_ok=True)
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt'
    kp=root/f'fixed_monitor_20260929/fold{args.fold}_keys.parquet'
    baseline=Path('artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json')
    bm=json.loads(baseline.read_text())
    if sha256(kp)!=bm['bindings'][str(kp)]:raise ValueError('Changed monitor')
    bindings={str(x):sha256(x) for x in (Path(__file__),parent,fp,sp,kp,baseline,Path('scripts/fixed_reconstruction_monitor.py'),Path('RECONSTRUCTION_MONITORED_CONTINUATION_PROTOCOL_20260929.md'))}
    plan=torch.load(fp,weights_only=False);s=torch.load(sp,weights_only=False)
    roles=plan['folds'][args.fold];scaler=s['scalers'][args.fold];variables=s['variables'];n=len(variables)
    keys=pd.read_parquet(kp)
    if not all(roles[x]=='selection' for x in keys.patient):raise ValueError('Monitor role violation')
    fit=np.array([roles[x]=='fit' for x in s['people']])
    pools={int(c):np.flatnonzero(fit & np.isfinite(s['centers'][:,c])) for c in np.flatnonzero(scaler['counts'])}
    links=pd.read_parquet(root/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    component=dict(links.itertuples(index=False,name=None))
    weights={c:bounded_weights([component[s['people'][i]] for i in ids]) for c,ids in pools.items()}
    ck=original
    for epoch in (9,10):
        path=folder/f'fold{args.fold}_{args.arm}_epoch{epoch}.pt'
        if path.exists():
            if epoch>args.through_epoch:raise ValueError('Requested older stage')
            ck=torch.load(path,weights_only=False)
            if ck['continuation_bindings']!=bindings:raise ValueError('Changed resume dependencies')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    schema=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)
    model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
    cache_path=root/'full_histories_20260922.sqlite'
    cr=json.loads(Path('artifacts/native_reconstruction_additional_full_cache_20260922.json').read_text())
    if sha256(cache_path)!=cr['private_sha256']:raise ValueError('Changed histories')
    cache=ReconstructionCache(cache_path)
    def monitor():
        output=folder/f'fold{args.fold}_{args.arm}_epoch{ck["epoch"]}_monitor.json'
        if output.exists():
            saved=json.loads(output.read_text())
            if saved['bindings']!=bindings:raise ValueError('Changed monitor binding')
            return
        rows=[]
        with torch.inference_mode():
            for row in keys.itertuples():
                c=variables.index(row.target);e=transform_example(cache.get(row.patient,c),scaler);x=e['tokens']
                if args.arm=='process':x=x.clone();x[:,:n]=0
                value=float(model(x,e['event_hours'],e['anchor_hour'],row.target)[0])
                rows.append(dict(patient=row.patient,target=row.target,component=row.component,
                    prediction=float(inverse_value(value,c,scaler)),truth=float(inverse_value(e['truth_normalized'],c,scaler)),fit_scale=float(scaler['scale'][c])))
        scores=monitor_scores(keys,pd.DataFrame(rows))
        if ck['epoch']==8:
            old=bm['folds'][str(args.fold)][args.arm]
            for t in scores:
                for metric in ('normalized_patient_mae','normalized_component_mae','smooth_l1'):
                    if not np.isclose(scores[t][metric],old[t][metric],rtol=1e-10,atol=1e-10):raise ValueError('Epoch8 monitor mismatch')
        output.write_text(json.dumps(dict(epoch=ck['epoch'],bindings=bindings,targets=scores,promotion_allowed=False),indent=2)+'\n')
        print('Monitor completed',args.fold,args.arm,ck['epoch'],len(rows),flush=True)
    try:
        monitor()
        for epoch in range(ck['epoch']+1,args.through_epoch+1):
            schedule=[]
            for c,ids in pools.items():
                draws,ck['pending'][c]=coverage_indices(weights[c],rng,1072,ck['pending'][c])
                schedule.extend((int(ids[i]),c) for i in draws)
            rng.shuffle(schedule);loss_sum=0.
            for step,(row,c) in enumerate(schedule,1):
                patient=s['people'][row]
                if roles[patient]!='fit':raise ValueError('Nonfit update')
                e=transform_example(cache.get(patient,c),scaler);x=e['tokens']
                if args.arm=='process':x=x.clone();x[:,:n]=0
                optimizer.zero_grad(set_to_none=True)
                pred=model(x,e['event_hours'],e['anchor_hour'],variables[c])
                loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach());ck['seen'].add((row,c))
                if step%5000==0:print('Progress',args.fold,args.arm,epoch,step,len(schedule),flush=True)
            ck['log'].append(dict(epoch=epoch,updates=len(schedule),fit_draw_loss=loss_sum/len(schedule),unique_fit_pairs_seen=len(ck['seen'])))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),continuation_bindings=bindings)
            dest=folder/f'fold{args.fold}_{args.arm}_epoch{epoch}.pt'
            if dest.exists():raise ValueError('Immutable checkpoint exists')
            tmp=dest.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(dest)
            monitor()
    finally:cache.close()


if __name__=='__main__':main()
