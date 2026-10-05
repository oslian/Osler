"""Fresh, source-fold-local encoder comparison; staged execution, fixed scope."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from scripts.reconstruction_source_store import SourceStore
from scripts.encoder_fold_support import supported_example
from scripts.bounded_site_sampling import bounded_weights, sample_indices
from scripts.train_joint_source_reconstruction import equal_group_median


def fresh_encoder(checkpoint):
    """Read shape/schema metadata, never checkpoint learned parameters."""
    a=checkpoint['architecture'];s=checkpoint['state_dict']
    model=JointWholeBodyJEPAWorldModel(
        n_state=len(checkpoint['variables']),input_dim=int(checkpoint['input_dim']),
        module_count=len(checkpoint['modules']),hidden=int(checkpoint['hidden']),
        latent=int(checkpoint['latent']),token_dim=int(s['variable_embedding'].shape[1]),
        variable_module_ids=s['variable_module_ids'].numpy(),
        variable_module_membership=s['variable_module_membership'].numpy(),
        horizon_values=s['horizon_bucket_values'].numpy(),regime_count=int(a['regime_count']),
        target_horizon_regime_adapter=bool(a['target_horizon_regime_adapter']),
        patient_residual_adapter=bool(a['patient_residual_adapter']),
        uncertainty_gate=bool(a['uncertainty_gate']),validated_edge_adapters=bool(a['validated_edge_adapters']),
        validated_edge_specs=(),measurement_time_head=bool(a['next_measurement_time_head']),
        belief_input_mode='observed',target_momentum=float(a['target_momentum']),
        target_min_observations=int(a['target_min_observations']),
        target_encoder_mode=str(a['target_encoder_mode']))
    expand_history_projection(model,6);model.belief_input_mode='observed_steps'
    return model


def epoch_plan(pools,weights,rng,draws=1072):
    columns=sorted({c for d in pools.values() for c,keys in d.items() if keys})
    plans=[]
    for c in columns:
        groups=[g for g in sorted(pools) if pools[g].get(c)]
        order=np.arange(draws)%len(groups);rng.shuffle(order)
        for index,g in enumerate(groups):
            count=int((order==index).sum())
            indices=sample_indices(weights[g][c],rng,count)
            plans.extend((g,pools[g][c][i]) for i in indices)
    rng.shuffle(plans)
    return plans


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--arm',choices=('real','process'),required=True)
    p.add_argument('--through-epoch',type=int,choices=range(1,9),required=True)
    args=p.parse_args()
    root=Path('data/research_recovery/broad_source_20260917')
    fp=root/'encoder_crossfit_folds_20260920.pt';sp=root/'encoder_fold_scaling_20260920.pt'
    support=Path('artifacts/native_reconstruction_encoder_fold_support_20260920.json')
    prior=json.loads(support.read_text())
    for name,h in prior['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('Changed input audit dependency')
    plan=torch.load(fp,weights_only=False);roles=plan['folds'][args.fold]
    scaler=torch.load(sp,weights_only=False)['scalers'][args.fold]
    manifest_path=root/'site_balance_manifest_20260920.pt'
    manifest=torch.load(manifest_path,weights_only=False)
    jp=root/'joint_source_manifest.pt';joint=torch.load(jp,weights_only=False)
    variables,targets=joint['variables'],joint['targets'];cols=[variables.index(t) for t in targets]
    stores={g:SourceStore([s for s in plan['shards'] if s['group']==g and roles[s['patient']]=='fit'],cols) for g in ('selected','broad')}
    pools={g:d.groups for g,d in stores.items()}
    if any(not any(pools[g][c] for g in pools) for c in cols):raise ValueError('Trained target unsupported')
    weights={g:{c:bounded_weights([manifest['sample_sites'][g,'fit',key[0],c] for key in keys]) for c,keys in d.items() if keys} for g,d in pools.items()}
    medians={c:equal_group_median([np.array([(stores[g].labels[k]-scaler['location'][c])/scaler['scale'][c] for k in pools[g][c]]) for g in pools if pools[g][c]]) for c in cols}
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    cp=Path('models/whole_body_joint_jepa.pt');schema=torch.load(cp,map_location='cpu',weights_only=False)
    if schema['variables']!=variables:raise ValueError('Architecture schema mismatch')
    model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
    with torch.no_grad():
        model.readout.weight.zero_();model.readout.bias.zero_()
        for c,value in medians.items():model.readout.bias[c]=value
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    paths=[Path(__file__),fp,sp,support,manifest_path,jp,cp,
        Path('RECONSTRUCTION_ENCODER_CROSSFIT_TRAINING_PROTOCOL_20260920.md'),
        Path('scripts/encoder_fold_support.py'),Path('scripts/encoder_fold_scaling.py'),
        Path('scripts/reconstruction_source_store.py'),Path('scripts/bounded_site_sampling.py'),
        Path('scripts/train_joint_source_reconstruction.py'),Path('whole_body_joint_jepa.py')]
    paths+=list(Path('whole_body_jepa').glob('*.py'));paths+=list(Path('osler_jepa').glob('*.py'))
    bindings={str(q):sha256(q) for q in paths}
    private=root/'encoder_crossfit_training_20260920';private.mkdir(exist_ok=True)
    dest=private/f'fold{args.fold}_{args.arm}.pt'
    rng=np.random.default_rng(19)
    ck=dict(epoch=0,log=[],bindings=bindings,fold=args.fold,arm=args.arm,
        variables=variables,targets=targets,medians=medians)
    if dest.exists():
        ck=torch.load(dest,weights_only=False)
        if ck['bindings']!=bindings:raise ValueError('Changed resume dependencies')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
    print('Starting',args.fold,args.arm,'after epoch',ck['epoch'],flush=True)
    for epoch in range(ck['epoch']+1,args.through_epoch+1):
        schedule=epoch_plan(pools,weights,rng);loss_sum=0.;steps=0;empty=0
        for i,(g,key) in enumerate(schedule):
            try:e=supported_example(stores[g][key],scaler)
            except ValueError as exc:
                if str(exc)!='No source-supported independent history remains':raise
                empty+=1;continue
            x=e['tokens']
            if args.arm=='process':x=x.clone();x[:,:len(variables)]=0
            optimizer.zero_grad(set_to_none=True)
            pred=model(x,e['event_hours'],e['anchor_hour'],variables[key[1]])
            loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            steps+=1;loss_sum+=float(loss.detach())
            if (i+1)%5000==0:print('Progress',args.fold,args.arm,epoch,i+1,len(schedule),flush=True)
        ck['log'].append(dict(epoch=epoch,scheduled=len(schedule),updates=steps,empty_history=empty,fit_draw_loss=loss_sum/steps))
        ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        tmp=dest.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(dest)
        print('Completed epoch',args.fold,args.arm,ck['log'][-1],flush=True)
    report=Path(f'artifacts/native_reconstruction_encoder_crossfit_fold{args.fold}_{args.arm}_20260920.json')
    report.write_text(json.dumps(dict(fold=args.fold,arm=args.arm,epoch=ck['epoch'],planned_epochs=8,
        arm_training_complete=ck['epoch']==8,full_experiment_complete=False,log=ck['log'],bindings=bindings,
        private_sha256=sha256(dest),runtime_modified=False,promotion_allowed=False,
        limitation='Training progress only, no outer efficacy evaluated. Full protocol is five folds/two arms/eight fixed epochs. Fresh parameters, fold-local inputs; architectural design and cohort remain reused development.'),indent=2)+'\n')


if __name__=='__main__':main()
