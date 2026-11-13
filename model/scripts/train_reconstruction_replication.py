"""Repeated patient-split reconstruction experiment; no old role reuse."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.multisource_intervals import sha256


import pandas as pd
from whole_body_jepa.reconstruction_partitions import patient_roles
from scripts.train_native_reconstruction_expanded import balanced_schedule, placebo_labels
from scripts.audit_expanded_reconstruction_numeric_path import neutralize_numbers


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--through-epoch',type=int,default=8)
    parser.add_argument('--seed',type=int,choices=(19,31,43,59,71,83),required=True)
    args=parser.parse_args()
    if not 1<=args.through_epoch<=8:raise ValueError('fixed eight-epoch protocol')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    source=Path('artifacts/native_reconstruction_expanded_20260913.json')
    report=json.loads(source.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('source dependency changed')
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    targets=[variables.index(v) for v in report['source_eligible']]
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
    partition_path=root/f'partition_{args.seed}.npz'
    roles=patient_roles(pd.read_parquet(root/'queries.parquet'),dict(np.load(partition_path)))
    data={r:{} for r in ('fit','tune')}
    for shard in index['shards']:
        role=roles[shard['patient']]
        if role not in data:continue
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(shard['path'],weights_only=False)
        if s['patient']!=shard['patient']:raise ValueError('patient identity mismatch')
        for c,e in s['samples'].items():
            if c in targets:data[role][(s['patient'],c)]=e
    unsupported=[c for c in targets if any(sum(k[1]==c for k in data[r])<30 for r in data)]
    targets=[c for c in targets if c not in unsupported]
    data={r:{k:e for k,e in d.items() if k[1] in targets} for r,d in data.items()}
    groups={c:[k for k in data['fit'] if k[1]==c] for c in targets}
    medians={c:float(np.median([data['fit'][k]['truth_normalized'] for k in groups[c]])) for c in targets}
    private=root/f'reconstruction_replication_{args.seed}_20260914';private.mkdir(exist_ok=True)
    dependencies=[Path(__file__),source,Path('NATIVE_RECONSTRUCTION_REPLICATION_PROTOCOL_20260914.md'),partition_path,
                  Path('whole_body_jepa/reconstruction_partitions.py'),Path('scripts/train_native_reconstruction_expanded.py'),
                  Path('scripts/audit_expanded_reconstruction_numeric_path.py'),
                  Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/native_organ_state_residual.py'),
                  Path('models/whole_body_joint_jepa.pt')]
    bindings={str(p):sha256(p) for p in dependencies}
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,m in medians.items():initial.readout.bias[c]=m
    def tune_score(model):
        errors={c:[] for c in targets}
        with torch.no_grad():
            for k,e in data['tune'].items():
                prediction=float(model((neutralize_numbers(e['tokens'],len(variables)) if arm=='process' else e['tokens']),e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                errors[k[1]].append(abs(prediction-e['truth_normalized']))
        return float(np.mean([np.mean(v) for v in errors.values()]))
    shuffled=placebo_labels(groups,data['fit'],np.random.default_rng(20260913))
    for arm in ('real','process','placebo'):
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        rng=np.random.default_rng(args.seed);path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings:raise ValueError('resume contract changed')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        else:
            best=tune_score(model)
            ck=dict(epoch=0,selected_epoch=0,best_tune=best,initial_tune=best,
                    best_state_dict=deepcopy(model.state_dict()),log=[],bindings=bindings)
        for epoch in range(ck['epoch']+1,args.through_epoch+1):
            schedule=balanced_schedule(groups,rng);loss_sum=0.
            for step,key in enumerate(schedule):
                e=data['fit'][key];optimizer.zero_grad(set_to_none=True)
                prediction=model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[key[1]])
                label=e['truth_normalized'] if arm!='placebo' else shuffled[key]
                loss=torch.nn.functional.smooth_l1_loss(prediction,prediction.new_tensor([label]))
                if not torch.isfinite(loss):raise ValueError('nonfinite objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach())
                if (step+1)%5000==0:print(arm,epoch,'updates',step+1,flush=True)
            score=tune_score(model)
            ck['log'].append(dict(epoch=epoch,tune_mae=score,draw_loss=loss_sum/len(schedule),
                                  updates=len(schedule),unique_fit_pairs=len(set(schedule))))
            if score<ck['best_tune']:
                ck.update(best_tune=score,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                      rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),targets=targets,variables=variables)
            temporary=path.with_suffix('.tmp');torch.save(ck,temporary);temporary.replace(path)
            print(arm,epoch,'tune',score,flush=True)
    summary=dict(seed=args.seed,source_unsupported=[variables[c] for c in unsupported],training_complete=args.through_epoch==8,through_epoch=args.through_epoch,
                 source_targets=[variables[c] for c in targets],runtime_modified=False,promotion_allowed=False,arms={},bindings=bindings)
    for arm in ('real','process','placebo'):
        ck=torch.load(private/f'{arm}.pt',weights_only=False)
        summary['arms'][arm]={k:ck[k] for k in ('epoch','selected_epoch','initial_tune','best_tune','log')}
    out=Path(f'artifacts/native_reconstruction_replication_{args.seed}_epoch{args.through_epoch}_20260914.json')
    if out.exists():raise ValueError('report already exists')
    out.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
