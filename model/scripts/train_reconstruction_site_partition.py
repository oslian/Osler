"""Resumable matched reconstruction on the frozen history-safe site split."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable
from scripts.train_native_reconstruction_expanded import balanced_schedule, placebo_labels
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--through-epoch',type=int,required=True,choices=range(1,9))
    args=parser.parse_args()
    report_path=Path('artifacts/native_reconstruction_site_partition_20260917.json')
    source=json.loads(report_path.read_text())
    for p,h in source['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed partition dependency')
    private=Path('data/research_recovery/site_reconstruction_training_20260917')
    private.mkdir(exist_ok=True)
    out=Path(f'artifacts/native_reconstruction_site_training_epoch{args.through_epoch}_20260917.json')
    if out.exists():raise ValueError('existing report')
    split_path=Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    roles=torch.load(split_path,weights_only=False)['roles']
    root=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    index_path=root/'expanded_reconstruction_20260913/index.pt'
    index=torch.load(index_path,weights_only=False)
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    targets=[variables.index(t) for t in source['source_eligible']]
    data={r:{} for r in ('fit','tune')}
    for shard in index['shards']:
        role=roles[shard['patient']]
        if role not in data:continue
        path=durable(shard['path'])
        if sha256(path)!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(path,weights_only=False)
        if s['patient']!=shard['patient']:raise ValueError('wrong patient')
        for c,e in s['samples'].items():
            if c in targets:data[role][(s['patient'],c)]=e
    for role,rows in data.items():
        for c in targets:
            if sum(k[1]==c for k in rows)!=source['target_counts'][variables[c]][role]:
                raise ValueError('support changed')
    groups={c:[k for k in data['fit'] if k[1]==c] for c in targets}
    medians={c:float(np.median([data['fit'][k]['truth_normalized'] for k in keys])) for c,keys in groups.items()}
    dependencies=[Path(__file__),report_path,split_path,index_path,
        Path('RECONSTRUCTION_SITE_TRAINING_PROTOCOL_20260917.md'),
        Path('scripts/train_native_reconstruction_expanded.py'),
        Path('scripts/train_reconstruction_replication_v2.py'),
        Path('scripts/audit_expanded_reconstruction_numeric_path.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),
        Path('whole_body_jepa/native_organ_state_residual.py'),
        Path('models/whole_body_joint_jepa.pt')]
    bindings={str(p):sha256(p) for p in dependencies}
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,m in medians.items():initial.readout.bias[c]=m
    def tune_score(model,arm):
        errors={c:[] for c in targets}
        with torch.no_grad():
            for k,e in data['tune'].items():
                p=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                errors[k[1]].append(abs(p-e['truth_normalized']))
        return float(np.mean([np.mean(v) for v in errors.values()]))
    shuffled=placebo_labels(groups,data['fit'],np.random.default_rng(20260913))
    summary={}
    for arm in ('real','process','placebo'):
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        rng=np.random.default_rng(19);path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings or ck['epoch']>args.through_epoch:raise ValueError('resume mismatch')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        else:
            best=tune_score(model,arm)
            ck=dict(epoch=0,selected_epoch=0,best_tune=best,initial_tune=best,
                    best_state_dict=deepcopy(model.state_dict()),log=[],bindings=bindings)
        for epoch in range(ck['epoch']+1,args.through_epoch+1):
            schedule=balanced_schedule(groups,rng);loss_sum=0.
            for step,k in enumerate(schedule):
                e=data['fit'][k];optimizer.zero_grad(set_to_none=True)
                p=model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])
                label=shuffled[k] if arm=='placebo' else e['truth_normalized']
                loss=torch.nn.functional.smooth_l1_loss(p,p.new_tensor([label]))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach())
                if (step+1)%5000==0:print(arm,epoch,'updates',step+1,flush=True)
            score=tune_score(model,arm)
            ck['log'].append(dict(epoch=epoch,tune_mae=score,draw_loss=loss_sum/len(schedule),
                updates=len(schedule),unique_fit_pairs=len(set(schedule))))
            if score<ck['best_tune']:ck.update(best_tune=score,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),variables=variables,targets=targets)
            tmp=path.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(path)
            print(arm,epoch,'tune',score,flush=True)
        summary[arm]={k:ck[k] for k in ('epoch','selected_epoch','initial_tune','best_tune','log')}
    out.write_text(json.dumps(dict(training_complete=args.through_epoch==8,through_epoch=args.through_epoch,
        arms=summary,bindings=bindings,source_targets=source['source_eligible'],
        source_unsupported=[t for t in source['target_counts'] if t not in source['source_eligible']],
        runtime_modified=False,promotion_allowed=False),indent=2)+'\n')
    print('complete segment',out,flush=True)


if __name__=='__main__':main()
