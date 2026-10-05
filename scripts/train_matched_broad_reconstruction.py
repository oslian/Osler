"""Matched source breadth, fixed per-target exposure and independent selection."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_native_reconstruction_expanded import placebo_labels
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def fixed_schedule(groups,rng,draws=536):
    if not groups or any(not keys or len(keys)>draws for keys in groups.values()):raise ValueError('Invalid exposure support')
    targets=sorted(groups);shuffled={c:[groups[c][i] for i in rng.permutation(len(groups[c]))] for c in targets}
    schedule=[shuffled[c][j%len(shuffled[c])] for j in range(draws) for c in rng.permutation(targets)]
    if set(schedule)!=set(k for g in groups.values() for k in g):raise ValueError('Incomplete exposure')
    return schedule


def main():
    out=Path('artifacts/native_reconstruction_matched_broad_training_20260917.json')
    if out.exists():raise ValueError('Completed training exists')
    source=Path('artifacts/native_reconstruction_matched_broad_samples_20260917.json')
    oldp=Path('artifacts/native_reconstruction_nested_training_20260917.json')
    sample=json.loads(source.read_text());old=json.loads(oldp.read_text())
    for report in (sample,old):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    ip=Path('data/research_recovery/broad_source_20260917/matched_samples/index.pt')
    index=torch.load(ip,weights_only=False)
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    if variables!=index['variables']:raise ValueError('Schema mismatch')
    targets=[variables.index(t) for t in old['source_targets']];data={r:{} for r in ('fit','encoder_tune')};people={r:set() for r in data}
    for shard in index['shards']:
        p=Path(shard['path'])
        if sha256(p)!=shard['sha256']:raise ValueError('Changed shard')
        s=torch.load(p,weights_only=False);role=shard['role']
        if s['patient']!=shard['patient'] or s['role']!=role:raise ValueError('Identity mismatch')
        people[role].add(s['patient'])
        for c,e in s['samples'].items():
            if c in targets:data[role][(s['patient'],c)]=e
    if people['fit']&people['encoder_tune'] or {r:len(v) for r,v in people.items()}!={'fit':536,'encoder_tune':90}:raise ValueError('Patient split mismatch')
    for role,rows in data.items():
        for c in targets:
            count=sum(k[1]==c for k in rows)
            if count!=sample['accepted'][role][variables[c]] or count==0:raise ValueError('Target support mismatch')
    groups={c:[k for k in data['fit'] if k[1]==c] for c in targets}
    medians={c:float(np.median([data['fit'][k]['truth_normalized'] for k in keys])) for c,keys in groups.items()}
    paths=[Path(__file__),source,oldp,ip,Path('RECONSTRUCTION_MATCHED_BROAD_TRAINING_PROTOCOL_20260917.md'),Path('scripts/train_native_reconstruction_expanded.py'),Path('scripts/train_reconstruction_replication_v2.py'),Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/native_organ_state_residual.py'),Path('whole_body_jepa/native_organ_observations.py'),Path('whole_body_jepa/physiology_state.py'),Path('models/whole_body_joint_jepa.pt')]
    bindings={str(p):sha256(p) for p in paths}
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,m in medians.items():initial.readout.bias[c]=m
    def score(model,arm):
        errors={c:[] for c in targets}
        with torch.no_grad():
            for k,e in data['encoder_tune'].items():
                pred=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                errors[k[1]].append(abs(pred-e['truth_normalized']))
        per={variables[c]:float(np.mean(v)) for c,v in errors.items()}
        return float(np.mean(list(per.values()))),per
    private=Path('data/research_recovery/broad_source_20260917/matched_training');private.mkdir(exist_ok=True)
    shuffled=placebo_labels(groups,data['fit'],np.random.default_rng(20260913));summary={}
    for arm in ('real','process','placebo'):
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);rng=np.random.default_rng(19);path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings:raise ValueError('Resume binding mismatch')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        else:
            best,per=score(model,arm)
            ck=dict(epoch=0,selected_epoch=0,best_tune=best,initial_tune=best,initial_target_scores=per,best_state_dict=deepcopy(model.state_dict()),log=[],bindings=bindings)
        for epoch in range(ck['epoch']+1,9):
            schedule=fixed_schedule(groups,rng);loss_sum=0.
            for step,k in enumerate(schedule):
                e=data['fit'][k];optimizer.zero_grad(set_to_none=True)
                pred=model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])
                label=shuffled[k] if arm=='placebo' else e['truth_normalized']
                loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([label]))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())
                if (step+1)%5000==0:print(arm,epoch,'updates',step+1,flush=True)
            value,per=score(model,arm)
            ck['log'].append(dict(epoch=epoch,tune_mae=value,target_scores=per,draw_loss=loss_sum/len(schedule),updates=len(schedule),unique_fit_pairs=len(set(schedule))))
            if value<ck['best_tune']:ck.update(best_tune=value,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),variables=variables,targets=targets)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,bindings=bindings),private/f'{arm}_epoch{epoch}.pt')
            temp=path.with_suffix('.tmp');torch.save(ck,temp);temp.replace(path)
            print(arm,epoch,'encoder_tune',value,'selected',ck['selected_epoch'],flush=True)
        summary[arm]={k:ck[k] for k in ('epoch','selected_epoch','initial_tune','best_tune','log')}
    out.write_text(json.dumps(dict(training_complete=True,arms=summary,source_targets=old['source_targets'],source_unsupported=old['source_unsupported'],support=sample['accepted'],bindings=bindings,runtime_modified=False,promotion_allowed=False),indent=2)+'\n')
    print('Matched broad training complete',flush=True)


if __name__=='__main__':main()
