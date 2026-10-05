"""One reconstruction state trained with equal exposure to both source groups."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_matched_broad_reconstruction import fixed_schedule
from scripts.train_native_reconstruction_expanded import placebo_labels
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def equal_group_median(groups):
    values=np.concatenate(groups)
    weights=np.concatenate([np.full(len(g),1/len(g)/len(groups)) for g in groups])
    order=np.argsort(values,kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(weights[order]),.5)])


def main():
    out=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    if out.exists():raise ValueError('Training already complete')
    rp=Path('artifacts/native_reconstruction_joint_source_manifest_20260917.json')
    report=json.loads(rp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Manifest changed')
    ip=Path('data/research_recovery/broad_source_20260917/joint_source_manifest.pt')
    index=torch.load(ip,weights_only=False);variables=index['variables'];targets=index['targets']
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    data={g:{r:{} for r in ('fit','encoder_tune')}for g in ('selected','broad')}
    for shard in index['shards']:
        p=Path(shard['path'])
        if sha256(p)!=shard['sha256']:raise ValueError('Sample changed')
        s=torch.load(p,weights_only=False)
        for c,e in s['samples'].items():
            if variables[c] in targets:data[shard['group']][shard['role']][(s['patient'],c)]=e
    columns=[variables.index(t)for t in targets]
    groups={g:{c:[k for k in d['fit']if k[1]==c]for c in columns}for g,d in data.items()}
    medians={c:equal_group_median([np.array([data[g]['fit'][k]['truth_normalized']for k in groups[g][c]])for g in data])for c in columns}
    assets=deepcopy(load_release(Path('models')))
    if assets.checkpoint['variables']!=variables:raise ValueError('Schema mismatch')
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,v in medians.items():initial.readout.bias[c]=v
    protocol=Path('RECONSTRUCTION_JOINT_SOURCE_PROTOCOL_20260917.md')
    dependencies=[Path(__file__),rp,ip,protocol,Path('models/whole_body_joint_jepa.pt'),
        Path('scripts/train_matched_broad_reconstruction.py'),Path('scripts/train_native_reconstruction_expanded.py'),
        Path('scripts/train_reconstruction_replication_v2.py'),Path('whole_body_jepa/native_body_reconstruction.py'),
        Path('whole_body_jepa/native_organ_state_residual.py'),Path('whole_body_jepa/history_event_input.py')]
    bindings={str(p):sha256(p)for p in dependencies}
    private=Path('data/research_recovery/broad_source_20260917/joint_training');private.mkdir(exist_ok=True)
    shuffled={g:placebo_labels(groups[g],data[g]['fit'],np.random.default_rng(20260913))for g in data}
    def score(model,arm):
        result={}
        with torch.inference_mode():
            for g,d in data.items():
                errors={c:[]for c in columns}
                for k,e in d['encoder_tune'].items():
                    p=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                    errors[k[1]].append(abs(p-e['truth_normalized']))
                result[g]={variables[c]:float(np.mean(v))for c,v in errors.items()}
        return float(np.mean([v for r in result.values()for v in r.values()])),result
    summary={}
    for arm in ('real','process','placebo'):
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);rng=np.random.default_rng(19)
        path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings:raise ValueError('Resume mismatch')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        else:
            value,per=score(model,arm)
            ck=dict(epoch=0,selected_epoch=0,best_tune=value,best_state_dict=deepcopy(model.state_dict()),initial_scores=per,log=[],bindings=bindings)
        for epoch in range(ck['epoch']+1,9):
            schedules={g:fixed_schedule(groups[g],rng)for g in data}
            plan=[(g,schedules[g][i])for i in range(17688)for g in rng.permutation(list(data))]
            loss_sum=0.
            for step,(g,k) in enumerate(plan):
                e=data[g]['fit'][k];optimizer.zero_grad(set_to_none=True)
                pred=model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])
                y=shuffled[g][k]if arm=='placebo'else e['truth_normalized']
                loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([y]))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())
            value,per=score(model,arm)
            ck['log'].append(dict(epoch=epoch,tune_mae=value,group_target_scores=per,updates=len(plan),draw_loss=loss_sum/len(plan)))
            if value<ck['best_tune']:ck.update(best_tune=value,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),variables=variables,targets=targets,medians=medians)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,bindings=bindings),private/f'{arm}_epoch{epoch}.pt')
            temp=path.with_suffix('.tmp');torch.save(ck,temp);temp.replace(path)
            print(arm,epoch,value,'selected',ck['selected_epoch'],flush=True)
        summary[arm]={k:ck[k]for k in ('epoch','selected_epoch','best_tune','initial_scores','log')}
    out.write_text(json.dumps(dict(training_complete=True,arms=summary,bindings=bindings,source_targets=targets,
        support=report['support'],runtime_modified=False,promotion_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
