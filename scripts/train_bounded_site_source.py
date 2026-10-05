"""Matched real-input source training: uniform versus bounded site sampling."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.bounded_site_sampling import bounded_weights, sample_indices
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def replacement_plan(groups, weights, rng, draws=536):
    schedules={}
    for group,targets in groups.items():
        selected={c:[keys[i] for i in sample_indices(weights[group][c],rng,draws)] for c,keys in targets.items()}
        schedules[group]=[selected[c][i] for i in range(draws) for c in rng.permutation(sorted(targets))]
    if len({len(v) for v in schedules.values()})!=1:
        raise ValueError('Unequal group budget')
    return [(g,schedules[g][i]) for i in range(len(next(iter(schedules.values())))) for g in rng.permutation(list(groups))]


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    out=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    if out.exists():raise ValueError('Completed training exists')
    pp=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    ip=root/'site_balance_manifest_20260920.pt'
    preflight=json.loads(pp.read_text())
    if sha256(ip)!=preflight['private_sha256']:raise ValueError('Site mapping changed')
    site_manifest=torch.load(ip,weights_only=False)
    for p,h in site_manifest['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError(f'Source provenance changed: {p}')
    jp,fp,sp=root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (jp,fp,sp)]
    variables,targets=joint['variables'],joint['targets'];columns=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    data={g:{r:SourceStore([s for s in shards if s['group']==g and s['role']==r],columns) for r in ('fit','encoder_tune')} for g in ('selected','broad')}
    groups={g:d['fit'].groups for g,d in data.items()}
    cp=root/'source_exposure_pair/expanded/real.pt'
    parent=torch.load(cp,weights_only=False);medians=parent['medians']
    if parent['variables']!=variables or parent['targets']!=targets:raise ValueError('Parent schema changed')
    paths=[Path(__file__),pp,ip,jp,fp,sp,cp,Path('RECONSTRUCTION_BOUNDED_SITE_PROTOCOL_20260920.md'),Path('scripts/bounded_site_sampling.py'),Path('scripts/reconstruction_source_store.py'),Path('scripts/train_reconstruction_replication_v2.py'),Path('models/whole_body_joint_jepa.pt')]
    paths+=list(Path('whole_body_jepa').glob('*.py'))
    bindings={str(p):sha256(p) for p in paths}
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,v in medians.items():initial.readout.bias[c]=v

    def score(model):
        per={}
        with torch.inference_mode():
            for g,d in data.items():
                errors={c:[] for c in columns}
                for key,truth in d['encoder_tune'].labels.items():
                    e=d['encoder_tune'][key]
                    pred=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],variables[key[1]])[0])
                    errors[key[1]].append(abs(pred-truth))
                per[g]={variables[c]:float(np.mean(v)) for c,v in errors.items()}
        value=float(np.mean([v for group in per.values() for v in group.values()]))
        if not np.isfinite(value):raise ValueError('Nonfinite selection')
        return value,per

    private=root/'bounded_site_training';private.mkdir(exist_ok=True);summary={}
    for arm in ('uniform','bounded'):
        weights={g:{c:(np.ones(len(keys)) if arm=='uniform' else bounded_weights([site_manifest['sample_sites'][g,'fit',k[0],c] for k in keys])) for c,keys in pool.items()} for g,pool in groups.items()}
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        rng=np.random.default_rng(19);torch.manual_seed(19)
        seen={g:{c:set() for c in columns} for g in groups};path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings:raise ValueError('Resume changed')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng']);seen=ck['seen']
        else:
            value,per=score(model)
            for g in per:
                for t,v in per[g].items():
                    if not np.isclose(v,parent['initial_scores'][g][t],rtol=0,atol=1e-7):raise ValueError('Initial score mismatch')
            ck=dict(epoch=0,selected_epoch=0,best_tune=value,best_state_dict=deepcopy(model.state_dict()),initial_scores=per,log=[],bindings=bindings)
        print(arm,'start after',ck['epoch'],flush=True)
        for epoch in range(ck['epoch']+1,9):
            plan=replacement_plan(groups,weights,rng);loss_sum=0.
            for group,key in plan:
                e=data[group]['fit'][key];seen[group][key[1]].add(key[0]);optimizer.zero_grad(set_to_none=True)
                pred=model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],variables[key[1]])
                loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())
            value,per=score(model)
            exposure={g:{variables[c]:len(ids) for c,ids in bytarget.items()} for g,bytarget in seen.items()}
            ck['log'].append(dict(epoch=epoch,tune_mae=value,group_target_scores=per,updates=len(plan),draw_loss=loss_sum/len(plan),unique_exposure=exposure))
            if value<ck['best_tune']:ck.update(best_tune=value,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),seen=seen,variables=variables,targets=targets,medians=medians)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,bindings=bindings),private/f'{arm}_epoch{epoch}.pt')
            temp=path.with_suffix('.tmp');torch.save(ck,temp);temp.replace(path)
            print(arm,epoch,value,'selected',ck['selected_epoch'],flush=True)
        summary[arm]={k:ck[k] for k in ('epoch','selected_epoch','best_tune','initial_scores','log')}
    out.write_text(json.dumps(dict(training_complete=True,arms=summary,bindings=bindings,source_targets=targets,runtime_modified=False,promotion_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
