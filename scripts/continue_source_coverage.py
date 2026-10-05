"""Continue all matched arms until every source target pool has been visited."""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_source_exposure_pair import epoch_plan, shuffled_labels
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def required_epochs(groups, draws=536):
    if draws<=0 or not groups or any(len(keys)==0 for keys in groups.values()):
        raise ValueError('Nonempty target pools and positive draw budget required')
    return math.ceil(max(map(len, groups.values()))/draws)


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    out=Path('artifacts/native_reconstruction_coverage_continuation_20260919.json')
    if out.exists(): raise ValueError('Completed continuation exists')
    old=json.loads(tp.read_text())
    for p,h in old['bindings'].items():
        if sha256(Path(p))!=h: raise ValueError(f'Original training changed: {p}')
    ip,ep=root/'joint_source_manifest.pt',root/'full_samples/index.pt'
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,ep,sp)]
    variables,targets=joint['variables'],joint['targets'];columns=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    data={g:{r:SourceStore([s for s in shards if s['group']==g and s['role']==r],columns)
             for r in ('fit','encoder_tune')} for g in ('selected','broad')}
    groups={g:d['fit'].groups for g,d in data.items()}
    epochs=required_epochs(groups['broad'])
    if epochs!=19: raise ValueError('Source-derived fixed budget changed')
    parents={a:root/f'source_exposure_pair/expanded/{a}.pt' for a in ('real','process','placebo')}
    paths=[Path(__file__),tp,ip,ep,sp,*parents.values(),
           Path('RECONSTRUCTION_COVERAGE_CONTINUATION_PROTOCOL_20260919.md'),
           Path('scripts/train_source_exposure_pair.py'),Path('scripts/reconstruction_source_store.py')]
    bindings={str(p):sha256(p) for p in paths}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6)
    assets.model.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    placebo={g:shuffled_labels(d['fit']) for g,d in data.items()}
    private=root/'coverage_continuation';private.mkdir(exist_ok=True)
    summary={}
    for arm,parent in parents.items():
        path=private/f'{arm}.pt'
        ck=torch.load(path if path.exists() else parent,weights_only=False)
        expected=bindings if path.exists() else old['bindings']
        if ck['bindings']!=expected or ck['epoch']<8: raise ValueError('Resume mismatch')
        model=deepcopy(template);model.load_state_dict(ck['state_dict'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.load_state_dict(ck['optimizer'])
        rng=np.random.default_rng();rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        queues,seen=ck['queues'],ck['seen']
        print(arm,'continue after',ck['epoch'],'through',epochs,flush=True)
        for epoch in range(ck['epoch']+1,epochs+1):
            plan=epoch_plan(groups,queues,rng);loss_sum=0.
            for group,key in plan:
                e=data[group]['fit'][key];seen[group][key[1]].add(key[0])
                optimizer.zero_grad(set_to_none=True)
                prediction=model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[key[1]])
                truth=placebo[group][key] if arm=='placebo' else e['truth_normalized']
                loss=torch.nn.functional.smooth_l1_loss(prediction,prediction.new_tensor([truth]))
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach())
            per={}
            with torch.inference_mode():
                for g,d in data.items():
                    errors={c:[] for c in columns}
                    for key,truth in d['encoder_tune'].labels.items():
                        e=d['encoder_tune'][key]
                        pred=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[key[1]])[0])
                        errors[key[1]].append(abs(pred-truth))
                    per[g]={variables[c]:float(np.mean(v)) for c,v in errors.items()}
            value=float(np.mean([v for group in per.values() for v in group.values()]))
            if not np.isfinite(value): raise ValueError('Nonfinite source score')
            for g in groups:
                for c,keys in groups[g].items():
                    if len(seen[g][c])!=min(len(keys),epoch*536): raise ValueError('Exposure mismatch')
            exposure={g:{variables[c]:len(v) for c,v in s.items()} for g,s in seen.items()}
            ck['log'].append(dict(epoch=epoch,tune_mae=value,group_target_scores=per,updates=len(plan),
                                 draw_loss=loss_sum/len(plan),unique_exposure=exposure))
            if value<ck['best_tune']:
                ck.update(best_tune=value,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                      rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),queues=queues,seen=seen,bindings=bindings)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,bindings=bindings),private/f'{arm}_epoch{epoch}.pt')
            temp=path.with_suffix('.tmp');torch.save(ck,temp);temp.replace(path)
            print(arm,epoch,value,'selected',ck['selected_epoch'],flush=True)
        summary[arm]={k:ck[k] for k in ('epoch','selected_epoch','best_tune','initial_scores','log')}
    out.write_text(json.dumps(dict(training_complete=True,arms=summary,bindings=bindings,source_targets=targets,
        total_epochs=epochs,runtime_modified=False,promotion_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
