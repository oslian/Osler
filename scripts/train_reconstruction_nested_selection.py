"""Fixed eight-epoch training with a genuinely separate head-selection cohort."""
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
    out=Path('artifacts/native_reconstruction_nested_training_20260917.json')
    if out.exists():raise ValueError('completed training exists')
    pp=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    oldp=Path('artifacts/native_reconstruction_site_training_epoch8_20260917.json')
    partition=json.loads(pp.read_text());old=json.loads(oldp.read_text())
    for report in (partition,old):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('changed dependency')
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    roles=torch.load(sp,weights_only=False)['roles']
    ip=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt')
    index=torch.load(ip,weights_only=False)
    torch.set_num_threads(4);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    targets=[variables.index(t) for t in old['source_targets']]
    data={r:{} for r in ('fit','encoder_tune')}
    for shard in index['shards']:
        role=roles[shard['patient']]
        if role not in data:continue
        p=durable(shard['path'])
        if sha256(p)!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(p,weights_only=False)
        if s['patient']!=shard['patient']:raise ValueError('wrong patient')
        for c,e in s['samples'].items():
            if c in targets:data[role][(s['patient'],c)]=e
    for role,rows in data.items():
        for c in targets:
            if sum(k[1]==c for k in rows)!=partition['target_counts'][variables[c]][role]:raise ValueError('support changed')
    groups={c:[k for k in data['fit'] if k[1]==c] for c in targets}
    medians={c:float(np.median([data['fit'][k]['truth_normalized'] for k in keys])) for c,keys in groups.items()}
    dependencies=[Path(__file__),pp,oldp,sp,ip,Path('RECONSTRUCTION_NESTED_TRAINING_PROTOCOL_20260917.md'),
        Path('scripts/train_native_reconstruction_expanded.py'),Path('scripts/train_reconstruction_replication_v2.py'),
        Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/native_organ_state_residual.py'),
        Path('whole_body_jepa/native_organ_observations.py'),Path('whole_body_jepa/physiology_state.py'),Path('models/whole_body_joint_jepa.pt')]
    bindings={str(p):sha256(p) for p in dependencies}
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    initial=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    with torch.no_grad():
        initial.readout.weight.zero_();initial.readout.bias.zero_()
        for c,m in medians.items():initial.readout.bias[c]=m
    def score(model,arm):
        errors={c:[] for c in targets}
        with torch.no_grad():
            for k,e in data['encoder_tune'].items():
                p=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                errors[k[1]].append(abs(p-e['truth_normalized']))
        per={variables[c]:float(np.mean(v)) for c,v in errors.items()}
        return float(np.mean(list(per.values()))),per
    private=Path('data/research_recovery/nested_reconstruction_training_20260917');private.mkdir(exist_ok=True)
    shuffled=placebo_labels(groups,data['fit'],np.random.default_rng(20260913));summary={}
    for arm in ('real','process','placebo'):
        model=deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        rng=np.random.default_rng(19);path=private/f'{arm}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False)
            if ck['bindings']!=bindings:raise ValueError('resume mismatch')
            model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
        else:
            best,per=score(model,arm)
            ck=dict(epoch=0,selected_epoch=0,best_tune=best,initial_tune=best,initial_target_scores=per,
                best_state_dict=deepcopy(model.state_dict()),log=[],bindings=bindings)
        for epoch in range(ck['epoch']+1,9):
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
            draw=loss_sum/len(schedule);previous=old['arms'][arm]['log'][epoch-1]
            if not np.isclose(draw,previous['draw_loss'],atol=1e-9,rtol=0):raise ValueError('optimizer trajectory changed')
            value,per=score(model,arm)
            ck['log'].append(dict(epoch=epoch,tune_mae=value,target_scores=per,draw_loss=draw,
                updates=len(schedule),unique_fit_pairs=len(set(schedule)),old_draw_loss_reproduced=True))
            if value<ck['best_tune']:ck.update(best_tune=value,selected_epoch=epoch,best_state_dict=deepcopy(model.state_dict()))
            ck.update(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),variables=variables,targets=targets)
            torch.save(dict(state_dict=model.state_dict(),epoch=epoch,bindings=bindings),private/f'{arm}_epoch{epoch}.pt')
            temp=path.with_suffix('.tmp');torch.save(ck,temp);temp.replace(path)
            print(arm,epoch,'encoder_tune',value,'selected',ck['selected_epoch'],flush=True)
        summary[arm]={k:ck[k] for k in ('epoch','selected_epoch','initial_tune','best_tune','log')}
    out.write_text(json.dumps(dict(training_complete=True,arms=summary,bindings=bindings,source_targets=old['source_targets'],
        source_unsupported=old['source_unsupported'],four_role_supported=partition['all_four_roles_supported'],
        runtime_modified=False,promotion_allowed=False),indent=2)+'\n')
    print('complete',out,flush=True)


if __name__=='__main__':main()
