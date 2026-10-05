"""Fixed source-derived finite updates, evaluated on separate selection patients."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scripts.audit_dynamics_descent import protected_direction
from scripts.audit_continued_selection_pairs import interval
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from scripts.reconstruction_source_store import SourceStore,source_shards
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release

STEP=0.01


def make_directions(tasks,gradients):
    bun=[tasks.index(f'{g}/bun') for g in ('selected','broad')];directions={}
    for i,matrix in enumerate(gradients):
        desired=matrix[bun].mean(0);desired=desired/np.linalg.norm(desired)
        projected,solver=protected_direction(desired,matrix)
        if not solver['success']:raise ValueError('Projection failed')
        directions[f'probe{i}_protected']=projected
        directions[f'probe{i}_matched_bun']=desired*np.linalg.norm(projected)
    return directions


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    gp=root/'dynamics_descent_20260920.pt';gr=Path('artifacts/native_reconstruction_dynamics_descent_20260920.json')
    pp=root/'history_suffix_20260920.pt';pr=Path('artifacts/native_reconstruction_history_suffix_20260920.json')
    cp=root/'bounded_site_training/bounded.pt'
    paths=[root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    out=Path('artifacts/native_reconstruction_dynamics_finite_step_20260920.json');private=root/'dynamics_finite_step_20260920.pt'
    if out.exists() or private.exists():raise ValueError('Existing diagnostic')
    gradient_report,previous=[json.loads(p.read_text()) for p in (gr,pr)]
    for report in (gradient_report,previous):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(gp)!=gradient_report['private_sha256'] or sha256(pp)!=previous['private_sha256']:raise ValueError('Changed private data')
    gradient_data=torch.load(gp,weights_only=False);old=torch.load(pp,weights_only=False)
    directions=make_directions(gradient_data['tasks'],gradient_data['gradients'])
    joint,full,split=[torch.load(p,weights_only=False) for p in paths];ck=torch.load(cp,weights_only=False)
    variables,targets=joint['variables'],joint['targets'];cols=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    fit={s['patient'] for s in shards if s['role']=='fit'}
    if fit & {s['patient'] for s in shards if s['role']=='encoder_tune'}:raise ValueError('Role overlap')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    params=list(model.trajectory.layer.body_dynamics.parameters());base=parameters_to_vector(params).detach().clone()
    vectors={n:base-STEP*torch.as_tensor(d,dtype=base.dtype,device=base.device) for n,d in directions.items()}
    if any(v.shape!=base.shape for v in vectors.values()):raise ValueError('Parameter mismatch')
    raw,result={},{}
    with torch.inference_mode():
        for group in ('selected','broad'):
            store=SourceStore([s for s in shards if s['group']==group and s['role']=='encoder_tune'],cols)
            previous_rows={(r['patient'],variables.index(r['target'])):r for r in old[group]}
            if set(previous_rows)!=set(store.labels):raise ValueError('Pair mismatch')
            rows=[]
            for key,truth in store.labels.items():
                e=store[key];target=variables[key[1]];errors={}
                try:
                    for name,vector in [('base',base),*vectors.items()]:
                        vector_to_parameters(vector,params)
                        pred=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],target)[0])
                        errors[name]=abs(pred-truth)
                finally:vector_to_parameters(base,params)
                if not np.isclose(errors['base'],previous_rows[key]['errors']['full'],atol=1e-7,rtol=0):raise ValueError('Baseline mismatch')
                rows.append(dict(patient=key[0],target=target,errors=errors))
            raw[group]=rows;per={}
            for target in targets:
                chosen=[r for r in rows if r['target']==target];metrics={}
                for metric in ('mae','smooth_l1'):
                    errors={n:np.array([r['errors'][n] for r in chosen]) for n in ('base',*vectors)}
                    if metric=='smooth_l1':errors={n:smooth_l1_from_error(v) for n,v in errors.items()}
                    metrics[metric]={n:dict(mean=float(errors[n].mean()),versus_base=interval(errors[n]-errors['base'])) for n in vectors}
                per[target]=dict(patients=len(chosen),metrics=metrics)
            result[group]=per;print(group,len(rows),'complete',flush=True)
    torch.save(dict(rows=raw,directions=directions,step=STEP),private)
    out.write_text(json.dumps(dict(groups=result,step=STEP,direction_norms={n:float(np.linalg.norm(d)) for n,d in directions.items()},
        bindings={str(p):sha256(p) for p in (Path(__file__),gp,gr,pp,pr,cp,*paths)},private_sha256=sha256(private),
        runtime_modified=False,promotion_allowed=False,
        limitation='Fixed single step in body dynamics only, equal-L2 unprotected control. Source-selection patients separate from gradient fits, previously used to choose checkpoint. Unadjusted paired intervals, not new lockbox or full training/promotion.'),indent=2)+'\n')
    for g,per in result.items():
        print(g,{n:dict(mean_harms=sum(r['metrics']['smooth_l1'][n]['versus_base']['mean']>0 for r in per.values()),bun=per['bun']['metrics']['smooth_l1'][n]['versus_base']) for n in vectors})


if __name__=='__main__':main()
