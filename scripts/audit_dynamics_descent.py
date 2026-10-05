"""Disjoint source probes of a loss-preserving body-dynamics descent direction."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.audit_gradient_sample_size import split_prefixes
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def protected_direction(desired, gradients):
    desired=np.asarray(desired,dtype=float);a=np.asarray(gradients,dtype=float)
    norms=np.linalg.norm(a,axis=1);a=a[norms>1e-12]/norms[norms>1e-12,None]
    if not len(a):return desired,dict(success=True,minimum_dot=0.)
    q=a@a.T;b=a@desired
    result=minimize(lambda x:(.5*x@q@x+b@x,q@x+b),np.zeros(len(a)),jac=True,
                    bounds=[(0,None)]*len(a),method='L-BFGS-B',options=dict(maxiter=10000,ftol=1e-14,gtol=1e-10))
    direction=desired+a.T@result.x
    minimum=float((a@direction).min())
    return direction,dict(success=bool(result.success and minimum>=-1e-6),minimum_dot=minimum,message=str(result.message))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json');cp=root/'bounded_site_training/bounded.pt'
    paths=[root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')]
    out=Path('artifacts/native_reconstruction_dynamics_descent_20260920.json');private=root/'dynamics_descent_20260920.pt'
    if out.exists() or private.exists():raise ValueError('Existing diagnostic')
    report=json.loads(tp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    ck=torch.load(cp,weights_only=False);joint,full,split=[torch.load(p,weights_only=False) for p in paths]
    if ck['bindings']!=report['bindings']:raise ValueError('Changed checkpoint')
    variables,targets=joint['variables'],joint['targets'];cols=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    stores={g:SourceStore([s for s in shards if s['group']==g and s['role']=='fit'],cols) for g in ('selected','broad')}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    params=list(model.trajectory.layer.body_dynamics.parameters());rng=np.random.default_rng(20260920)
    tasks=[];probes=[[],[]];support={}
    for group,store in stores.items():
        for col in cols:
            task=f'{group}/{variables[col]}';keys=store.groups[col];orders=split_prefixes(len(keys),rng,64)
            tasks.append(task);support[task]=len(orders[0])
            for index,order in enumerate(orders):
                total=np.zeros(sum(p.numel() for p in params),dtype=float)
                for pos in order:
                    e=store[keys[pos]];pred=model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[col])
                    loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
                    grad=torch.autograd.grad(loss,params,allow_unused=True)
                    total+=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grad)]).double().numpy()/len(order)
                if not np.isfinite(total).all():raise ValueError('Invalid gradient')
                probes[index].append(total)
        print(group,'gradient probes complete',flush=True)
    arrays=[np.stack(p) for p in probes];results={}
    bun=[tasks.index(f'{g}/bun') for g in stores]
    for source in (0,1):
        desired=arrays[source][bun].mean(0);norm=np.linalg.norm(desired)
        if norm==0:raise ValueError('No BUN gradient')
        desired=desired/norm
        projected,solver=protected_direction(desired,arrays[source]);scores={}
        for name,d in (('bun_only',desired),('protected',projected)):
            scores[name]=dict(norm=float(np.linalg.norm(d)),
                source_dots={t:float(v) for t,v in zip(tasks,arrays[source]@d)},
                other_probe_dots={t:float(v) for t,v in zip(tasks,arrays[1-source]@d)})
        results[str(source)]=dict(solver=solver,directions=scores)
    torch.save(dict(tasks=tasks,gradients=arrays,support=support),private)
    out.write_text(json.dumps(dict(results=results,support=support,parameter_count=sum(p.numel() for p in params),
        bindings={str(p):sha256(p) for p in (Path(__file__),tp,cp,*paths)},private_sha256=sha256(private),
        runtime_modified=False,promotion_allowed=False,
        limitation='Up to64patients per task per disjoint probe; sparse tasks use fewer. Local body-dynamics first-order SmoothL1 geometry only, no actual step or Adam/generalization/promotion proof. Positive dot means predicted decrease under a negative step.'),indent=2)+'\n')
    for s,r in results.items():
        print(s,r['solver'],{n:dict(norm=v['norm'],other_harms=sum(x < -1e-8 for x in v['other_probe_dots'].values()),bun=[v['other_probe_dots'][tasks[i]] for i in bun]) for n,v in r['directions'].items()})


if __name__=='__main__':main()
