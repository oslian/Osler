"""Same-capacity frozen-state SmoothL1 readout refit; source-only decisions."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from scripts.audit_confirmed_reconstruction_heads import evidence,confirmed
from scripts.bounded_site_sampling import bounded_weights
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def objective(theta,x,y,w):
    residual=x@theta[:-1]+theta[-1]-y
    psi=np.clip(residual,-1,1)*w
    return float(w@smooth_l1_from_error(abs(residual))),np.r_[x.T@psi,psi.sum()]


def solve(x,y,w,initial_weight,initial_bias):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);w=np.asarray(w,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or (w<=0).any():
        raise ValueError('Invalid fitting arrays')
    w=w/w.sum();mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1
    z=(x-mean)/scale;theta=np.r_[initial_weight*scale,initial_bias+mean@initial_weight]
    before=objective(theta,z,y,w)[0]
    fitted=minimize(objective,theta,args=(z,y,w),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=50))
    weight=fitted.x[:-1]/scale;bias=float(fitted.x[-1]-mean@weight)
    after=objective(fitted.x,z,y,w)[0]
    if after>before+1e-10:raise ValueError('Loss increased')
    return weight,bias,dict(success=bool(fitted.success),message=str(fitted.message),iterations=int(fitted.nit),
        loss_before=before,loss_after=after,gradient_max=float(abs(fitted.jac).max()))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    rp=Path('artifacts/native_reconstruction_bias_polish_20260920.json')
    private=root/'frozen_readout_refit_20260920';private.mkdir(exist_ok=True)
    out=Path('artifacts/native_reconstruction_frozen_readout_refit_20260920.json')
    if out.exists():raise ValueError('Completed experiment')
    previous=json.loads(rp.read_text())
    for p,h in previous['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    bindings={str(p):sha256(p) for p in (Path(__file__),cp,mp,jp,rp)};bp=private/'bindings.json'
    if bp.exists() and json.loads(bp.read_text())!=bindings:raise ValueError('Resume mismatch')
    if not bp.exists():bp.write_text(json.dumps(bindings,indent=2)+'\n')
    ck,manifest,joint=[torch.load(p,weights_only=False) for p in (cp,mp,jp)]
    variables,targets=joint['variables'],joint['targets'];cols={variables.index(t) for t in targets}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    captured=[];hook=model.readout.register_forward_pre_hook(lambda m,args:captured.append(args[0][0].detach().cpu().numpy().copy()))
    grouped={role:{g:{t:[] for t in targets} for g in ('selected','broad')} for role in ('fit','encoder_tune')}
    seen={role:set() for role in grouped};counter={role:0 for role in grouped}
    try:
        with torch.inference_mode():
            for i,s in enumerate(manifest['shards']):
                role=s['role']
                if role not in grouped:continue
                if s['patient'] in seen[role]:raise ValueError('Duplicate patient')
                seen[role].add(s['patient']);p=Path(s['path']);save=private/f'features_{i:05d}.pt'
                if sha256(p)!=s['sha256']:raise ValueError('Changed sample')
                oldpath=root/('bias_polish_20260920' if role=='fit' else 'bias_polish_selection_20260920')/f'{"fit" if role=="fit" else "patient"}_{counter[role]:05d}.pt'
                prior=torch.load(oldpath,weights_only=False)
                if prior['shard']!=s:raise ValueError('Prior cache identity')
                refs={r['target']:r for r in prior['rows']}
                if save.exists():
                    saved=torch.load(save,weights_only=False)
                    if saved['shard']!=s:raise ValueError('Feature cache identity')
                    rows=saved['rows']
                else:
                    data=torch.load(p,weights_only=False)
                    if data['patient']!=s['patient']:raise ValueError('Raw identity')
                    rows=[]
                    for col,e in data['samples'].items():
                        if col not in cols:continue
                        target=variables[col];captured.clear()
                        pred=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],target)[0])
                        if len(captured)!=1:raise ValueError('Hook mismatch')
                        rows.append(dict(patient=s['patient'],target=target,body=captured[0],prediction=pred,truth=float(e['truth_normalized']),
                            site=manifest['sample_sites'][s['group'],role,s['patient'],col]))
                    temp=save.with_suffix('.tmp');torch.save(dict(shard=s,rows=rows),temp);temp.replace(save)
                if set(refs)!={r['target'] for r in rows}:raise ValueError('Pair mismatch')
                for r in rows:
                    ref=refs[r['target']]
                    if r['truth']!=ref['truth'] or not np.isclose(r['prediction'],ref['prediction'],atol=1e-7,rtol=0):raise ValueError('Prediction mismatch')
                    grouped[role][s['group']][r['target']].append(r)
                counter[role]+=1
                if (i+1)%500==0:print('State extraction',i+1,'/',len(manifest['shards']),flush=True)
    finally:hook.remove()
    if seen['fit']&seen['encoder_tune']:raise ValueError('Role overlap')
    state=deepcopy(ck['best_state_dict']);solvers={}
    with threadpool_limits(limits=4):
        for target in targets:
            parts=[grouped['fit'][g][target] for g in ('selected','broad')]
            x=np.stack([r['body'] for part in parts for r in part]);y=np.array([r['truth'] for part in parts for r in part]);weights=[]
            for part in parts:
                w=bounded_weights([r['site'] for r in part]);weights.extend(w/w.sum()/2)
            col=variables.index(target)
            weight,bias,solver=solve(x,y,weights,state['readout.weight'][col].numpy(),float(state['readout.bias'][col]))
            solvers[target]=solver
            if not solver['success']:raise ValueError(f'Unconverged readout: {target}, {solver}')
            state['readout.weight'][col]=torch.as_tensor(weight,dtype=state['readout.weight'].dtype)
            state['readout.bias'][col]=bias
            print('Solved',target,solver['iterations'],flush=True)
    candidate=private/'candidate.pt';torch.save(dict(state_dict=state,solvers=solvers,bindings=bindings,promotion_allowed=False),candidate)
    results={}
    for target in targets:
        col=variables.index(target);groups={}
        for g in ('selected','broad'):
            rows=grouped['encoder_tune'][g][target];x=np.stack([r['body'] for r in rows]);y=np.array([r['truth'] for r in rows]);original=np.array([r['prediction'] for r in rows])
            prediction=x@state['readout.weight'][col].numpy()+float(state['readout.bias'][col]);delta=abs(prediction-y)-abs(original-y)
            check=evidence(delta,[r['site'] for r in rows]);groups[g]=dict(patients=len(rows),before=float(abs(original-y).mean()),after=float(abs(prediction-y).mean()),comparison=check,pass_incremental=confirmed({'base':check},len(rows)))
        results[target]=groups
    out.write_text(json.dumps(dict(solvers=solvers,source_selection=results,bindings=bindings,candidate_sha256=sha256(candidate),runtime_modified=False,promotion_allowed=False,
        limitation='Same-capacity convex SmoothL1 data-risk refit of frozen current body features. No AdamW equivalence or global end-to-end convergence claim. Reused source-selection, not new external test. All33trained rows refit; six untrained unchanged.'),indent=2)+'\n')
    print('Source passes',{g:[t for t,v in results.items() if v[g]['pass_incremental']] for g in ('selected','broad')},flush=True)


if __name__=='__main__':main()
