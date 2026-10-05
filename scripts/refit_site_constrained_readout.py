"""Same-body readout repair with source-site non-regression constraints."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from scripts.refit_anchored_body_readout import anchored_objective, fit, source_weights
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.multisource_intervals import sha256


def losses(residual):
    a = abs(residual)
    return np.where(a < 1, .5*residual**2, a-.5)


def constraints(theta, design, y, masks, reference):
    residual = design @ theta-y
    loss = losses(residual)
    grad = np.clip(residual, -1, 1)[:, None]*design
    return (np.array([reference[i]-loss[m].mean() for i,m in enumerate(masks)]),
            np.stack([-grad[m].mean(axis=0) for m in masks]))


def fit_constrained(x, y, groups, sites, oldw, oldb):
    w = source_weights(groups, sites)
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale[scale<1e-8] = 1.
    z = (x-mean)/scale; design = np.c_[z, np.ones(len(y))]
    anchor = np.r_[oldw*scale, oldb+mean@oldw]
    masks = [(groups==g)&(sites==s) for g in sorted(set(groups)) for s in sorted(set(sites[groups==g]))]
    original = losses(design@anchor-y)
    reference = np.array([original[m].mean() for m in masks])
    result = minimize(anchored_objective, anchor, args=(z,y,w,anchor,1.), jac=True,
        method='SLSQP', constraints={'type':'ineq',
            'fun':lambda theta:constraints(theta,design,y,masks,reference)[0],
            'jac':lambda theta:constraints(theta,design,y,masks,reference)[1]},
        options=dict(maxiter=500, ftol=1e-10))
    slack, _ = constraints(result.x,design,y,masks,reference)
    accepted = bool(result.success and np.isfinite(result.x).all() and slack.min()>=-1e-8)
    theta = result.x if accepted else anchor
    weight = theta[:-1]/scale; bias = float(theta[-1]-mean@weight)
    return weight, bias, dict(accepted=accepted, message=str(result.message), iterations=int(result.nit),
        constraints=len(masks), min_slack=float(slack.min()),
        fit_loss_before=float(w@original), fit_loss_after=float(w@losses(design@theta-y)))


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    pp = Path('artifacts/native_reconstruction_anchored_readout_20260920.json')
    sp = Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    out = Path('artifacts/native_reconstruction_site_constrained_readout_20260920.json')
    if out.exists():raise ValueError('Existing experiment')
    parent = json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    mp = root/'site_balance_manifest_20260920.pt'; cp = root/'bounded_site_training/bounded.pt'
    ck = torch.load(cp,weights_only=False); manifest = torch.load(mp,weights_only=False)
    variables = ck['variables']; targets = list(parent['choices'])
    data = {r:{t:[] for t in targets} for r in ('fit','encoder_tune')}
    digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        path=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt'
        digest.update((path.name+sha256(path)).encode()); saved=torch.load(path,weights_only=False)
        if saved['shard']!=s:raise ValueError('Wrong feature identity')
        for row in saved['rows']:data[s['role']][row['target']].append(dict(row,group=s['group']))
    if digest.hexdigest()!=json.loads(sp.read_text())['cache_digest']:raise ValueError('Changed cache')
    bindings={str(p):sha256(p) for p in (Path(__file__), pp, sp, mp, cp, Path('scripts/refit_anchored_body_readout.py'), Path('RECONSTRUCTION_SITE_CONSTRAINED_PROTOCOL_20260920.md'))}
    private=root/'site_constrained_readout_20260920';private.mkdir(exist_ok=True)
    choices={}
    with threadpool_limits(limits=4):
        for target,rows in data['fit'].items():
            path=private/f'{target}.pt'
            if path.exists():
                saved=torch.load(path,weights_only=False)
                if saved['bindings']!=bindings:raise ValueError('Resume mismatch')
                choices[target]=saved;continue
            col=variables.index(target);oldw=ck['best_state_dict']['readout.weight'][col].numpy().astype(float);oldb=float(ck['best_state_dict']['readout.bias'][col])
            x=np.stack([r['body'] for r in rows]).astype(float);y=np.array([r['truth'] for r in rows])
            groups=np.array([r['group'] for r in rows]);sites=np.array([str(r['site']) for r in rows])
            cw,cb,info=fit_constrained(x,y,groups,sites,oldw,oldb)
            uw,ub=fit(x,y,source_weights(groups,sites),oldw,oldb,1.)
            saved=dict(constrained=(cw,cb),unconstrained=(uw,ub),solver=info,bindings=bindings)
            temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path);choices[target]=saved
            print(target,info,flush=True)
    # All fitted coefficients are frozen on disk before reading held-out errors.
    results={}
    for target,rows in data['encoder_tune'].items():
        per={}
        for group in ('selected','broad'):
            rr=[r for r in rows if r['group']==group]
            x=np.stack([r['body'] for r in rr]).astype(float);y=np.array([r['truth'] for r in rr]);old=np.array([r['prediction'] for r in rr]);sites=[r['site'] for r in rr]
            errors={'original':abs(old-y)}
            for arm in ('unconstrained','constrained'):
                w,b=choices[target][arm]
                pred=old.copy() if arm=='constrained' and not choices[target]['solver']['accepted'] else x@w+b
                errors[arm]=abs(pred-y)
            checks={arm:evidence(errors['constrained']-errors[arm],sites) for arm in ('original','unconstrained')}
            per[group]=dict(patients=len(rr),mae={k:float(v.mean()) for k,v in errors.items()},comparisons=checks,
                pass_incremental=confirmed(checks,len(rr)))
        results[target]=per
    summary={g:dict(mae={a:float(np.mean([v[g]['mae'][a] for v in results.values()])) for a in ('original','unconstrained','constrained')},
        passed=[t for t,v in results.items() if v[g]['pass_incremental']]) for g in ('selected','broad')}
    out.write_text(json.dumps(dict(targets=results,solvers={t:v['solver'] for t,v in choices.items()},summary=summary,
        bindings=bindings,cache_digest=digest.hexdigest(),coefficient_hashes={t:sha256(private/f'{t}.pt') for t in choices},
        runtime_modified=False,promotion_allowed=False,
        limitation='Fixed strength1 both arms; convex per-source-group/site SmoothL1 non-regression constraints. In-sample constraints are not held-out safety guarantees. Reused source-selection; current masked reconstruction only; no placebo/interval/full promotion.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
