"""Exact MAE/L1-anchored linear heads, with a matched site-constrained arm."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scripts.refit_anchored_body_readout import source_weights
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.multisource_intervals import sha256


def fit_mae(x,y,groups,sites,oldw,oldb,constrained,alpha=.1):
    w=source_weights(groups,sites);mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.
    design=np.c_[(x-mean)/scale,np.ones(len(y))]
    anchor=np.r_[oldw*scale,oldb+mean@oldw];residual=design@anchor-y
    n,d=design.shape;z=csr_matrix(design);ident=eye(d,format='csr');obs=eye(n,format='csr')
    # Variables: coefficient change, absolute residual, absolute change.
    blocks=[hstack((z,-obs,csr_matrix((n,d)))),hstack((-z,-obs,csr_matrix((n,d)))),
            hstack((ident,csr_matrix((d,n)),-ident)),hstack((-ident,csr_matrix((d,n)),-ident))]
    rhs=[-residual,residual,np.zeros(d),np.zeros(d)]
    masks=[(groups==g)&(sites==s) for g in sorted(set(groups)) for s in sorted(set(sites[groups==g]))]
    baseline=np.array([abs(residual[m]).mean() for m in masks])
    if constrained:
        averaging=csr_matrix(np.stack([m/m.sum() for m in masks]))
        blocks.append(hstack((csr_matrix((len(masks),d)),averaging,csr_matrix((len(masks),d)))))
        rhs.append(baseline)
    cost=np.r_[np.zeros(d),w,np.full(d,alpha)]
    solved=linprog(cost,A_ub=vstack(blocks,format='csr'),b_ub=np.concatenate(rhs),
        bounds=[(None,None)]*d+[(0,None)]*(n+d),method='highs')
    if not solved.success:raise ValueError('MAE fit failed: '+solved.message)
    delta=solved.x[:d];theta=anchor+delta
    new=design@theta-y;site_change=np.array([abs(new[m]).mean() for m in masks])-baseline
    actual=float(w@abs(new)+alpha*abs(delta).sum())
    if not np.isfinite(theta).all() or actual>float(w@abs(residual))+1e-7 or abs(actual-solved.fun)>1e-6:
        raise ValueError('Invalid LP objective')
    if constrained and site_change.max()>1e-7:raise ValueError('MAE site constraint violated')
    weight=theta[:-1]/scale;bias=float(theta[-1]-mean@weight)
    return weight,bias,dict(iterations=int(solved.nit),sites=len(masks),max_site_mae_change=float(site_change.max()),
        source_mae_before=float(w@abs(residual)),source_mae_after=float(w@abs(new)),objective=actual,
        correction_l1=float(abs(delta).sum()),alpha=alpha)


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_site_constrained_readout_20260920.json')
    out=Path('artifacts/native_reconstruction_mae_site_readout_20260920.json')
    if out.exists():raise ValueError('Existing experiment')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Parent changed')
    mp=root/'site_balance_manifest_20260920.pt';cp=root/'bounded_site_training/bounded.pt'
    manifest=torch.load(mp,weights_only=False);ck=torch.load(cp,weights_only=False)
    targets=list(parent['targets']);data={r:{t:[] for t in targets} for r in ('fit','encoder_tune')};digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        p=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt';digest.update((p.name+sha256(p)).encode())
        saved=torch.load(p,weights_only=False)
        if saved['shard']!=s:raise ValueError('Wrong cache identity')
        for row in saved['rows']:data[s['role']][row['target']].append(dict(row,group=s['group']))
    if digest.hexdigest()!=parent['cache_digest']:raise ValueError('Changed cache')
    bindings={str(p):sha256(p) for p in (Path(__file__),pp,mp,cp,Path('RECONSTRUCTION_MAE_SITE_PROTOCOL_20260920.md'),Path('scripts/refit_anchored_body_readout.py'))}
    private=root/'mae_site_readout_20260920';private.mkdir(exist_ok=True);fitted={}
    for target,rows in data['fit'].items():
        path=private/f'{target}.pt'
        if path.exists():
            saved=torch.load(path,weights_only=False)
            if saved['bindings']!=bindings:raise ValueError('Resume mismatch')
            fitted[target]=saved;continue
        col=ck['variables'].index(target);oldw=ck['best_state_dict']['readout.weight'][col].numpy().astype(float);oldb=float(ck['best_state_dict']['readout.bias'][col])
        x=np.stack([r['body'] for r in rows]).astype(float);y=np.array([r['truth'] for r in rows]);groups=np.array([r['group'] for r in rows]);sites=np.array([str(r['site']) for r in rows])
        saved={'bindings':bindings}
        for arm in ('unconstrained','constrained'):
            weight,bias,info=fit_mae(x,y,groups,sites,oldw,oldb,arm=='constrained')
            saved[arm]=dict(weight=weight,bias=bias,solver=info)
            print(target,arm,info,flush=True)
        temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path);fitted[target]=saved
    results={}
    for target,rows in data['encoder_tune'].items():
        per={}
        for group in ('selected','broad'):
            rr=[r for r in rows if r['group']==group];x=np.stack([r['body'] for r in rr]).astype(float);y=np.array([r['truth'] for r in rr]);sites=[r['site'] for r in rr]
            errors={'original':abs(np.array([r['prediction'] for r in rr])-y)}
            for arm in ('unconstrained','constrained'):
                head=fitted[target][arm];errors[arm]=abs(x@head['weight']+head['bias']-y)
            checks={arm:evidence(errors['constrained']-errors[arm],sites) for arm in ('original','unconstrained')}
            per[group]=dict(patients=len(rr),mae={a:float(e.mean()) for a,e in errors.items()},comparisons=checks,pass_incremental=confirmed(checks,len(rr)))
        results[target]=per
    summary={g:dict(mae={a:float(np.mean([v[g]['mae'][a] for v in results.values()])) for a in ('original','unconstrained','constrained')},passed=[t for t,v in results.items() if v[g]['pass_incremental']]) for g in ('selected','broad')}
    out.write_text(json.dumps(dict(targets=results,summary=summary,solvers={t:{a:v[a]['solver'] for a in ('unconstrained','constrained')} for t,v in fitted.items()},
        bindings=bindings,coefficient_hashes={t:sha256(private/f'{t}.pt') for t in targets},cache_digest=digest.hexdigest(),runtime_modified=False,promotion_allowed=False,
        limitation='Fixedalpha0.1 MAE/L1 matched arms; empirical source constraints not heldout guarantees. Reused source selection, frozen encoder, current reconstruction; no full promotion.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
