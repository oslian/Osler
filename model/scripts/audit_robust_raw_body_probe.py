"""Matched64 source probes: SmoothL1 versus previous squared-error loss."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from scripts.audit_raw_probe_scaling import audited_projection
from scripts.refit_anchored_body_readout import source_weights
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.multisource_intervals import sha256


def fit_smooth(x,y,w,alpha=100.):
    n=len(y);w=w/w.sum()
    def objective(theta):
        r=x@theta[:-1]+theta[-1]-y;a=abs(r)
        loss=np.where(a<1,.5*r*r,a-.5);derivative=w*np.clip(r,-1,1)
        return (float(w@loss+.5*alpha/n*(theta[:-1]@theta[:-1])),
                np.r_[x.T@derivative+alpha/n*theta[:-1],derivative.sum()])
    initial=np.r_[np.zeros(x.shape[1]),w@y]
    solved=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7))
    if not solved.success or not np.isfinite(solved.x).all():raise ValueError('Failed SmoothL1 fit: '+str(solved.message))
    return solved.x[:-1],float(solved.x[-1]),dict(iterations=int(solved.nit),objective=float(solved.fun),gradient_max=float(abs(solved.jac).max()))


def main():
    root=Path('data/research_recovery/broad_source_20260917');private=root/'current_raw_body_probe_20260920'
    pp=Path('artifacts/native_reconstruction_current_raw_body_probe_20260920.json')
    sp=Path('artifacts/native_reconstruction_raw_probe_scaling_20260920.json')
    out=Path('artifacts/native_reconstruction_robust_raw_body_probe_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Parent changed')
    mp=root/'site_balance_manifest_20260920.pt';manifest=torch.load(mp,weights_only=False)
    targets=list(parent['targets']);data={r:{t:[] for t in targets} for r in ('fit','encoder_tune')};bd=hashlib.sha256();rd=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        bp=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt';rp=private/f'raw_{i:05d}.pt'
        bd.update((bp.name+sha256(bp)).encode());rd.update((rp.name+sha256(rp)).encode())
        body=torch.load(bp,weights_only=False);raw=torch.load(rp,weights_only=False)
        if body['shard']!=s or raw['shard']!=s:raise ValueError('Wrong cache identity')
        for r in body['rows']:data[s['role']][r['target']].append(dict(r,group=s['group'],**raw['features'][r['target']]))
    if bd.hexdigest()!=parent['cache_digest'] or rd.hexdigest()!=parent['raw_cache_digest']:raise ValueError('Cache changed')
    results={}
    with threadpool_limits(limits=4):
        for t in targets:
            rows=data['fit'][t];held=data['encoder_tune'][t]
            groups=np.array([r['group'] for r in rows]);sites=np.array([str(r['site']) for r in rows]);w=source_weights(groups,sites)*len(rows)
            y=np.array([r['truth'] for r in rows]);ty=np.array([r['truth'] for r in held]);pred={'original':np.array([r['prediction'] for r in held])};solvers={}
            for key in ('body','raw','process'):
                x,q,_=audited_projection(np.stack([r[key] for r in rows]).astype(float),np.stack([r[key] for r in held]).astype(float),w,key!='body',True)
                weight,bias,info=fit_smooth(x,y,w);pred[key]=q@weight+bias;solvers[key]=info
            per={}
            for group in ('selected','broad'):
                mask=np.array([r['group']==group for r in held]);ss=[r['site'] for r in held if r['group']==group]
                errors={k:abs(v[mask]-ty[mask]) for k,v in pred.items()}
                checks={ref:evidence(errors['raw']-errors[ref],ss) for ref in ('body','process','original')}
                per[group]=dict(patients=int(mask.sum()),mae={k:float(v.mean()) for k,v in errors.items()},comparisons=checks,raw_pass=confirmed(checks,int(mask.sum())))
            results[t]=dict(groups=per,solvers=solvers);print('Robust probe',t,flush=True)
    summary={g:dict(mae={k:float(np.mean([v['groups'][g]['mae'][k] for v in results.values()])) for k in pred},raw_pass=[t for t,v in results.items() if v['groups'][g]['raw_pass']],
        raw_better_than_body=sum(v['groups'][g]['mae']['raw']<v['groups'][g]['mae']['body'] for v in results.values())) for g in ('selected','broad')}
    out.write_text(json.dumps(dict(targets=results,summary=summary,bindings={str(p):sha256(p) for p in (Path(__file__),pp,sp,mp,Path('scripts/audit_raw_probe_scaling.py'),Path('RECONSTRUCTION_ROBUST_RAW_BODY_PROTOCOL_20260920.md'))},runtime_modified=False,promotion_allowed=False,
        limitation='Same64representations, SmoothL1beta1 replaces half-squared loss; same alpha100/n L2 with unpenalized intercept. Source-only transforms, no hyperparameter search. Raw summary/PCA limitations remain; current reconstruction and reused source selection only.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
