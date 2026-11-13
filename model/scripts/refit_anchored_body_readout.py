"""Source-hospital CV for same-capacity, original-head-anchored SmoothL1."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from scripts.refit_frozen_body_readout import objective
from scripts.bounded_site_sampling import bounded_weights
from scripts.audit_confirmed_reconstruction_heads import evidence,confirmed
from whole_body_jepa.multisource_intervals import sha256

STRENGTHS=(.01,.1,1.,10.)


def anchored_objective(theta,x,y,w,anchor,strength):
    value,grad=objective(theta,x,y,w);difference=theta-anchor
    return value+.5*strength*float(difference@difference),grad+strength*difference


def fit(x,y,w,oldw,oldb,strength):
    w=w/w.sum();mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.
    z=(x-mean)/scale;anchor=np.r_[oldw*scale,oldb+mean@oldw]
    solved=minimize(anchored_objective,anchor,args=(z,y,w,anchor,strength),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=50))
    if not solved.success:raise ValueError('Readout did not converge: '+str(solved.message))
    weight=solved.x[:-1]/scale
    return weight,float(solved.x[-1]-mean@weight)


def source_weights(groups,sites):
    result=np.empty(len(groups));names=sorted(set(groups))
    for g in names:
        m=groups==g;w=bounded_weights(sites[m].tolist());result[m]=w/w.sum()/len(names)
    return result


def source_score(errors,groups,sites):
    return float(np.mean([np.mean([errors[(groups==g)&(sites==s)].mean() for s in sorted(set(sites[groups==g]))]) for g in sorted(set(groups))]))


def main():
    root=Path('data/research_recovery/broad_source_20260917');features=root/'frozen_readout_refit_20260920'
    rp=Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    private=root/'anchored_readout_20260920';private.mkdir(exist_ok=True)
    out=Path('artifacts/native_reconstruction_anchored_readout_20260920.json')
    if out.exists():raise ValueError('Completed experiment')
    parent_report=json.loads(rp.read_text())
    for p,h in parent_report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    bindings={str(p):sha256(p) for p in (Path(__file__),rp,cp,mp,jp,Path('scripts/refit_frozen_body_readout.py'))}
    bp=private/'bindings.json'
    if bp.exists() and json.loads(bp.read_text())!=bindings:raise ValueError('Resume mismatch')
    if not bp.exists():bp.write_text(json.dumps(bindings,indent=2)+'\n')
    ck,manifest,joint=[torch.load(p,weights_only=False) for p in (cp,mp,jp)]
    variables,targets=joint['variables'],joint['targets'];data={r:{t:[] for t in targets} for r in ('fit','encoder_tune')};digest=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        if s['role'] not in data:continue
        p=features/f'features_{i:05d}.pt';digest.update((p.name+sha256(p)).encode());saved=torch.load(p,weights_only=False)
        if saved['shard']!=s:raise ValueError('Identity mismatch')
        for row in saved['rows']:data[s['role']][row['target']].append(dict(row,group=s['group']))
    if digest.hexdigest()!=parent_report['cache_digest']:raise ValueError('Feature digest changed')
    state=deepcopy(ck['best_state_dict']);choices={}
    with threadpool_limits(limits=4):
        for target in targets:
            col=variables.index(target);oldw=state['readout.weight'][col].numpy().astype(float);oldb=float(state['readout.bias'][col]);path=private/f'{target}.pt'
            if path.exists():choice=torch.load(path,weights_only=False)
            else:
                rows=data['fit'][target];x=np.stack([r['body'] for r in rows]).astype(float);y=np.array([r['truth'] for r in rows]);groups=np.array([r['group'] for r in rows]);sites=np.array([str(r['site']) for r in rows])
                candidates={'unchanged':source_score(abs(x@oldw+oldb-y),groups,sites)}
                folds=list(GroupKFold(n_splits=min(5,len(set(sites)))).split(x,y,sites))
                for strength in STRENGTHS:
                    predicted=np.empty(len(y))
                    for train,valid in folds:
                        w,b=fit(x[train],y[train],source_weights(groups[train],sites[train]),oldw,oldb,strength)
                        predicted[valid]=x[valid]@w+b
                    candidates[str(strength)]=source_score(abs(predicted-y),groups,sites)
                name=min(candidates,key=candidates.get)
                weight,bias=(oldw,oldb) if name=='unchanged' else fit(x,y,source_weights(groups,sites),oldw,oldb,float(name))
                choice=dict(name=name,cv_scores=candidates,weight=weight,bias=bias,folds=len(folds))
                torch.save(choice,path)
            state['readout.weight'][col]=torch.as_tensor(choice['weight'],dtype=state['readout.weight'].dtype);state['readout.bias'][col]=choice['bias']
            choices[target]={k:v for k,v in choice.items() if k not in ('weight','bias')}
            print('Source CV',target,choice['name'],flush=True)
    candidate=private/'candidate.pt';torch.save(dict(state_dict=state,choices=choices,bindings=bindings,promotion_allowed=False),candidate)
    results={}
    for target in targets:
        col=variables.index(target);per={}
        for group in ('selected','broad'):
            rows=[r for r in data['encoder_tune'][target] if r['group']==group]
            x=np.stack([r['body'] for r in rows]);y=np.array([r['truth'] for r in rows]);old=np.array([r['prediction'] for r in rows])
            pred=old.copy() if choices[target]['name']=='unchanged' else x@state['readout.weight'][col].numpy()+float(state['readout.bias'][col])
            check=evidence(abs(pred-y)-abs(old-y),[r['site'] for r in rows])
            per[group]=dict(patients=len(rows),before=float(abs(old-y).mean()),after=float(abs(pred-y).mean()),comparison=check,pass_incremental=confirmed({'base':check},len(rows)))
        results[target]=per
    summary={g:dict(before=float(np.mean([v[g]['before'] for v in results.values()])),after=float(np.mean([v[g]['after'] for v in results.values()])),passed=[t for t,v in results.items() if v[g]['pass_incremental']]) for g in ('selected','broad')}
    out.write_text(json.dumps(dict(choices=choices,source_selection=results,summary=summary,bindings=bindings,candidate_sha256=sha256(candidate),runtime_modified=False,promotion_allowed=False,
        limitation='Strength selected on fit-only hospital folds; fixed encoder already saw fitting sites, so inner CV is not end-to-end held-out validation. Untouched by this refit source-selection hospitals still reused historically. Same linear capacity; anchored L2 in source-standardized coordinates, not changing masks, labels or runtime.'),indent=2)+'\n')
    print('choices',dict(Counter(v['name'] for v in choices.values())));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
