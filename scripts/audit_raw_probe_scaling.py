"""Locate weighted constant-column warnings and verify an exact-constant repair."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from scripts.refit_anchored_body_readout import source_weights
from whole_body_jepa.multisource_intervals import sha256


def scale_exact_constants(train,query,weights):
    constant=np.ptp(train,axis=0)==0
    x=np.zeros_like(train,dtype=float);q=np.empty_like(query,dtype=float)
    q[:,constant]=query[:,constant]-train[0,constant]
    if (~constant).any():
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            scaler=StandardScaler().fit(train[:,~constant],sample_weight=weights)
        x[:,~constant]=scaler.transform(train[:,~constant]);q[:,~constant]=scaler.transform(query[:,~constant])
    if not np.isfinite(x).all() or not np.isfinite(q).all():raise ValueError('Nonfinite standardized features')
    return x,q


def audited_projection(train,query,w,reduce,fixed):
    stages=[]
    def scale(a,b):
        if fixed:return scale_exact_constants(a,b,w)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',RuntimeWarning)
            scaler=StandardScaler().fit(a,sample_weight=w)
        negative=scaler.var_<0;constant=np.ptp(a,axis=0)==0
        stages.append(dict(warnings=len(caught),negative_variances=int(negative.sum()),
            negative_nonconstant=int((negative&~constant).sum()),min_variance=float(scaler.var_.min()),
            finite_scale=bool(np.isfinite(scaler.scale_).all())))
        return scaler.transform(a),scaler.transform(b)
    x,q=scale(train,query)
    if reduce:
        _,_,vt=np.linalg.svd(x*np.sqrt(w[:,None]),full_matrices=False)
        x,q=scale(x@vt[:64].T,q@vt[:64].T)
    return x,q,stages


def main():
    root=Path('data/research_recovery/broad_source_20260917');private=root/'current_raw_body_probe_20260920'
    pp=Path('artifacts/native_reconstruction_current_raw_body_probe_20260920.json')
    out=Path('artifacts/native_reconstruction_raw_probe_scaling_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed parent')
    mp=root/'site_balance_manifest_20260920.pt';manifest=torch.load(mp,weights_only=False)
    targets=list(parent['targets']);data={r:{t:[] for t in targets} for r in ('fit','encoder_tune')}
    bd=hashlib.sha256();rd=hashlib.sha256()
    for i,s in enumerate(manifest['shards']):
        bp=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt';rp=private/f'raw_{i:05d}.pt'
        bd.update((bp.name+sha256(bp)).encode());rd.update((rp.name+sha256(rp)).encode())
        body=torch.load(bp,weights_only=False);raw=torch.load(rp,weights_only=False)
        if body['shard']!=s or raw['shard']!=s:raise ValueError('Identity mismatch')
        for r in body['rows']:data[s['role']][r['target']].append(dict(r,group=s['group'],**raw['features'][r['target']]))
    if bd.hexdigest()!=parent['cache_digest'] or rd.hexdigest()!=parent['raw_cache_digest']:raise ValueError('Cache changed')
    result={}
    with threadpool_limits(limits=4):
        for t in targets:
            fitrows=data['fit'][t];held=data['encoder_tune'][t]
            groups=np.array([r['group'] for r in fitrows]);sites=np.array([str(r['site']) for r in fitrows]);w=source_weights(groups,sites)*len(fitrows)
            y=np.array([r['truth'] for r in fitrows]);ty=np.array([r['truth'] for r in held]);per={}
            for key in ('body','raw','process'):
                train=np.stack([r[key] for r in fitrows]).astype(float);query=np.stack([r[key] for r in held]).astype(float)
                x,q,stages=audited_projection(train,query,w,key!='body',False)
                old=Ridge(alpha=100.,solver='svd').fit(x,y,sample_weight=w).predict(q)
                x,q,_=audited_projection(train,query,w,key!='body',True)
                new=Ridge(alpha=100.,solver='svd').fit(x,y,sample_weight=w).predict(q)
                groupscores={}
                for g in ('selected','broad'):
                    m=np.array([r['group']==g for r in held]);before=float(abs(old[m]-ty[m]).mean());after=float(abs(new[m]-ty[m]).mean())
                    if not np.isclose(before,parent['targets'][t][g]['mae'][key],atol=1e-8,rtol=0):raise ValueError('Prior score mismatch')
                    groupscores[g]=dict(before=before,after=after,delta=after-before)
                per[key]=dict(stages=stages,max_prediction_change=float(abs(old-new).max()),scores=groupscores)
            result[t]=per;print('Verified',t,flush=True)
    stages=[s for p in result.values() for v in p.values() for s in v['stages']]
    summary=dict(warnings=sum(s['warnings'] for s in stages),negative_variances=sum(s['negative_variances'] for s in stages),
        negative_nonconstant=sum(s['negative_nonconstant'] for s in stages),min_variance=min(s['min_variance'] for s in stages),
        all_scales_finite=all(s['finite_scale'] for s in stages),max_prediction_change=max(v['max_prediction_change'] for p in result.values() for v in p.values()),
        max_mae_change=max(abs(s['delta']) for p in result.values() for v in p.values() for s in v['scores'].values()))
    out.write_text(json.dumps(dict(targets=result,summary=summary,bindings={str(p):sha256(p) for p in (Path(__file__),pp,mp)},runtime_modified=False,promotion_allowed=False,
        limitation='Exact-constant numerical handling only, all other scaling/PCA/ridge unchanged. Prior reports immutable. Not an accuracy intervention or fresh validation.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
