"""Current body versus dimension-matched raw masked history diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.audit_reconstruction_raw_state import summarize
from scripts.refit_anchored_body_readout import source_weights
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.multisource_intervals import sha256


def project_fit(train, test, weights, reduce):
    scaler=StandardScaler().fit(train,sample_weight=weights)
    x=scaler.transform(train); tx=scaler.transform(test)
    if reduce:
        if min(x.shape)<64:raise ValueError('Insufficient rows/dimensions for matched projection')
        _,_,vt=np.linalg.svd(x*np.sqrt(weights[:,None]),full_matrices=False)
        x=x@vt[:64].T;tx=tx@vt[:64].T
        scaler=StandardScaler().fit(x,sample_weight=weights)
        x=scaler.transform(x);tx=scaler.transform(tx)
    if x.shape[1]!=64:raise ValueError('Unmatched capacity')
    return x,tx


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_frozen_readout_spectrum_20260920.json')
    out=Path('artifacts/native_reconstruction_current_raw_body_probe_20260920.json')
    if out.exists():raise ValueError('Existing experiment')
    parent=json.loads(pp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Parent changed')
    mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    manifest=torch.load(mp,weights_only=False);joint=torch.load(jp,weights_only=False)
    variables=joint['variables'];targets=joint['targets']
    private=root/'current_raw_body_probe_20260920';private.mkdir(exist_ok=True)
    bindings={str(p):sha256(p) for p in (Path(__file__),pp,mp,jp,Path('scripts/audit_reconstruction_raw_state.py'),Path('whole_body_jepa/native_body_reconstruction.py'),Path('RECONSTRUCTION_CURRENT_RAW_BODY_PROTOCOL_20260920.md'))}
    bp=private/'bindings.json'
    if bp.exists() and json.loads(bp.read_text())!=bindings:raise ValueError('Resume changed')
    if not bp.exists():bp.write_text(json.dumps(bindings,indent=2)+'\n')
    data={r:{t:[] for t in targets} for r in ('fit','encoder_tune')};digest=hashlib.sha256();raw_digest=hashlib.sha256()
    torch.set_num_threads(4)
    for i,s in enumerate(manifest['shards']):
        fp=root/'frozen_readout_refit_20260920'/f'features_{i:05d}.pt'
        digest.update((fp.name+sha256(fp)).encode());body=torch.load(fp,weights_only=False)
        if body['shard']!=s:raise ValueError('Body identity changed')
        cache=private/f'raw_{i:05d}.pt'
        if cache.exists():saved=torch.load(cache,weights_only=False)
        else:
            path=Path(s['path'])
            if sha256(path)!=s['sha256']:raise ValueError('Source history changed')
            patient=torch.load(path,weights_only=False)
            if patient['patient']!=s['patient']:raise ValueError('Wrong history patient')
            features={}
            for row in body['rows']:
                target=row['target'];e=patient['samples'][variables.index(target)]
                if e['truth_normalized']!=row['truth']:raise ValueError('Label mismatch')
                raw,process=summarize(e['tokens'],e['event_hours'],e['anchor_hour'],variables,target)
                features[target]={'raw':raw,'process':process}
            saved=dict(shard=s,features=features)
            temp=cache.with_suffix('.tmp');torch.save(saved,temp);temp.replace(cache)
        if saved['shard']!=s or set(saved['features'])!={r['target'] for r in body['rows']}:raise ValueError('Raw identity mismatch')
        raw_digest.update((cache.name+sha256(cache)).encode())
        for row in body['rows']:data[s['role']][row['target']].append(dict(row,group=s['group'],**saved['features'][row['target']]))
        if (i+1)%2000==0:print('Matched histories',i+1,flush=True)
    if digest.hexdigest()!=parent['cache_digest']:raise ValueError('Body cache changed')
    results={}
    with threadpool_limits(limits=4):
        for target in targets:
            fitrows=data['fit'][target];held=data['encoder_tune'][target]
            if {r['patient'] for r in fitrows}&{r['patient'] for r in held}:raise ValueError('Patient overlap')
            groups=np.array([r['group'] for r in fitrows]);sites=np.array([str(r['site']) for r in fitrows]);w=source_weights(groups,sites)*len(fitrows)
            y=np.array([r['truth'] for r in fitrows]);ty=np.array([r['truth'] for r in held])
            predictions={'original':np.array([r['prediction'] for r in held])}
            for key in ('body','raw','process'):
                x,tx=project_fit(np.stack([r[key] for r in fitrows]).astype(float),np.stack([r[key] for r in held]).astype(float),w,key!='body')
                model=Ridge(alpha=100.,solver='svd').fit(x,y,sample_weight=w)
                predictions[key]=model.predict(tx)
            per={}
            for group in ('selected','broad'):
                mask=np.array([r['group']==group for r in held]);ss=[r['site'] for r in held if r['group']==group]
                errors={k:abs(v[mask]-ty[mask]) for k,v in predictions.items()}
                checks={ref:evidence(errors['raw']-errors[ref],ss) for ref in ('body','process','original')}
                per[group]=dict(patients=int(mask.sum()),mae={k:float(v.mean()) for k,v in errors.items()},comparisons=checks,
                    raw_pass=confirmed(checks,int(mask.sum())))
            results[target]=per;print('Compared',target,flush=True)
    summary={g:dict(mae={k:float(np.mean([v[g]['mae'][k] for v in results.values()])) for k in predictions},
        raw_pass=[t for t,v in results.items() if v[g]['raw_pass']],
        raw_better_than_body=sum(v[g]['mae']['raw']<v[g]['mae']['body'] for v in results.values())) for g in ('selected','broad')}
    out.write_text(json.dumps(dict(targets=results,summary=summary,bindings=bindings,cache_digest=digest.hexdigest(),raw_cache_digest=raw_digest.hexdigest(),
        runtime_modified=False,promotion_allowed=False,
        limitation='64-dimensional fixed linear readouts; raw/process weighted PCA fit only on source, body64 existing encoder. Fixedridge100; PCA may discard raw predictive directions and summaries omit detailed order. Not end-to-end encoder crossfit or fresh lockbox. Current reconstruction only.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
