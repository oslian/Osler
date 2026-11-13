"""Fixed fit-prediction bins; descriptive residual shift, not routing."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_bun_affine_transfer import arrays
from whole_body_jepa.multisource_intervals import sha256


def median_interval(values):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError('Invalid values')
    if len(values)<30:return dict(n=len(values),median=float(np.median(values)),lower=None,upper=None)
    rng=np.random.default_rng(20260920);draws=[]
    for _ in range(40):
        draws.extend(np.median(values[rng.integers(len(values),size=(100,len(values)))],axis=1))
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(n=len(values),median=float(np.median(values)),lower=float(lo),upper=float(hi))


def summarize(rows,cuts):
    x,y=arrays(rows,'full');bins=np.searchsorted(cuts,x,side='right')
    result=[]
    for i in range(len(cuts)+1):
        mask=bins==i
        result.append(dict(bin=i,patients=int(mask.sum()),residual=median_interval((y-x)[mask]) if mask.any() else None))
    return result


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    fp=root/'bun_dynamics_fit_20260920.pt';vp=root/'bun_offset_transfer_20260920.pt'
    report=Path('artifacts/native_reconstruction_bun_offset_transfer_20260920.json')
    out=Path('artifacts/native_reconstruction_bun_conditional_residual_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    prior=json.loads(report.read_text())
    if sha256(vp)!=prior['private_sha256']:raise ValueError('Changed selection')
    for p,h in prior['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    fit,tune=[torch.load(p,weights_only=False) for p in (fp,vp)]
    if {r['patient'] for rows in fit.values() for r in rows}&{r['patient'] for rows in tune.values() for r in rows}:raise ValueError('Overlap')
    cuts=np.unique(np.quantile(np.concatenate([arrays(rows,'full')[0] for rows in fit.values()]),[.25,.5,.75]))
    result={role:{g:summarize(rows,cuts) for g,rows in data.items()} for role,data in (('fit',fit),('selection',tune))}
    output=dict(prediction_bin_cuts=cuts.tolist(),groups=result,minimum_interval_support=30,
        runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),fp,vp,report)},
        limitation='Pooled fit-prediction quartiles, no truth-defined bins or correction fitting. Unadjusted patient median bootstrap, reused selection. Coarse matching cannot establish identical full states or uniquely isolate biological versus measurement-process shift.')
    out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
