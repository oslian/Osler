"""Exact one-dimensional expected training-risk optimum with frozen predictions."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import brentq
from scripts.audit_bun_affine_transfer import arrays
from scripts.audit_bun_dynamics_fit import smooth_l1_from_error
from scripts.audit_continued_selection_pairs import interval
from scripts.bounded_site_sampling import bounded_weights
from whole_body_jepa.multisource_intervals import sha256


def optimum(residual,weights):
    r=np.asarray(residual,dtype=float);w=np.asarray(weights,dtype=float)
    if r.shape!=w.shape or r.ndim!=1 or not len(r) or not np.isfinite(r).all() or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Finite residuals and positive weights required')
    gradient=lambda shift:float(np.average(np.clip(r+shift,-1,1),weights=w))
    shift=float(brentq(gradient,-float(r.max())-1,-float(r.min())+1,xtol=1e-12))
    return shift,gradient(0),gradient(shift)


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    fp=root/'bun_dynamics_fit_20260920.pt';vp=root/'bun_offset_transfer_20260920.pt'
    mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    rp=Path('artifacts/native_reconstruction_bun_offset_transfer_20260920.json')
    out=Path('artifacts/native_reconstruction_bun_bias_stationarity_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    prior=json.loads(rp.read_text())
    if sha256(vp)!=prior['private_sha256']:raise ValueError('Selection changed')
    for p,h in prior['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    fit,tune,manifest,joint=[torch.load(p,weights_only=False) for p in (fp,vp,mp,jp)]
    col=joint['variables'].index('bun');residuals=[];weights=[]
    for g,rows in fit.items():
        x,y=arrays(rows,'full');residuals.append(x-y)
        w=bounded_weights([manifest['sample_sites'][g,'fit',r['patient'],col] for r in rows])
        weights.append(w/w.sum()/len(fit))
    r=np.concatenate(residuals);w=np.concatenate(weights)
    shift,gradient_before,gradient_after=optimum(r,w)
    training={}
    for name,delta in [('original',0.),('training_optimum',shift),*[(g+'_median',v['full']) for g,v in prior['offsets'].items()]]:
        training[name]=dict(shift=delta,mae=float(np.average(abs(r+delta),weights=w)),
            smooth_l1=float(np.average(smooth_l1_from_error(abs(r+delta)),weights=w)))
    selection={}
    for g,rows in tune.items():
        x,y=arrays(rows,'full');before=abs(x-y);after=abs(x+shift-y)
        selection[g]=dict(patients=len(rows),mae=float(after.mean()),vs_original=interval(after-before),
            smooth_l1_delta=interval(smooth_l1_from_error(after)-smooth_l1_from_error(before)))
    result=dict(shift=shift,gradient_before=gradient_before,gradient_after=gradient_after,training=training,selection=selection,
        bindings={str(p):sha256(p) for p in (Path(__file__),fp,vp,mp,jp,rp)},runtime_modified=False,promotion_allowed=False,
        limitation='Expected bounded-site weights with equal source-group mass. Frozen current-reconstruction readout bias direction only; no optimizer convergence proof, AdamW penalty, or historical per-step loss claim. Reused selection, descriptive intervals, not new external validation.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
