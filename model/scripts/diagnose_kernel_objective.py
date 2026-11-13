"""FIT-only spectrum and regularized coefficient-ray diagnosis, no refitting."""
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.diagnose_nonlinear_fit_gap import weighted_median
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_kernel_probe import map_query, DEST as OLD_DEST
from scripts.train_scaled_kernel import ROOT, PRIVATE, DEST
from scripts.train_nonlinear_probe import load_target
from whole_body_jepa.multisource_intervals import sha256


def spectrum(features, weights):
    z=np.asarray(features,float); w=np.asarray(weights,float)
    if (z.ndim!=2 or w.shape!=(len(z),) or not np.isfinite(z).all()
            or not np.isfinite(w).all() or not (w>0).all()): raise ValueError('Invalid spectrum arrays')
    z=z-np.average(z,axis=0,weights=w)
    covariance=(z.T*w)@z/w.sum()
    eigen=np.linalg.eigvalsh(covariance)
    if eigen.min() < -1e-10*max(1.,float(abs(eigen).max())): raise ValueError('Non-PSD covariance')
    eigen=np.maximum(eigen,0)[::-1]; trace=float(eigen.sum())
    if trace==0: return dict(trace=0.,effective_rank=0.,variance95_rank=0,top1_fraction=None)
    p=eigen[eigen>0]/trace
    return dict(trace=trace,effective_rank=float(np.exp(-np.sum(p*np.log(p)))),
                variance95_rank=int(np.searchsorted(np.cumsum(p),.95)+1),top1_fraction=float(p[0]))


def ray_audit(delta, centered_truth, weights, norm_sq, c):
    p,y,w=[np.asarray(a,float) for a in (delta,centered_truth,weights)]
    if (p.ndim!=1 or not len(p) or y.shape!=p.shape or w.shape!=p.shape
            or not all(np.isfinite(a).all() for a in (p,y,w)) or not (w>0).all()
            or not np.isfinite([norm_sq,c]).all() or norm_sq<0 or c<=0):
        raise ValueError('Invalid objective arrays')
    nonzero=p!=0
    gain=max(0.,weighted_median(y[nonzero]/p[nonzero],w[nonzero]*abs(p[nonzero]))) if nonzero.any() else 0.
    def mae(s): return float(np.average(abs(s*p-y),weights=w))
    def objective(s): return .5*s*s*norm_sq+c*float(np.sum(w*abs(s*p-y)))
    optimized=minimize_scalar(objective,bounds=(0.,max(2.,gain)),method='bounded',options={'xatol':1e-9})
    if not optimized.success: raise ValueError('Ray objective solve failed')
    candidates=(0.,1.,gain,float(optimized.x))
    optimum=min(candidates,key=objective)
    base=objective(1.)
    return dict(current_weighted_mae=mae(1.),constant_weighted_mae=mae(0.),
                lad_gain=gain,lad_weighted_mae=mae(gain),
                current_penalty=.5*norm_sq,lad_penalty=.5*gain*gain*norm_sq,
                current_objective=base,constant_objective=objective(0.),lad_objective=objective(gain),
                ray_objective_gain=optimum,ray_min_objective=objective(optimum),
                ray_relative_objective_reduction=(base-objective(optimum))/max(1.,abs(base)))


def dispersion(prediction, truth, weights):
    p,y,w=[np.asarray(a,float) for a in (prediction,truth,weights)]
    pc=p-np.average(p,weights=w); yc=y-np.average(y,weights=w)
    pv=float(np.average(pc**2,weights=w)); yv=float(np.average(yc**2,weights=w))
    pi=float(np.subtract(*np.percentile(p,[75,25]))); yi=float(np.subtract(*np.percentile(y,[75,25])))
    return dict(weighted_prediction_std=float(np.sqrt(pv)),weighted_truth_std=float(np.sqrt(yv)),
                weighted_std_ratio=float(np.sqrt(pv/yv)) if yv>0 else None,
                patient_iqr_ratio=pi/yi if yi>0 else None,
                weighted_correlation=float(np.average(pc*yc,weights=w)/np.sqrt(pv*yv)) if pv>0 and yv>0 else None)


def main():
    out=Path('artifacts/native_reconstruction_kernel_objective_20261003.json')
    if out.exists(): raise ValueError('Existing objective diagnosis')
    sp=ROOT/'scaling_20260922.pt'; fp=ROOT/'partitions_20260922.pt'; pp=ROOT/'provenance.parquet'
    variables=torch.load(sp,weights_only=False)['variables']; roles=torch.load(fp,weights_only=False)['folds']
    links=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    components=dict(links.itertuples(index=False,name=None))
    paths=[Path(__file__),Path('RECONSTRUCTION_KERNEL_OBJECTIVE_PROTOCOL_20261003.md'),sp,fp,pp,
           Path('scripts/train_kernel_probe.py'),Path('scripts/diagnose_nonlinear_fit_gap.py')]
    checked={}; folds={}
    with threadpool_limits(limits=4):
        for f in range(5):
            reports={}
            for version in ('kernel','scaled_kernel'):
                rp=Path(f'artifacts/native_reconstruction_{version}_fold{f}_20261003.json')
                reports[version]=json.loads(rp.read_text()); verify_bound_files(reports[version]['bindings'],checked); paths.append(rp)
            targets=sorted(reports['kernel']['targets'])
            if len(targets)!=47 or targets!=sorted(reports['scaled_kernel']['targets']): raise ValueError('Changed grid')
            cache=PRIVATE/f'fold{f}_fit.sqlite'; paths.append(cache)
            db=sqlite3.connect(cache.resolve().as_uri()+'?mode=ro',uri=True); folds[str(f)]={}
            try:
                for t in targets:
                    ids,data=load_target(db,variables.index(t))
                    if any(roles[f][p]!='fit' for p in ids): raise ValueError('Non-fitting diagnosis')
                    weights=bounded_weights([components[p] for p in ids]); row={}
                    for version,dest,prefix in (('kernel',OLD_DEST,'kernel'),('scaled_kernel',DEST,'scaled')):
                        meta=reports[version]['targets'][t]; mp=dest/f'fold{f}/{t}.joblib'
                        if meta['fit_patients']!=len(ids) or sha256(mp)!=meta['model_sha256']: raise ValueError('Changed fit artifacts')
                        models=joblib.load(mp); paths.append(mp)
                        for source in ('body','raw','process'):
                            a=prefix+'_'+source; saved=models[a]; model=saved['model']
                            x=np.array(data['body' if source=='body' else 'raw'],float,copy=True)
                            if source=='process': x[:,:106]=0
                            if version=='scaled_kernel': x=saved['scaler'].transform(x)
                            features=map_query(x,saved['mapping']); delta=model.predict(features)
                            y=data['truth']; prediction=delta+saved['offset']
                            if not np.isclose(np.mean(abs(prediction-y)),meta['checks'][a]['fit_mae'],rtol=1e-10,atol=1e-12):
                                raise ValueError('Original fit MAE not reproduced')
                            if model.epsilon!=0 or model.loss!='epsilon_insensitive' or not model.fit_intercept:
                                raise ValueError('Unsupported objective')
                            norm_sq=float(np.sum(model.coef_**2)+np.sum((model.intercept_/model.intercept_scaling)**2))
                            row[a]=dict(spectrum=spectrum(features,weights),dispersion=dispersion(prediction,y,weights),
                                        ray=ray_audit(delta,y-saved['offset'],weights,norm_sq,model.C))
                    folds[str(f)][t]=dict(fit_patients=len(ids),arms=row)
            finally: db.close()
            print('Objective/spectrum complete fold',f,flush=True)
    arms=next(iter(folds['0'].values()))['arms']; summary={}
    for a in arms:
        rows=[r['arms'][a] for f in folds.values() for r in f.values()]
        ratios=[r['dispersion']['weighted_std_ratio'] for r in rows if r['dispersion']['weighted_std_ratio'] is not None]
        summary[a]=dict(fold_targets=len(rows),median_effective_rank=float(np.median([r['spectrum']['effective_rank'] for r in rows])),
            median_variance95_rank=float(np.median([r['spectrum']['variance95_rank'] for r in rows])),
            median_top1_fraction=float(np.median([r['spectrum']['top1_fraction'] for r in rows if r['spectrum']['top1_fraction'] is not None])),
            median_std_ratio=float(np.median(ratios)),median_lad_gain=float(np.median([r['ray']['lad_gain'] for r in rows])),
            lad_gain_above_1_05=sum(r['ray']['lad_gain']>1.05 for r in rows),
            median_lad_relative_mae_reduction=float(np.median([(r['ray']['current_weighted_mae']-r['ray']['lad_weighted_mae'])/max(1e-12,r['ray']['current_weighted_mae']) for r in rows])),
            max_ray_relative_objective_reduction=max(r['ray']['ray_relative_objective_reduction'] for r in rows),
            ray_objective_improvement_above_1e_6=sum(r['ray']['ray_relative_objective_reduction']>1e-6 for r in rows),
            lad_objective_worse=sum(r['ray']['lad_objective']>r['ray']['current_objective']+1e-6 for r in rows))
    report=dict(summary=summary,folds=folds,bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
                limitation='Fit-only exploratory diagnostic. Ray gain uses fitting truth; never a query feature or deployed correction. One-dimensional stationarity not full solver optimality. Finite-feature spectrum not information sufficiency or physiological rank.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
