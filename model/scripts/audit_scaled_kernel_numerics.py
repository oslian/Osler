"""Audit saved scalers and fitting errors without changing completed models."""
import json
from pathlib import Path
import joblib
import numpy as np

from scripts.train_scaled_kernel import DEST
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def check_scaler(scaler):
    v,m,s=np.asarray(scaler.var_),np.asarray(scaler.mean_),np.asarray(scaler.scale_)
    if not all(np.isfinite(a).all() for a in (v,m,s)) or not (s>0).all():
        raise ValueError('Nonfinite/nonpositive scaler')
    eps=np.finfo(float).eps
    bound=scaler.n_samples_seen_*eps*v+(scaler.n_samples_seen_*m*eps)**2
    negative=v<0
    if np.any((-v[negative])>abs(bound[negative])) or np.any(s[negative]!=1):
        raise ValueError('Negative variance exceeds constant-feature roundoff or has invalid scale')
    return dict(negative_variances=int(negative.sum()),minimum_variance=float(v.min()),
                guarded_near_constants=int((v<=bound).sum()),exact_zero_variances=int((v==0).sum()),
                finite_positive_scale=True,negative_variance_scales_are_one=True)


def main():
    out=Path('artifacts/native_reconstruction_scaled_kernel_numerics_20261003.json')
    if out.exists(): raise ValueError('Existing numerical audit')
    hp=Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    old=json.loads(hp.read_text()); checked={}; verify_bound_files(old['bindings'],checked)
    paths=[Path(__file__),hp]; folds={}
    for f in range(5):
        p=Path(f'artifacts/native_reconstruction_scaled_kernel_fold{f}_20261003.json')
        report=json.loads(p.read_text()); verify_bound_files(report['bindings'],checked)
        paths.append(p); folds[str(f)]={}
        if len(report['targets'])!=47: raise ValueError('Incomplete source targets')
        for t,meta in report['targets'].items():
            mp=DEST/f'fold{f}/{t}.joblib'
            if sha256(mp)!=meta['model_sha256']: raise ValueError('Changed fitted artifact')
            models=joblib.load(mp); paths.append(mp)
            fit=old['folds'][str(f)][t]['fit']
            if fit['patients']!=meta['fit_patients']: raise ValueError('Changed fitting support')
            arms={}
            for a,b in (('scaled_body','body_probe'),('scaled_raw','raw_probe'),('scaled_process','process_summary')):
                arms[a]=dict(check_scaler(models[a]['scaler']),
                    fit_delta_vs_HGB=meta['checks'][a]['fit_mae']-fit['normalized_mae']['patient'][b])
            folds[str(f)][t]=arms
    rows=[r for f in folds.values() for t in f.values() for r in t.values()]
    summary=dict(maps=len(rows),negative_variance_maps=sum(r['negative_variances']>0 for r in rows),
        negative_variances=sum(r['negative_variances'] for r in rows),minimum_variance=min(r['minimum_variance'] for r in rows),
        finite_positive_scales=True,negative_variance_scales_are_one=True,
        fit_harms_vs_HGB={a:sum(t[a]['fit_delta_vs_HGB']>0 for f in folds.values() for t in f.values())
                          for a in ('scaled_body','scaled_raw','scaled_process')})
    out.write_text(json.dumps(dict(summary=summary,folds=folds,bindings={str(p):sha256(p) for p in paths},
        limitation='Post-run numerical diagnostic. sklearn warning evaluated sqrt before replacing near-constant scales with one. Original constant_features diagnostics count EXACT zeros only; guarded_near_constants includes roundoff. No refit, no efficacy gate change.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
