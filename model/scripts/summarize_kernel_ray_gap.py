"""Compare oracle FIT-ray MAE with same-row bounded-weight HGB fitting MAE."""
import json
from pathlib import Path
import numpy as np
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def gap_summary(current, ray, reference):
    current,ray,reference=[np.asarray(x,float) for x in (current,ray,reference)]
    if (current.ndim!=1 or len(current)==0 or current.shape!=ray.shape or ray.shape!=reference.shape
            or not all(np.isfinite(x).all() for x in (current,ray,reference))
            or any((x<0).any() for x in (current,ray,reference)) or (ray>current+1e-9).any()):
        raise ValueError('Invalid comparable fitting errors')
    gap=current-reference; supported=gap>1e-12
    recovered=(current[supported]-ray[supported])/gap[supported]
    return dict(combinations=len(current),current_harms=int((gap>0).sum()),
                oracle_ray_harms=int((ray>reference).sum()),positive_gap_combinations=int(supported.sum()),
                median_gap_fraction_recovered=float(np.median(recovered)) if len(recovered) else None)


def main():
    out=Path('artifacts/native_reconstruction_kernel_ray_gap_20261003.json')
    if out.exists(): raise ValueError('Existing ray gap audit')
    rp=Path('artifacts/native_reconstruction_kernel_objective_20261003.json')
    hp=Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    r,h=[json.loads(p.read_text()) for p in (rp,hp)]; checked={}
    for report in (r,h): verify_bound_files(report['bindings'],checked)
    results={}
    for arm,ref in (('scaled_body','body_probe'),('scaled_raw','raw_probe'),('scaled_process','process_summary')):
        current,ray,reference=[],[],[]
        for fold,rows in r['folds'].items():
            if len(rows)!=47: raise ValueError('Incomplete target grid')
            for target,z in rows.items():
                fit=h['folds'][fold][target]['fit']
                if fit['patients']!=z['fit_patients']: raise ValueError('Changed fitting support')
                current.append(z['arms'][arm]['ray']['current_weighted_mae'])
                ray.append(z['arms'][arm]['ray']['lad_weighted_mae'])
                reference.append(fit['normalized_mae']['bounded'][ref])
        if len(current)!=235: raise ValueError('Incomplete fold grid')
        results[arm]=gap_summary(current,ray,reference)
    out.write_text(json.dumps(dict(summary=results,bindings={str(p):sha256(p) for p in (Path(__file__),rp,hp)},
        runtime_modified=False,promotion_allowed=False,
        limitation='Training-label optimum along frozen coefficient ray only. No validation/external inference, no deployable correction. Same bounded weights; median fractional recovery only where initial gap positive. Does not bound gains from new coefficient directions.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
