"""Post-result, fit-only metric decomposition; no new head or model selection."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_kernel_probe import SETTINGS, SEED, ROOT, PRIVATE
from scripts.train_nonlinear_probe import load_target
from whole_body_jepa.multisource_intervals import sha256


def decompose(raw, weights, seed, gamma):
    raw=np.asarray(raw,dtype=float); weights=np.asarray(weights,dtype=float)
    if (raw.ndim!=2 or raw.shape[1]!=265 or len(raw)<2 or weights.shape!=(len(raw),)
            or not np.isfinite(raw).all() or not np.isfinite(weights).all() or not (weights>0).all()):
        raise ValueError('Invalid metric arrays')
    rng=np.random.default_rng(seed)
    pairs=rng.choice(len(raw),size=(SETTINGS['pairs'],2),p=weights/weights.sum())
    difference=(raw[pairs[:,0]]-raw[pairs[:,1]])**2
    blocks={name:difference[:,i*53:(i+1)*53].sum(1) for i,name in enumerate(('latest','mean','present','lag','logcount'))}
    total=difference.sum(1); valid=total>0
    expected=1./np.median(total[valid]) if valid.any() else 1.
    if not np.isclose(expected,gamma,rtol=1e-10,atol=1e-14): raise ValueError('Bandwidth not reproduced')
    if not valid.any():
        return dict(nonzero_pairs=0,median_fraction={k:None for k in blocks},numeric_fraction_median=None,
                    numeric_removal_kernel_change_mean=0.)
    numeric=blocks['latest']+blocks['mean']
    return dict(nonzero_pairs=int(valid.sum()),median_fraction={k:float(np.median(v[valid]/total[valid])) for k,v in blocks.items()},
        numeric_fraction_median=float(np.median(numeric[valid]/total[valid])),
        numeric_removal_kernel_change_mean=float(np.mean(np.exp(-gamma*(total-numeric))-np.exp(-gamma*total))))


def main():
    out=Path('artifacts/native_reconstruction_kernel_metric_20261003.json')
    if out.exists(): raise ValueError('Existing metric diagnosis')
    kp=Path('artifacts/native_reconstruction_kernel_assessment_20261003.json')
    hp=Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    kernel=json.loads(kp.read_text()); old=json.loads(hp.read_text())
    checked={}
    for r in (kernel,old): verify_bound_files(r['bindings'],checked)
    fp=ROOT/'partitions_20260922.pt'; sp=ROOT/'scaling_20260922.pt'; pp=ROOT/'provenance.parquet'
    roles=torch.load(fp,weights_only=False)['folds']; variables=torch.load(sp,weights_only=False)['variables']
    links=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    groups=dict(links.itertuples(index=False,name=None))
    paths=[Path(__file__),kp,hp,fp,sp,pp]; results={}
    for f in range(5):
        path=PRIVATE/f'fold{f}_fit.sqlite'; paths.append(path); results[str(f)]={}
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for t,diag in kernel['diagnostics'][str(f)].items():
                c=variables.index(t); ids,data=load_target(db,c)
                if any(roles[f][p]!='fit' for p in ids): raise ValueError('Query in fitting diagnosis')
                metric=decompose(data['raw'],bounded_weights([groups[p] for p in ids]),SEED+c,diag['kernel_raw']['gamma'])
                fit_reference=old['folds'][str(f)][t]['fit']
                if fit_reference['patients']!=len(ids): raise ValueError('Changed fitting support')
                metric['fit_delta_vs_HGB']={a:diag[a]['fit_mae']-fit_reference['normalized_mae']['patient'][b]
                    for a,b in (('kernel_body','body_probe'),('kernel_raw','raw_probe'),('kernel_process','process_summary'))}
                results[str(f)][t]=metric
        finally: db.close()
        print('Metric diagnosis complete fold',f,flush=True)
    rows=[r for f in results.values() for r in f.values()]
    valid=[r for r in rows if r['nonzero_pairs']]
    summary=dict(fold_targets=len(rows),nonzero_fold_targets=len(valid),
        median_numeric_fraction=float(np.median([r['numeric_fraction_median'] for r in valid])),
        median_lag_fraction=float(np.median([r['median_fraction']['lag'] for r in valid])),
        lag_dominant_above_90_percent=sum(r['median_fraction']['lag']>.9 for r in valid),
        median_kernel_change_removing_numeric=float(np.median([r['numeric_removal_kernel_change_mean'] for r in valid])),
        fit_harms_vs_HGB={a:sum(r['fit_delta_vs_HGB'][a]>0 for r in rows) for a in rows[0]['fit_delta_vs_HGB']})
    out.write_text(json.dumps(dict(summary=summary,folds=results,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Post-result exploratory fit-only metric decomposition on original1024bandwidth pairs. No refit/query selection. Dominant Euclidean units not proof of absent physiology or unique cause; kernel transform, regularization and finite landmark budget confounded.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
