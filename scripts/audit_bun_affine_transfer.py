"""Fixed one-dimensional MAE calibration and rank diagnosis on cached pairs."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import QuantileRegressor
from scripts.audit_continued_selection_pairs import interval
from scripts.audit_bun_offset_transfer import offsets
from whole_body_jepa.multisource_intervals import sha256


def arrays(rows,arm):
    x=np.array([r['predictions'][arm] for r in rows],dtype=float)
    y=np.array([r['truth'] for r in rows],dtype=float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite nonempty pairs required')
    return x,y


def fit_affine(x,y):
    model=QuantileRegressor(quantile=.5,alpha=0,solver='highs').fit(x[:,None],y)
    return dict(slope=float(model.coef_[0]),intercept=float(model.intercept_))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    fp=root/'bun_dynamics_fit_20260920.pt';vp=root/'bun_offset_transfer_20260920.pt'
    report=Path('artifacts/native_reconstruction_bun_offset_transfer_20260920.json')
    out=Path('artifacts/native_reconstruction_bun_affine_transfer_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    prior=json.loads(report.read_text())
    if sha256(vp)!=prior['private_sha256']:raise ValueError('Selection cache changed')
    for p,h in prior['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed input')
    fit,tune=[torch.load(p,weights_only=False) for p in (fp,vp)]
    if {r['patient'] for rows in fit.values() for r in rows}&{r['patient'] for rows in tune.values() for r in rows}:
        raise ValueError('Patient overlap')
    coefficients={g:{a:fit_affine(*arrays(rows,a)) for a in ('full','no_body')} for g,rows in fit.items()}
    shifts={g:offsets(rows) for g,rows in fit.items()}
    result={}
    for g,rows in tune.items():
        comparisons={};distributions={}
        for arm in ('full','no_body'):
            x,y=arrays(rows,arm)
            distributions[arm]=dict(spearman=float(spearmanr(x,y).statistic),
                prediction_std=float(x.std()),truth_std=float(y.std()),
                prediction_quantiles=np.quantile(x,[.1,.5,.9]).tolist(),truth_quantiles=np.quantile(y,[.1,.5,.9]).tolist())
        for training,c in coefficients.items():
            arms={}
            for arm in ('full','no_body'):
                x,y=arrays(rows,arm);base=abs(x-y);shifted=abs(x+shifts[training][arm]-y)
                corrected=abs(c[arm]['slope']*x+c[arm]['intercept']-y)
                arms[arm]=dict(mae=float(corrected.mean()),vs_original=interval(corrected-base),vs_offset=interval(corrected-shifted))
            comparisons[training]=arms
        result[g]=dict(patients=len(rows),distributions=distributions,by_fit_group=comparisons)
    output=dict(coefficients=coefficients,groups=result,runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),fp,vp,report)},
        limitation='Unpenalized median affine regression fit on source-fit only, both source fits cross-applied without selection. Reused selection data and unadjusted descriptive intervals, no runtime correction or new external validation. Correlation and marginal spread do not prove recoverable latent information.')
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
