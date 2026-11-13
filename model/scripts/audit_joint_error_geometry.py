"""Source-only bias/ranking/tail diagnosis; no deployable target-value strata."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/native_reconstruction_joint_error_geometry_20260918.json')
    if out.exists():raise ValueError('Existing audit')
    rp=Path('artifacts/native_reconstruction_joint_fit_gap_20260918.json')
    parent=json.loads(rp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Parent changed')
    source=Path('data/research_recovery/broad_source_20260917/joint_fit_gap.pt')
    rows=torch.load(source,weights_only=False);result={}
    for group,targets in parent['groups'].items():
        result[group]={}
        for t in targets:
            parts={r:[x for x in rows if x['group']==group and x['target']==t and x['role']==r]for r in ('fit','encoder_tune')}
            fy=np.array([r['truth']for r in parts['fit']]);lo,hi=np.quantile(fy,[.05,.95])
            scores={}
            for role,rr in parts.items():
                y=np.array([r['truth']for r in rr]);p=np.array([r['prediction']for r in rr]);base=np.array([r['population']for r in rr])
                delta=abs(p-y)-abs(base-y)
                rho=float(spearmanr(y,p).statistic)if np.std(y)>0 and np.std(p)>0 else None
                center=(y>=lo)&(y<=hi)
                scores[role]=dict(patients=len(y),truth_quantiles=np.quantile(y,[.05,.5,.95]).tolist(),
                    prediction_quantiles=np.quantile(p,[.05,.5,.95]).tolist(),spearman=rho,
                    mean_signed_error=float((p-y).mean()),median_signed_error=float(np.median(p-y)),
                    fraction_individual_harm=float((delta>0).mean()),median_error_increment=float(np.median(delta)),
                    fit_central_support=int(center.sum()),central_increment=float(delta[center].mean())if center.any()else None,
                    outside_fit_central_support=int((~center).sum()),outside_increment=float(delta[~center].mean())if (~center).any()else None)
                if not np.isclose(delta.mean(),targets[t][role]['increment'],atol=1e-10):raise ValueError('Increment mismatch')
            result[group][t]=scores
    out.write_text(json.dumps(dict(groups=result,bindings={str(p):sha256(p)for p in (Path(__file__),rp,source)},
        descriptive_only=True,runtime_modified=False,
        limitation='Truth-based strata are retrospective diagnostic only, never prediction-time routes. Selected tune is reused; correlations are not causal evidence.'),indent=2,allow_nan=False)+'\n')
    for group in result:
        for t in ('glucose','cpk','lipase','troponin_i','temperature'):
            r=result[group][t];print(group,t,'rank',r['fit']['spearman'],r['encoder_tune']['spearman'],
                'central increment',r['encoder_tune']['central_increment'],'individual harm',r['encoder_tune']['fraction_individual_harm'])


if __name__=='__main__':main()
