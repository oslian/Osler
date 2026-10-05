"""Descriptive component influence, never outcome-driven cohort exclusion."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def influence(groups, control):
    d=np.asarray([g[f'real_minus_{control}'] for g in groups],float)
    n=np.asarray([g['patients'] for g in groups],int)
    if len(d)<2 or not np.isfinite(d).all() or (n<=0).any():
        raise ValueError('Invalid component comparison')
    leave=(d.sum()-d)/(len(d)-1)
    large=n>=30
    return dict(components=len(d),mean=float(d.mean()),patient_weighted_mean=float(np.average(d,weights=n)),
        median=float(np.median(d)),fraction_better=float(np.mean(d<0)),
        leave_one_mean_min=float(leave.min()),leave_one_mean_max=float(leave.max()),
        leave_one_sign_changes=int(np.sum((leave<0)!=(d.mean()<0))),
        small_components=int((~large).sum()),small_component_mean=float(d[~large].mean()) if (~large).any() else None,
        supported_component_mean=float(d[large].mean()) if large.any() else None,
        supported_fraction_better=float(np.mean(d[large]<0)) if large.any() else None)


def main():
    path=Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json')
    source=json.loads(path.read_text());rows={}
    for target,r in source['targets'].items():
        rows[target]=dict(base_screen=r['base_screen'],
            **{c:influence(r['base_assessment']['per_component'],c) for c in ('process','population')})
    failed={t:r for t,r in rows.items() if not r['base_screen']}
    summary={}
    for control in ('process','population'):
        v=[r[control] for r in failed.values()]
        summary[control]=dict(failed_targets=len(v),majority_components_better=sum(x['fraction_better']>.5 for x in v),
            negative_mean_robust_to_any_single_removal=sum(x['leave_one_mean_max']<0 for x in v),
            at_least_one_leave_one_sign_flip=sum(x['leave_one_sign_changes']>0 for x in v),
            supported_mean_sign_differs=sum(x['supported_component_mean'] is not None and (x['supported_component_mean']<0)!=(x['mean']<0) for x in v))
    dest=Path('artifacts/native_reconstruction_coverage_component_influence_20260929.json')
    dest.write_text(json.dumps(dict(summary=summary,targets=rows,source_sha256=sha256(path),
        promotion_allowed=False,runtime_modified=False,
        limitation='Posthoc sensitivity only. No groups excluded from scoring or promotion. Supported means use existing30-person descriptive threshold; no significance claims.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summary,map=rows['map']),indent=2))


if __name__=='__main__':main()
