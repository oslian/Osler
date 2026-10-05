"""Conditional paired-error variance decomposition; not a promotion test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256
from scripts.score_additional_reconstruction import paired_summary


def decompose(means, mean_variances):
    means=np.asarray(means,float); variances=np.asarray(mean_variances,float)
    if len(means)<2 or means.shape!=variances.shape or not np.isfinite(means).all():
        raise ValueError('Invalid group means')
    spread=float(np.var(means,ddof=1))
    if not np.isfinite(variances).all():
        return dict(observed_between_variance=spread,estimated_sampling_variance=None,
                    excess_variance=None,sampling_fraction=None)
    noise=float(variances.mean())
    return dict(observed_between_variance=spread,estimated_sampling_variance=noise,
                excess_variance=spread-noise,sampling_fraction=noise/spread if spread>0 else None)


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    frames=[];bindings={}
    for fold in range(5):
        path=root/f'coverage_outer_20260927/fold{fold}.parquet'
        meta=json.loads(path.with_suffix('.json').read_text())
        if sha256(path)!=meta['prediction_sha256']:raise ValueError('Changed predictions')
        frames.append(pd.read_parquet(path));bindings[str(path)]=sha256(path)
    data=pd.concat(frames,ignore_index=True)
    if data.duplicated(['patient','target']).any():raise ValueError('Repeated patient target')
    assessment=json.loads(Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json').read_text())
    targets={}
    for i,(target,part) in enumerate(data.groupby('target',sort=True)):
        comparisons={}
        for j,control in enumerate(('process','population')):
            d=(abs(part.real-part.truth)-abs(part[control]-part.truth))/part.fit_scale
            groups=[]
            for k,(component,values) in enumerate(d.groupby(part.component,sort=True)):
                stats=paired_summary(values.to_numpy(),20260929+10000*i+1000*j+k,repetitions=1000)
                groups.append(dict(component=str(component),patients=len(values),mean=stats['mean'],
                    mean_variance=float(values.var(ddof=1)/len(values)) if len(values)>1 else None,
                    nominal_bootstrap_ci95=stats['ci95']))
            comparisons[control]=dict(decomposition=decompose([g['mean'] for g in groups],
                [np.nan if g['mean_variance'] is None else g['mean_variance'] for g in groups]),
                groups=groups,unsupported_variance_groups=sum(g['mean_variance'] is None for g in groups),
                supported_nominal_harm_groups=sum(g['patients']>=30 and g['nominal_bootstrap_ci95'] is not None and g['nominal_bootstrap_ci95'][0]>0 for g in groups),
                supported_nominal_benefit_groups=sum(g['patients']>=30 and g['nominal_bootstrap_ci95'] is not None and g['nominal_bootstrap_ci95'][1]<0 for g in groups))
        targets[target]=dict(base_screen=assessment['targets'][target]['base_screen'],comparisons=comparisons)
        print('Variance',target,flush=True)
    summary={}
    for c in ('process','population'):
        records=[r['comparisons'][c] for r in targets.values() if not r['base_screen']]
        valid=[r['decomposition'] for r in records if r['decomposition']['sampling_fraction'] is not None]
        summary[c]=dict(failed_targets=len(records),decomposable=len(valid),
            median_sampling_fraction=float(np.median([v['sampling_fraction'] for v in valid])),
            positive_excess_variance=sum(v['excess_variance']>0 for v in valid),
            targets_with_supported_nominal_harm=sum(r['supported_nominal_harm_groups']>0 for r in records))
    output=dict(summary=summary,targets=targets,bindings=bindings,runtime_modified=False,promotion_allowed=False,
        limitation='Conditional descriptive estimate assumes independent patient errors within components. Shared fitted weights and overlapping training can induce dependence; variance subtraction is not a causal or definitive heterogeneity estimate. Singleton groups retain unknown noise. Nominal subgroup CIs are not multiplicity-adjusted discoveries. No groups removed.')
    Path('artifacts/native_reconstruction_coverage_variance_20260929.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(summary=summary,map=targets['map']['comparisons']['process']['decomposition']),indent=2))


if __name__=='__main__':main()
