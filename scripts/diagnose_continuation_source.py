"""Source warning and monitor fidelity audit; no source selection or fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256


def source_harms(frame, roles):
    if frame.duplicated(['patient','target']).any() or not all(roles[p]=='selection' for p in frame.patient):
        raise ValueError('Invalid source query identity or role')
    if not np.isfinite(frame[['real','process','population','truth','fit_scale']]).all().all() or not frame.fit_scale.gt(0).all():
        raise ValueError('Invalid source predictions')
    errors=frame[['real','process','population']].subtract(frame.truth,axis=0).abs().div(frame.fit_scale,axis=0)
    groups=errors.groupby([frame.target,frame.component]).mean()
    counts=frame.groupby(['target','component']).size()
    result={}
    for target,part in groups.groupby(level=0):
        supported=counts.loc[target]>=30
        result[target]={c:dict(components=len(part),supported_components=int(supported.sum()),
            mean_harm_components=int((part.real>part[c]).sum()),
            supported_mean_harm_components=int(((part.real>part[c]).to_numpy() & supported.to_numpy()).sum()))
            for c in ('process','population')}
    return result


def compare_views(full8, monitor8, monitor10, outer8, outer10):
    arrays = [np.asarray(x, dtype=float) for x in
              (full8, monitor8, monitor10, outer8, outer10)]
    if any(x.shape != (5, 3) or np.isinf(x).any() or (x < 0).any()
           or np.any(np.isnan(x).any(axis=1) != np.isnan(x).all(axis=1)) for x in arrays):
        raise ValueError('Expected nonnegative five-fold errors or wholly unsupported rows')
    f, m8, m10, o8, o10 = arrays
    if not all(np.isfinite(x).all() for x in (f,m8,m10)):
        raise ValueError('Incomplete source evidence')
    if not np.array_equal(m8[:, 2], m10[:, 2]) or not np.array_equal(o8[:, 2], o10[:, 2],equal_nan=True):
        raise ValueError('Changed population reference')
    def nullable(x):return [float(v) if np.isfinite(v) else None for v in x]
    views = dict(full_source8=f, monitor8=m8, monitor10=m10, outer8=o8, outer10=o10)
    records = {}
    for control, col in (('process', 1), ('population', 2)):
        margins = {name: nullable(x[:, 0]-x[:, col]) for name, x in views.items()}
        records[control] = dict(margins=margins,
            full_source8_favorable_folds=int(np.sum(f[:, 0] < f[:, col])),
            monitor10_favorable_folds=int(np.sum(m10[:, 0] < m10[:, col])),
            full_source8_monitor8_sign_disagreements=int(np.sum(
                (f[:, 0] < f[:, col]) != (m8[:, 0] < m8[:, col]))))
    monitor_delta = m10[:, 0]-m8[:, 0]
    outer_delta = o10[:, 0]-o8[:, 0]
    valid=np.isfinite(outer_delta)
    return dict(comparisons=records, monitor_real_change=monitor_delta.tolist(),
        outer_real_change=nullable(outer_delta), supported_change_comparisons=int(valid.sum()),
        change_sign_disagreements=int(np.sum(((monitor_delta < 0) != (outer_delta < 0)) & valid)))


def outer_errors(assessment, target, level):
    groups = assessment['targets'][target]['base_assessment']['per_component']
    rows = []
    for fold in range(5):
        selected = [g for g in groups if g['fold'] == fold]
        if not selected:
            rows.append([np.nan]*3)
            continue
        weights = [g['patients'] if level == 'patient' else 1 for g in selected]
        rows.append([float(np.average([g['normalized_mae'][a] for g in selected], weights=weights))
                     for a in ('real', 'process', 'population')])
    return rows


def main():
    paths = {name: Path('artifacts') / filename for name, filename in dict(
        source='native_reconstruction_coverage_selection_summary_20260927.json',
        monitor='native_reconstruction_fixed_monitor_epoch8_20260929.json',
        continuation='native_reconstruction_monitored_continuation_summary_20260929.json',
        old='native_reconstruction_coverage_outer_evaluation_20260929.json',
        new='native_reconstruction_continuation_outer_20260929.json').items()}
    reports = {name: json.loads(p.read_text()) for name, p in paths.items()}
    checked = {}
    for report in reports.values():
        for name, digest in report['bindings'].items():
            if name not in checked:
                checked[name] = sha256(Path(name))
            if checked[name] != digest:
                raise ValueError('Changed dependency: '+name)
    source = {r['fold']: r for r in reports['source']['folds']}
    root=Path('data/research_recovery/additional_development_domains_20260921')
    plan_path=root/'partitions_20260922.pt'
    if sha256(plan_path)!=reports['monitor']['bindings'][str(plan_path)]:
        raise ValueError('Changed source roles')
    plan=torch.load(plan_path,weights_only=False)
    harms={}
    for fold in range(5):
        frame=pd.read_parquet(root/f'coverage_selection_20260926/fold{fold}.parquet')
        if not frame.fold.eq(fold).all():raise ValueError('Wrong source fold')
        harms[fold]=source_harms(frame,plan['folds'][fold])
        errors=frame[['real','process','population']].subtract(frame.truth,axis=0).abs().div(frame.fit_scale,axis=0)
        for level,metrics in (
            ('patient',errors.groupby(frame.target).mean()),
            ('component',errors.groupby([frame.target,frame.component]).mean().groupby(level=0).mean())):
            for target,row in metrics.iterrows():
                for arm in ('real','process','population'):
                    if not np.isclose(row[arm],source[fold][f'{level}_target_errors'][target][arm+'_error'],rtol=1e-12,atol=1e-12):
                        raise ValueError('Source aggregate reproduction failed')
    targets = sorted(reports['new']['targets'])
    if len(targets) != 47 or targets != sorted(reports['old']['targets']):
        raise ValueError('Changed target family')
    folder = Path('data/research_recovery/additional_development_domains_20260921/monitored_continuation_20260929')
    latest = {}
    for fold in range(5):
        for arm in ('real', 'process'):
            p = folder / f'fold{fold}_{arm}_epoch10_monitor.json'
            if sha256(p) != reports['continuation']['bindings'][str(p)]:
                raise ValueError('Changed final monitor')
            latest[fold, arm] = json.loads(p.read_text())['targets']
    results = {}
    for target in targets:
        results[target] = dict(base_screen=reports['new']['targets'][target]['base_screen'],
            source8_component_harms={str(i):harms[i][target] for i in range(5)})
        for level in ('patient', 'component'):
            metric = f'normalized_{level}_mae'
            f = [[source[i][f'{level}_target_errors'][target][a+'_error']
                  for a in ('real', 'process', 'population')] for i in range(5)]
            m8 = [[reports['monitor']['folds'][str(i)][a][target][metric]
                   for a in ('real', 'process', 'population')] for i in range(5)]
            m10 = [[latest[i, a][target][metric] for a in ('real', 'process')]+[m8[i][2]]
                   for i in range(5)]
            results[target][level] = compare_views(f, m8, m10,
                outer_errors(reports['old'], target, level), outer_errors(reports['new'], target, level))
    summary = {}
    for level in ('patient', 'component'):
        summary[level] = dict(
            monitor_outer_change_sign_disagreements=sum(r[level]['change_sign_disagreements'] for r in results.values()),
            nominal_target_fold_comparisons=235,
            supported_change_comparisons=sum(r[level]['supported_change_comparisons'] for r in results.values()),
            by_control={c: dict(
                full_source8_monitor8_sign_disagreements=sum(
                    r[level]['comparisons'][c]['full_source8_monitor8_sign_disagreements'] for r in results.values()),
                failed_targets_source8_all_folds_favorable=[t for t,r in results.items()
                    if not r['base_screen'] and r[level]['comparisons'][c]['full_source8_favorable_folds'] == 5],
                failed_targets_source8_and_monitor10_all_folds_favorable=[t for t,r in results.items()
                    if not r['base_screen'] and r[level]['comparisons'][c]['full_source8_favorable_folds'] == 5
                    and r[level]['comparisons'][c]['monitor10_favorable_folds'] == 5])
                for c in ('process', 'population')})
    summary['source8_supported_harm']={c:dict(
        failed_targets_with_any_source_fold_harm=[t for t,r in results.items() if not r['base_screen']
            and any(h[c]['supported_mean_harm_components']>0 for h in r['source8_component_harms'].values())],
        failed_targets_with_every_source_fold_harm=[t for t,r in results.items() if not r['base_screen']
            and all(h[c]['supported_mean_harm_components']>0 for h in r['source8_component_harms'].values())])
        for c in ('process','population')}
    out = Path('artifacts/native_reconstruction_continuation_source_20260930.json')
    if out.exists():raise ValueError('Existing diagnostic')
    out.write_text(json.dumps(dict(summary=summary, targets=results,
        bindings={str(p):sha256(p) for p in (*paths.values(),Path(__file__))},
        runtime_modified=False,promotion_allowed=False,
        limitation='Descriptive signs, not efficacy gates. Source/outer cohorts differ; monitor samples at most32 per component. Fold fits and source queries overlap, not235 independent experiments. No full source epoch10 inference, no trained selector or outer-derived routing. Monitor did not select continuation endpoint.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':main()
