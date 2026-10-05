"""Compare complete frozen source predictions; descriptive, never a selector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_continuation_source import outer_errors
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


ARMS = ('real8', 'process8', 'real10', 'process10', 'population')


def summarize_fold(base, fresh):
    pd.testing.assert_frame_equal(base[['patient', 'target', 'fold']],
                                  fresh[['patient', 'target', 'fold']])
    if base.duplicated(['patient', 'target']).any():
        raise ValueError('Duplicate patient-target pairs')
    if (not np.isfinite(base[['real', 'process', 'population', 'truth', 'fit_scale']]).all().all()
            or not base.fit_scale.gt(0).all()
            or not np.isfinite(fresh[['real10', 'process10']]).all().all()):
        raise ValueError('Invalid prediction or normalization')
    frame = base.rename(columns={'real': 'real8', 'process': 'process8'}).copy()
    for arm in ('real10', 'process10'):
        frame[arm] = fresh[arm].to_numpy()
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs().div(frame.fit_scale, axis=0)
    results = {}
    for target, indices in frame.groupby('target', sort=True).groups.items():
        part = frame.loc[indices]
        error = errors.loc[indices]
        groups = error.groupby(part.component).mean()
        counts = part.groupby('component').size()
        harms = {}
        for control in ('process', 'population'):
            old_control = 'process8' if control == 'process' else 'population'
            new_control = 'process10' if control == 'process' else 'population'
            old_harm = groups.real8 > groups[old_control]
            new_harm = groups.real10 > groups[new_control]
            harms[control] = {
                'old': int(old_harm.sum()), 'new': int(new_harm.sum()),
                'recurring': int((old_harm & new_harm).sum()),
                'supported_recurring': int((old_harm & new_harm & (counts >= 30)).sum()),
            }
        results[target] = {
            'patients': len(part), 'components': len(groups),
            'normalized_mae': {'patient': error.mean().to_dict(),
                               'component': groups.mean().to_dict()},
            'component_harms': harms,
            'per_component': [dict(component=str(name), patients=int(counts[name]),
                                   normalized_mae=row.to_dict())
                              for name, row in groups.iterrows()],
        }
    return results


def change_agreement(full8, full10, monitor8, monitor10, outer8, outer10):
    arrays = [np.asarray(values, dtype=float) for values in
              (full8, full10, monitor8, monitor10, outer8, outer10)]
    if any(x.shape != (5,) or np.isinf(x).any() for x in arrays):
        raise ValueError('Expected five-fold metrics')
    if not all(np.isfinite(x).all() for x in arrays[:4]):
        raise ValueError('Incomplete source evidence')
    a, b, c, d, e, f = arrays
    if not np.array_equal(np.isnan(e), np.isnan(f)):
        raise ValueError('Changed outer support')
    full_delta = b-a
    monitor_delta = d-c
    outer_delta = f-e
    supported = np.isfinite(outer_delta)
    return {
        'full_source_real_change': full_delta.tolist(),
        'monitor_real_change': monitor_delta.tolist(),
        'outer_real_change': [float(x) if np.isfinite(x) else None for x in outer_delta],
        'full_monitor_sign_disagreements': int(((full_delta < 0) != (monitor_delta < 0)).sum()),
        'full_outer_sign_disagreements': int((((full_delta < 0) != (outer_delta < 0)) & supported).sum()),
        'supported_outer_comparisons': int(supported.sum()),
    }


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    output = Path('artifacts/native_reconstruction_full_continuation_selection_20260930.json')
    if output.exists():
        raise ValueError('Existing assessment')
    paths = [Path(__file__), Path('scripts/diagnose_continuation_source.py'),
             Path('scripts/score_monitored_continuation_outer.py')]
    report_names = {
        'source8': 'native_reconstruction_coverage_selection_summary_20260927.json',
        'monitor8': 'native_reconstruction_fixed_monitor_epoch8_20260929.json',
        'monitor10': 'native_reconstruction_monitored_continuation_summary_20260929.json',
        'outer8': 'native_reconstruction_coverage_outer_evaluation_20260929.json',
        'outer10': 'native_reconstruction_continuation_outer_20260929.json',
    }
    reports, checked = {}, {}
    for name, filename in report_names.items():
        path = Path('artifacts') / filename
        reports[name] = json.loads(path.read_text())
        verify_bound_files(reports[name]['bindings'], checked)
        paths.append(path)
    old_source = {row['fold']: row for row in reports['source8']['folds']}
    folds, monitors = {}, {}
    for fold in range(5):
        proof_path = Path(f'artifacts/native_reconstruction_continuation_selection_fold{fold}_verification_20260930.json')
        proof = json.loads(proof_path.read_text())
        if proof['fold'] != fold or proof['promotion_allowed'] is not False:
            raise ValueError('Invalid verification contract')
        verify_bound_files(proof['bindings'], checked)
        base_path = root / f'coverage_selection_20260926/fold{fold}.parquet'
        fresh_path = root / f'continuation_selection_20260930/fold{fold}.parquet'
        base, fresh = pd.read_parquet(base_path), pd.read_parquet(fresh_path)
        fold_results = summarize_fold(base, fresh)
        if len(fold_results) != 47 or sum(row['patients'] for row in fold_results.values()) != proof['pairs']:
            raise ValueError('Incomplete target family')
        for target, result in fold_results.items():
            for level in ('patient', 'component'):
                for old_arm, arm in (('real', 'real8'), ('process', 'process8'), ('population', 'population')):
                    expected = old_source[fold][level+'_target_errors'][target][old_arm+'_error']
                    if not np.isclose(result['normalized_mae'][level][arm], expected, rtol=1e-12, atol=1e-12):
                        raise ValueError('Original source aggregate mismatch')
        folds[str(fold)] = fold_results
        monitor_path = root / f'monitored_continuation_20260929/fold{fold}_real_epoch10_monitor.json'
        if sha256(monitor_path) != reports['monitor10']['bindings'][str(monitor_path)]:
            raise ValueError('Changed final monitor')
        monitors[fold] = json.loads(monitor_path.read_text())['targets']
        paths.extend((proof_path, base_path, fresh_path, monitor_path))
        print('Verified complete source fold', fold, len(fresh), flush=True)
    targets = sorted(reports['outer10']['targets'])
    if len(targets) != 47 or any(sorted(rows) != targets for rows in folds.values()):
        raise ValueError('Inconsistent target family')
    changes = {}
    for target in targets:
        changes[target] = {}
        for level in ('patient', 'component'):
            key = f'normalized_{level}_mae'
            changes[target][level] = change_agreement(
                [folds[str(i)][target]['normalized_mae'][level]['real8'] for i in range(5)],
                [folds[str(i)][target]['normalized_mae'][level]['real10'] for i in range(5)],
                [reports['monitor8']['folds'][str(i)]['real'][target][key] for i in range(5)],
                [monitors[i][target][key] for i in range(5)],
                [row[0] for row in outer_errors(reports['outer8'], target, level)],
                [row[0] for row in outer_errors(reports['outer10'], target, level)])
    failed = [t for t in targets if not reports['outer10']['targets'][t]['base_screen']]
    summary = {
        'fold_macro': {fold: {level: {arm: float(np.mean([
            row['normalized_mae'][level][arm] for row in rows.values()])) for arm in ARMS}
            for level in ('patient', 'component')} for fold, rows in folds.items()},
        'change_agreement': {level: {key: sum(row[level][key] for row in changes.values())
            for key in ('full_monitor_sign_disagreements', 'full_outer_sign_disagreements',
                        'supported_outer_comparisons')} for level in ('patient', 'component')},
        'outer_failed_targets_with_source_supported_recurring_harm': {
            control: [t for t in failed if any(
                rows[t]['component_harms'][control]['supported_recurring'] > 0 for rows in folds.values())]
            for control in ('process', 'population')},
        'outer_failed_targets_with_no_source_supported_recurring_harm': [t for t in failed if not any(
            rows[t]['component_harms'][control]['supported_recurring'] > 0
            for rows in folds.values() for control in ('process', 'population'))],
    }
    report = dict(summary=summary, folds=folds, changes=changes,
                  bindings={str(path): sha256(path) for path in paths},
                  runtime_modified=False, promotion_allowed=False,
                  limitation='All 47 current-value reconstruction targets. Descriptive means/signs, not statistical harm or promotion. Source query populations and fitted models overlap; do not treat folds as independent replications or pool them as unique patients. Monitor sampling and source/outer population differences are distinct. Outer data are reused development. All components retained; >=30 support only labels a diagnostic subgroup, never a gate exclusion. No model fitting or outcome-derived routing.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
