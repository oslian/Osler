"""Variable-identity and within-component associations; no fitted correction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256


def variable_summary(frame, variable):
    mask = frame[f'observed_{variable}'].to_numpy(dtype=bool)
    age = frame[f'age_{variable}'].to_numpy()
    sizes = frame.groupby('component').size()
    weights = (1/frame.component.map(sizes)).to_numpy()
    result = dict(patients=len(frame), observed_patients=int(mask.sum()),
                  observed_fraction=float(np.average(mask, weights=weights)),
                  fresh_1h_fraction=float(np.average(mask & (age <= 1+1e-6), weights=weights)))
    part = frame.loc[mask].copy()
    counts = part.groupby('component').size()
    supported = counts.index[counts >= 30]
    part = part.loc[part.component.isin(supported)]
    result.update(supported_components=len(supported), retained_observed_patients=len(part))
    if len(supported) < 3:
        return dict(result, status='insufficient_support')
    value = f'value_{variable}'
    part['residual'] = part.truth - part.real
    part['relative_error'] = abs(part.truth-part.real) - abs(part.truth-part.process)
    fields = [value, 'truth', 'real', 'residual', 'relative_error']
    numbers = part[fields]
    if not np.isfinite(numbers.to_numpy()).all():
        raise ValueError('Nonfinite observed value or prediction')
    centered = numbers - numbers.groupby(part.component).transform('mean')
    w = (1/part.component.map(part.groupby('component').size())).to_numpy()
    x = centered[value].to_numpy()
    variance = np.average(x*x, weights=w)
    correlations = {}
    for field in fields[1:]:
        y = centered[field].to_numpy()
        vy = np.average(y*y, weights=w)
        correlations[field] = (float(np.average(x*y, weights=w)/np.sqrt(variance*vy))
                               if variance > 1e-20 and vy > 1e-20 else None)
    return dict(result, status='supported', correlations=correlations,
                observed_component_equal_value_mean=float(numbers.groupby(part.component).mean()[value].mean()))


def diagnose(source, outer, variables):
    if source.empty or outer.empty:
        return dict(status='empty_role', variables={})
    result = {}
    for v in variables:
        a = variable_summary(source, v)
        b = variable_summary(outer, v)
        item = dict(source=a, outer=b,
                    observation_fraction_shift=b['observed_fraction']-a['observed_fraction'],
                    fresh_fraction_shift=b['fresh_1h_fraction']-a['fresh_1h_fraction'])
        if a['status'] == b['status'] == 'supported':
            u = a['correlations']['residual']; w = b['correlations']['residual']
            if u is not None and w is not None:
                item['residual_correlation_sign_flip'] = bool(u*w < 0)
                item['both_residual_abs_correlation_at_least_point_one'] = bool(min(abs(u), abs(w)) >= .1)
        result[v] = item
    return dict(status='assessed', variables=result)


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    cp = Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    out = Path('artifacts/native_reconstruction_variable_transport_20260925.json')
    if out.exists():
        raise ValueError('Existing diagnosis')
    report = json.loads(rp.read_text()); cal = json.loads(cp.read_text())
    paths = {**report['bindings'], **cal['bindings']}
    for path, digest in paths.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed dependency')
    snapshot = root/'variable_snapshot_20260925.parquet'
    if sha256(snapshot) != report['private_sha256']:
        raise ValueError('Changed snapshots')
    predictions = {(fold, role): pd.read_parquet(root/f'{role}_20260924/fold{fold}.parquet')
                   for fold in range(5) for role in ('calibration', 'outer')}
    results = {str(fold): {} for fold in range(5)}
    for i, target in enumerate(cal['targets']):
        features = pd.read_parquet(snapshot, filters=[('target', '=', target)])
        if features.duplicated(['patient', 'target']).any():
            raise ValueError('Duplicate snapshot')
        for fold in range(5):
            frames = []
            for role in ('calibration', 'outer'):
                pred = predictions[fold, role]
                pred = pred.loc[pred.target == target]
                joined = pred.merge(features, on=['patient', 'target'], how='left', validate='one_to_one', indicator=True)
                if not (joined._merge == 'both').all():
                    raise ValueError('Missing snapshot')
                frames.append(joined)
            if set(frames[0].patient) & set(frames[1].patient) or set(frames[0].component) & set(frames[1].component):
                raise ValueError('Role overlap')
            results[str(fold)][target] = diagnose(*frames, report['variables'])
        print('Variable transport', i+1, 'of', len(cal['targets']), flush=True)
    comparisons = [v for f in results.values() for t in f.values() for v in t['variables'].values()]
    correlations = [v for v in comparisons if 'residual_correlation_sign_flip' in v]
    flips = [v for v in correlations if v['residual_correlation_sign_flip']]
    summary = dict(target_folds=5*len(cal['targets']), variable_comparisons=len(comparisons),
                   supported_residual_correlations=len(correlations), opposite_residual_correlation_signs=len(flips),
                   opposite_signs_both_absolute_at_least_point_one=sum(v['both_residual_abs_correlation_at_least_point_one'] for v in flips),
                   observation_fraction_shifts_at_least_point_two=sum(abs(v['observation_fraction_shift']) >= .2 for v in comparisons))
    out.write_text(json.dumps(dict(summary=summary, folds=results,
        bindings={str(p): sha256(p) for p in (rp, cp, snapshot, Path(__file__))},
        runtime_modified=False, promotion_allowed=False,
        limitation='Posthoc descriptive associations within component, equal component weighting. Correlation signs are not significance tests, causal effects, or isolated encoder sensitivity. No error-based feature deletion, calibration, threshold selection or new prediction head. Current masked reconstruction only.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
