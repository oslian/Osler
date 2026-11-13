"""Fixed source-only conditional correction of a frozen reconstruction state."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.calibrate_additional_reconstruction import fit_offsets
from scripts.score_additional_reconstruction import paired_summary
from whole_body_jepa.multisource_intervals import sha256

ARMS = ('unchanged', 'offset', 'process_context', 'numeric_context')


def weights(frame):
    result = (1/frame.component.map(frame.groupby('component').size())).to_numpy()
    return result/result.sum()


def features(source, query, variables):
    w = weights(source)
    selected = []
    for v in variables:
        seen = source.loc[source[f'observed_{v}']]
        if (seen.groupby('component').size() >= 30).sum() >= 3:
            selected.append(v)
    outputs = {}
    numeric = {}
    for role, frame in (('source', source), ('query', query)):
        blocks = [frame.real.to_numpy()[:, None]]
        values = []
        for v in selected:
            mask = frame[f'observed_{v}'].to_numpy(dtype=bool)
            srcmask = source[f'observed_{v}'].to_numpy(dtype=bool)
            srcvalue = source[f'value_{v}'].to_numpy()
            mean = np.average(srcvalue[srcmask], weights=w[srcmask])
            age = frame[f'age_{v}'].to_numpy()
            raw = frame[f'value_{v}'].to_numpy()
            if not np.isfinite(raw[mask]).all() or not np.isfinite(age[mask]).all() or (age[mask] < -1e-6).any():
                raise ValueError('Invalid observed snapshot')
            blocks.extend((mask[:, None].astype(float), np.log1p(np.where(mask, np.maximum(age, 0), 0))[:, None]))
            values.append(np.where(mask, raw, mean)[:, None])
        outputs[role] = np.concatenate(blocks, axis=1)
        numeric[role] = np.concatenate([outputs[role], *values], axis=1)
    return outputs, numeric, selected


def predict(source, query, variables):
    base = query.real.to_numpy().copy()
    result = {a: base.copy() for a in ARMS}
    offset = fit_offsets(source)
    info = dict(status=offset['status'], input_variables=[])
    if offset['status'] != 'fitted' or query.empty:
        return result, info
    result['offset'] += offset['offsets']['real']
    if source.fit_scale.nunique() != 1 or not (source.fit_scale > 0).all():
        raise ValueError('Inconsistent target scale')
    scale = float(source.fit_scale.iloc[0])
    y = (source.truth-source.real).to_numpy()/scale
    w = weights(source)
    process, numeric, selected = features(source, query, variables)
    for name, x in (('process_context', process), ('numeric_context', numeric)):
        scaler = StandardScaler().fit(x['source'], sample_weight=w)
        model = Ridge(alpha=.1, solver='cholesky').fit(scaler.transform(x['source']), y, sample_weight=w)
        result[name] += scale*model.predict(scaler.transform(x['query']))
        if not np.isfinite(result[name]).all():
            raise ValueError('Nonfinite corrected prediction')
    info['input_variables'] = selected
    return result, info


def source_cv(source, variables):
    rows = []
    for component, query in source.groupby('component', sort=True):
        fit = source.loc[source.component != component]
        prediction, info = predict(fit, query, variables)
        rows.append(dict(component=str(component), patients=len(query), status=info['status'],
                         mae={a: float(np.mean(abs(prediction[a]-query.truth.to_numpy()))) for a in ARMS}))
    return dict(complete=bool(rows) and all(r['status'] == 'fitted' for r in rows),
                components=rows,
                component_equal_mae={a: float(np.mean([r['mae'][a] for r in rows])) if rows else None for a in ARMS})


def score(frame, seed):
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    grouped = errors.groupby(frame.component).mean()
    normalized = errors.div(frame.fit_scale, axis=0)
    comparisons = {a: dict(patient=paired_summary(errors.numeric_context-errors[a], seed+i),
                           component=paired_summary(grouped.numeric_context-grouped[a], seed+100+i))
                   for i, a in enumerate(ARMS[:-1])}
    passed = len(frame) >= 30 and len(grouped) >= 3 and all(
        s['mean'] < 0 and s['p_less'] is not None and s['p_less'] < .05/47
        for c in comparisons.values() for s in c.values())
    return dict(patients=len(frame), components=len(grouped),
                raw_mae=dict(patient=errors.mean().to_dict(), component=grouped.mean().to_dict()),
                normalized_mae=dict(patient=normalized.mean().to_dict(), component=normalized.groupby(frame.component).mean().mean().to_dict()),
                comparisons=comparisons, incremental_descriptive_screen=bool(passed))


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    cp = Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    protocol = Path('RECONSTRUCTION_CONDITIONAL_REPAIR_PROTOCOL_20260925.md')
    snapshot = root/'variable_snapshot_20260925.parquet'
    out = Path('artifacts/native_reconstruction_conditional_repair_20260925.json')
    private = root/'conditional_repair_20260925.parquet'
    if out.exists() or private.exists():
        raise ValueError('Existing correction run')
    r = json.loads(rp.read_text()); c = json.loads(cp.read_text())
    for path, digest in {**r['bindings'], **c['bindings']}.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed dependency')
    if sha256(snapshot) != r['private_sha256']:
        raise ValueError('Changed snapshot')
    preds = {(f, k): pd.read_parquet(root/f'{k}_20260924/fold{f}.parquet')
             for f in range(5) for k in ('calibration', 'outer')}
    data = []; diagnostics = {}; results = {}
    with threadpool_limits(limits=4):
        for i, target in enumerate(c['targets']):
            snap = pd.read_parquet(snapshot, filters=[('target', '=', target)])
            parts = []; diagnostics[target] = {}
            for fold in range(5):
                frames = []
                for role in ('calibration', 'outer'):
                    p = preds[fold, role]; p = p.loc[p.target == target]
                    joined = p.merge(snap, on=['patient', 'target'], how='left', validate='one_to_one', indicator=True)
                    if not (joined._merge == 'both').all():
                        raise ValueError('Missing snapshot')
                    frames.append(joined)
                source, outer = frames
                if set(source.patient)&set(outer.patient) or set(source.component)&set(outer.component):
                    raise ValueError('Role overlap')
                cv = source_cv(source, r['variables'])
                predictions, info = predict(source, outer, r['variables'])
                diagnostics[target][str(fold)] = dict(fit=info, source_cv=cv)
                part = outer[['patient', 'target', 'component', 'fold', 'truth', 'fit_scale']].copy()
                for name, value in predictions.items():
                    part[name] = value
                parts.append(part)
            merged = pd.concat(parts, ignore_index=True)
            if merged.patient.duplicated().any():
                raise ValueError('Duplicate target patients')
            results[target] = score(merged, 20260925+1000*i)
            data.append(merged)
            print('Conditional correction', i+1, 'of', len(c['targets']), flush=True)
    data = pd.concat(data, ignore_index=True)
    if len(data) != 1800214 or data.duplicated(['patient', 'target']).any():
        raise ValueError('Incomplete outer coverage')
    data.to_parquet(private, index=False)
    summary = dict(pairs=len(data), patients=int(data.patient.nunique()), targets=len(results),
        incremental_descriptive_passes=[t for t, r in results.items() if r['incremental_descriptive_screen']],
        normalized_target_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in results.values()])) for a in ARMS} for level in ('patient', 'component')})
    out.write_text(json.dumps(dict(summary=summary, targets=results, fits=diagnostics,
        private_sha256=sha256(private), bindings={str(p): sha256(p) for p in (rp, cp, snapshot, protocol, Path(__file__), Path('scripts/calibrate_additional_reconstruction.py'), Path('scripts/score_additional_reconstruction.py'))},
        runtime_modified=False, promotion_allowed=False,
        limitation='Fixed source-only residual corrections on current masked reconstruction. Reused validation, overlapping crossfit fits, squared-loss ridge vs MAE. Unequal feature dimensions, not capacity-matched placebo. No encoder update, uncertainty calibration or forecast promotion.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
