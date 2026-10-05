"""Descriptive domain failure localization, never an outer-fitted repair."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def domain_summary(frame):
    rows = []
    for component, g in frame.groupby('component', sort=True):
        row = dict(component=str(component), patients=len(g))
        for arm in ('real', 'process', 'population'):
            residual = g[arm] - g.truth
            row[arm] = dict(mae=float(residual.abs().mean()),
                mean_bias=float(residual.mean()), median_bias=float(residual.median()))
        rows.append(row)
    comparisons = {}
    for arm in ('process', 'population'):
        d = np.array([r['real']['mae'] - r[arm]['mae'] for r in rows])
        support = np.array([r['patients'] for r in rows])
        supported = support >= 30
        loo = (d.sum() - d) / (len(d) - 1) if len(d) > 1 else np.array([])
        comparisons[arm] = dict(
            all_component_mean=float(d.mean()),
            supported_component_mean=float(d[supported].mean()) if supported.any() else None,
            harmed_components=int((d > 0).sum()),
            harmed_supported_components=int(((d > 0) & supported).sum()),
            supported_components=int(supported.sum()),
            leave_one_component_mean_range=[float(loo.min()), float(loo.max())] if len(loo) else None)
    bias = np.array([r['real']['median_bias'] for r in rows])
    return dict(components=rows, comparisons=comparisons,
        positive_median_bias_components=int((bias > 0).sum()),
        negative_median_bias_components=int((bias < 0).sum()))


def main():
    source = Path('artifacts/native_reconstruction_additional_evaluation_20260924.json')
    report = json.loads(source.read_text())
    for name, digest in report['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed assessment dependency')
    frames = [pd.read_parquet(p) for p in report['bindings'] if p.endswith('.parquet') and '/outer_' in p]
    if len(frames) != 5:
        raise ValueError('Incomplete outer folds')
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['patient', 'target']).any():
        raise ValueError('Duplicate patient-target')
    results = {t:domain_summary(g) for t, g in data.groupby('target', sort=True)}
    out = Path('artifacts/native_reconstruction_additional_domain_diagnosis_20260924.json')
    if out.exists():
        raise ValueError('Existing diagnosis')
    out.write_text(json.dumps(dict(targets=results, bindings={str(source):sha256(source),
        str(Path(__file__)):sha256(Path(__file__))}, runtime_modified=False, promotion_allowed=False,
        limitation='Posthoc descriptive sensitivity only. No component exclusion, bias correction, feature selection or gate change. Bias sign heterogeneity does not establish clinical cause or source-only calibration benefit.'), indent=2, allow_nan=False)+'\n')
    for t in ('map','sodium','glucose','heart_rate','bicarbonate','creatinine'):
        r=results[t]
        print(t,json.dumps({k:v for k,v in r.items() if k!='components'}),flush=True)


if __name__ == '__main__':main()
