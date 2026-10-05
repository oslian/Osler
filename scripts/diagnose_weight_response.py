"""Descriptive weight sensitivity; correlated folds preclude naive p-values."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from whole_body_jepa.multisource_intervals import sha256


def main():
    paths = [Path('artifacts/native_reconstruction_patient_equal_20260925.json'),
             Path('artifacts/native_reconstruction_weight_concentration_20260925.json')]
    experiment, weights = [json.loads(p.read_text()) for p in paths]
    lookup = {(r['target'], str(r['fold'])): r for r in weights['records']}
    rows = [r for r in experiment['records'] if r['matched']]
    x = [lookup[(r['target'], r['fold'])]['ess_ratio'] for r in rows]
    y = [(r['means']['component']['mae']-r['means']['component']['balanced_mae']) /
         r['means']['component']['balanced_mae'] for r in rows]
    bins = []
    for lo, hi in ((0, 30), (30, 300), (300, float('inf'))):
        components = [c for r in rows for c in r['components'] if lo <= c['patients'] < hi]
        bins.append(dict(min_patients=lo, max_patients_exclusive=hi if np.isfinite(hi) else None,
                         comparisons=len(components),
                         harms=sum(c['mae']>c['balanced_mae'] for c in components),
                         median_relative_delta=float(np.median([
                             (c['mae']-c['balanced_mae'])/c['balanced_mae'] for c in components]))))
    report = dict(rank_correlation=float(spearmanr(x, y).statistic), matched=len(rows),
                  size_bins=bins, bindings={str(p): sha256(p) for p in paths},
                  limitation='Exploratory descriptive association only. Folds and component queries overlap; no independence-based p-value or causal claim. ESS from full calibration roles, not per-query training refits. No model selection.',
                  runtime_modified=False, promotion_allowed=False)
    path = Path('artifacts/native_reconstruction_weight_response_20260925.json')
    if path.exists():
        raise ValueError('Existing response audit')
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
