"""Audit training-weight concentration, not statistical information content."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256


def concentration(sizes):
    sizes = np.asarray(sizes, dtype=float)
    if not len(sizes) or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Positive component sizes required')
    ess = len(sizes) ** 2 / np.sum(1 / sizes)
    return dict(patients=int(sizes.sum()), components=len(sizes),
                small_components=int(np.sum(sizes < 30)),
                small_component_weight_mass=float(np.mean(sizes < 30)),
                kish_weight_ess=float(ess), ess_ratio=float(ess / sizes.sum()),
                max_to_min_patient_weight=float(sizes.max() / sizes.min()))


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921/calibration_20260924')
    records, bindings = [], {}
    for fold in range(5):
        path = root / f'fold{fold}.parquet'
        bindings[str(path)] = sha256(path)
        frame = pd.read_parquet(path, columns=['patient', 'target', 'component'])
        for target, group in frame.groupby('target'):
            if group.patient.duplicated().any():
                raise ValueError('Repeated patient within target-role')
            records.append(dict(target=target, fold=fold,
                                **concentration(group.groupby('component').size())))
    result = dict(records=records, bindings=bindings,
                  nonempty_roles=len(records),
                  roles_with_small_components=sum(r['small_components'] > 0 for r in records),
                  median_ess_ratio=float(np.median([r['ess_ratio'] for r in records])),
                  runtime_modified=False, promotion_allowed=False,
                  limitation='Kish weight ESS describes weight concentration, not independent physiological sample size, estimator variance or a proven cause of error. Full calibration roles, not each leave-component-out refit.')
    output = Path('artifacts/native_reconstruction_weight_concentration_20260925.json')
    if output.exists():
        raise ValueError('Existing weight audit')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'bindings')}, indent=2))


if __name__ == '__main__':
    main()
