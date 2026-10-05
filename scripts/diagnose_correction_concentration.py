"""Descriptive component influence audit; never a promotion or selection rule."""
import json
from pathlib import Path

import numpy as np

from whole_body_jepa.multisource_intervals import sha256


def summarize_components(rows):
    delta = np.asarray([r['mae'] - r['reference_mae']['unchanged'] for r in rows])
    sizes = np.asarray([r['patients'] for r in rows])
    if len(delta) < 2 or not np.isfinite(delta).all() or (sizes <= 0).any():
        raise ValueError('Need at least two finite, nonempty components')
    supported = sizes >= 30
    return dict(
        components=len(rows), mean_delta=float(delta.mean()),
        majority_improves=bool(np.sum(delta < 0) > len(delta) / 2),
        worst_leave_one_out_delta=float(np.max((delta.sum() - delta) / (len(delta) - 1))),
        every_component_improves=bool((delta < 0).all()),
        small_components=int((~supported).sum()),
        supported_only_delta=float(delta[supported].mean()) if supported.any() else None,
    )


def main():
    source = Path('artifacts/native_reconstruction_point_only_20260925.json')
    report = json.loads(source.read_text())
    records = []
    for target, folds in report['targets'].items():
        for fold, record in folds.items():
            if record['matched']:
                records.append(dict(target=target, fold=fold, **summarize_components(record['components'])))
    winners = [r for r in records if r['mean_delta'] < 0]
    result = dict(
        summary=dict(matched=len(records), mean_wins=len(winners),
                     majority_wins=sum(r['majority_improves'] for r in winners),
                     leave_any_one_out_wins=sum(r['worst_leave_one_out_delta'] < 0 for r in winners),
                     all_component_wins=sum(r['every_component_improves'] for r in winners),
                     small_component_sign_sensitivity=sum(
                         (r['mean_delta'] < 0) != (r['supported_only_delta'] < 0)
                         for r in records if r['supported_only_delta'] is not None)),
        records=records, bindings={str(source): sha256(source), __file__: sha256(Path(__file__))},
        promotion_allowed=False, runtime_modified=False,
        limitation='Source-CV descriptive influence analysis, not independent replications or causal attribution. Supported-only is sensitivity, not a replacement cohort or gate. No refitting or outer evaluation.',
    )
    destination = Path('artifacts/native_reconstruction_concentration_20260925.json')
    if destination.exists():
        raise ValueError('Existing concentration audit')
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
