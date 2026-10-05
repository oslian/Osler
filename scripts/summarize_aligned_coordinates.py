"""Post-assessment failure localization from complete aggregate reports only."""
import json
from pathlib import Path

import numpy as np

from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def summarize(result, old):
    targets = sorted(result['targets'])
    if len(targets) != 47 or set(result['fit_checks']) != set(map(str, range(5))):
        raise ValueError('Incomplete family')
    fit_deltas = {a: [] for a in ('body_probe', 'raw_probe', 'random_axes', 'shuffled_axes')}
    retention = {role: {block: [] for block in ('latest', 'mean', 'process')}
                 for role in ('fit', 'selection')}
    for f in map(str, range(5)):
        if sorted(result['fit_checks'][f]) != targets or sorted(old['folds'][f]) != targets:
            raise ValueError('Incomplete fit target family')
        for t in targets:
            new = result['fit_checks'][f][t]
            refs = old['folds'][f][t]['fit']['normalized_mae']['patient']
            for a in fit_deltas:
                reference = new[a]['fit_mae'] if a in new else refs[a]
                fit_deltas[a].append(new['aligned_body']['fit_mae'] - reference)
            for role in retention:
                for block in retention[role]:
                    v = result['retention'][f][t][role][block]['median_mse_ratio']
                    if v is not None:
                        retention[role][block].append(v)
    selection = {}
    for a in fit_deltas:
        selection[a] = {level: dict(
            mean_better_targets=[t for t in targets if result['targets'][t]['normalized_mae'][level]['aligned_body']
                                < result['targets'][t]['normalized_mae'][level][a]],
            mean_worse_targets=[t for t in targets if result['targets'][t]['normalized_mae'][level]['aligned_body']
                               > result['targets'][t]['normalized_mae'][level][a]])
            for level in ('patient', 'component')}
    return dict(
        fit={a: dict(comparisons=len(v), improves=int((np.asarray(v) < 0).sum()),
                     worsens=int((np.asarray(v) > 0).sum()), mean_delta=float(np.mean(v)))
             for a, v in fit_deltas.items()},
        selection=selection,
        retention={r: {b: dict(target_fold_summaries=len(v),
                              median_of_median_mse_ratios=float(np.median(v)) if v else None,
                              below_fit_mean_count=sum(x < 1 for x in v))
                       for b, v in blocks.items()} for r, blocks in retention.items()},
        macro_relative_improvement_vs_body={level: 1 - rows['aligned_body']/rows['body_probe']
                                           for level, rows in result['summary']['normalized_target_macro'].items()})


def main():
    paths = [Path('artifacts/native_reconstruction_aligned_coordinates_assessment_20261003.json'),
             Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json'),
             Path('artifacts/native_reconstruction_aligned_coordinates_numerics_20261003.json')]
    reports, checked = [], {}
    for p in paths:
        r = json.loads(p.read_text())
        verify_bound_files(r['bindings'], checked)
        reports.append(r)
    if not reports[2]['summary']['all_checks_passed']:
        raise ValueError('Failed numerical audit')
    result = summarize(reports[0], reports[1])
    out = Path('artifacts/native_reconstruction_aligned_coordinates_diagnosis_20261003.json')
    if out.exists():
        raise ValueError('Existing diagnosis')
    paths.extend((Path(__file__), Path('scripts/score_monitored_continuation_outer.py')))
    out.write_text(json.dumps(dict(summary=result, bindings={str(p): sha256(p) for p in paths},
        runtime_modified=False, promotion_allowed=False,
        limitation='Post-result descriptive aggregate diagnosis. No new significance screen. '
                   'Fit target-fold errors are not independent replications. Retention summaries are '
                   'linear observation decoding relative to FIT means, not proof of invertibility or physiological truth.'),
        indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
