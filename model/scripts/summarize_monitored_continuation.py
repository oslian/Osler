"""Summarize the fixed source monitors without selecting a model epoch."""

import json
from pathlib import Path

import numpy as np

from whole_body_jepa.multisource_intervals import sha256


METRICS = ('normalized_patient_mae', 'normalized_component_mae', 'smooth_l1')


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    folder = root / 'monitored_continuation_20260929'
    baseline = Path('artifacts/native_reconstruction_fixed_monitor_epoch8_20260929.json')
    baseline_data = json.loads(baseline.read_text())
    certificate = Path('artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json')
    proof = json.loads(certificate.read_text())
    if not proof['fit_identity_equal'] or not proof['scalers_equal']:
        raise ValueError('Uncertified fold-2 reuse')
    paths = [Path(__file__), baseline, certificate]
    reports = {}
    for fold in range(5):
        for arm in ('real', 'process'):
            for epoch in (8, 9, 10):
                path = folder / f'fold{fold}_{arm}_epoch{epoch}_monitor.json'
                report = json.loads(path.read_text())
                if report['epoch'] != epoch or len(report['targets']) != 47:
                    raise ValueError('Incomplete monitor')
                if fold == 2:
                    if report['fold'] != 2 or report['arm'] != arm or report['independently_trained']:
                        raise ValueError('Incorrect fold-2 reuse metadata')
                elif epoch > 8:
                    ck = folder / f'fold{fold}_{arm}_epoch{epoch}.pt'
                    if report['bindings'][str(root / f'coverage_training_20260925/fold{fold}_{arm}.pt')] != sha256(root / f'coverage_training_20260925/fold{fold}_{arm}.pt'):
                        raise ValueError('Changed source checkpoint')
                    if not ck.exists():
                        raise ValueError('Missing fitted checkpoint')
                if epoch == 8:
                    for target, score in report['targets'].items():
                        for metric in METRICS:
                            if not np.isclose(score[metric], baseline_data['folds'][str(fold)][arm][target][metric], rtol=1e-10, atol=1e-10):
                                raise ValueError('Epoch-8 score mismatch')
                reports[fold, arm, epoch] = report['targets']
                paths.append(path)
    targets = sorted(reports[0, 'real', 8])
    if any(sorted(scores) != targets for scores in reports.values()):
        raise ValueError('Target mismatch')
    folds = {}
    for fold in range(5):
        folds[str(fold)] = {
            arm: {str(epoch): {
                metric: float(np.mean([score[metric] for score in reports[fold, arm, epoch].values()]))
                for metric in METRICS
            } for epoch in (8, 9, 10)} for arm in ('real', 'process')
        }
    per_target = {}
    for target in targets:
        per_target[target] = {
            metric: {
                'real_delta_by_fold': [reports[f, 'real', 10][target][metric] - reports[f, 'real', 8][target][metric] for f in range(5)],
                'incremental_delta_by_fold': [
                    (reports[f, 'real', 10][target][metric] - reports[f, 'process', 10][target][metric])
                    - (reports[f, 'real', 8][target][metric] - reports[f, 'process', 8][target][metric])
                    for f in range(5)
                ],
            } for metric in METRICS
        }
    output = Path('artifacts/native_reconstruction_monitored_continuation_summary_20260929.json')
    if output.exists():
        raise ValueError('Existing summary')
    result = dict(folds=folds, targets=per_target,
                  bindings={str(path): sha256(path) for path in paths},
                  fit_independent_folds=[0, 1, 3, 4], equivalent_fit_fold=2,
                  independently_replicated=False, promotion_allowed=False,
                  runtime_modified=False,
                  limitation='Source-development fixed monitors overlap in fit/evaluation populations. Descriptive per-target patterns have not passed paired outer or external gates.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Wrote', output, 'for', len(targets), 'targets and', len(folds), 'folds')


if __name__ == '__main__':
    main()
