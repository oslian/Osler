"""Verify E24 saved predictions and describe fitted tree complexity, without tuning."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_reader_fit_support import ARMS, OUTPUT, PRIVATE, ROOT, SMALL, summarize
from scripts.probe_organ_compartments import write_json
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.verify_state_suffix import numeric_summary_parity
from whole_body_jepa.multisource_intervals import sha256


def leaf_counts(model):
    if len(model._predictors) != 100:
        raise ValueError('Incomplete fixed reader')
    result = []
    for trees in model._predictors:
        if len(trees) != 1: raise ValueError('Unexpected regressor outputs')
        nodes = trees[0].nodes
        if not {'count', 'is_leaf'} <= set(nodes.dtype.names):
            raise ValueError('Unsupported tree node schema')
        count = int(np.count_nonzero(nodes['is_leaf']))
        if not 1 <= count <= 15: raise ValueError('Changed tree specification')
        result.append(count)
    return result


def assessment_parity(targets, summary, report):
    if set(targets) != set(report['targets']): raise ValueError('Changed target inventory')
    errors = []
    for v, row in targets.items():
        if set(row) != set(report['targets'][v]): raise ValueError('Changed result schema')
        for k, value in row.items():
            expected = report['targets'][v][k]
            if k == 'normalized_mae': errors.append(numeric_summary_parity(value, expected))
            elif value != expected: raise ValueError('Changed paired assessment')
    expected = {k:v for k,v in report['summary'].items() if k != 'fit_support'}
    if set(summary) != set(expected): raise ValueError('Changed summary schema')
    for k, value in summary.items():
        if k == 'normalized_macro': errors.append(numeric_summary_parity(value, expected[k]))
        elif value != expected[k]: raise ValueError('Changed summary support or screens')
    return max(errors, default=0.)


def main():
    output = Path('artifacts/native_reconstruction_reader_support_verification_20261003.json')
    if output.exists(): raise ValueError('Existing verification')
    report = json.loads(OUTPUT.read_text()); checked = {}
    verify_bound_files(report['bindings'], checked)
    data = pd.read_parquet(PRIVATE/'predictions.parquet')
    replayed, leaves, differences = 0, [], []
    with threadpool_limits(limits=4):
        for fold in range(5):
            old_report = json.loads(Path(f'artifacts/native_reconstruction_organ_compartments_fold{fold}_20261003.json').read_text())
            variables = old_report['contract']['variables']
            cache = joblib.load(SMALL/f'fold{fold}/selection.joblib')
            for c, v in enumerate(variables):
                if old_report['targets'][v]['status'] != 'fitted': continue
                rows = np.flatnonzero(cache['target'] == c)
                part = data[(data.fold == fold) & (data.variable == v)].reset_index(drop=True)
                np.testing.assert_array_equal(part.patient, cache['patient'][rows])
                np.testing.assert_array_equal(part.truth, cache['truth'][rows])
                np.testing.assert_array_equal(part.native, cache['native'][rows])
                paths = dict(full_raw=ROOT/f'nonlinear_probe_20261001/fold{fold}_probes/{v}.joblib',
                             small_raw=SMALL/f'fold{fold}/{v}.joblib')
                record = dict(fold=fold, variable=v)
                for arm, path in paths.items():
                    model = joblib.load(path)['models']
                    model = model['raw_probe'] if arm == 'full_raw' else model['raw']['hgb']
                    actual = model.predict(cache['raw'][rows])
                    np.testing.assert_array_equal(actual, part[arm])
                    count = leaf_counts(model)
                    record[arm] = dict(mean=float(np.mean(count)), min=min(count), max=max(count))
                    replayed += 1
                leaves.append(record)
            print('Replayed fixed support readers', fold, flush=True)
    targets, summary = summarize(data, variables)
    roundoff = assessment_parity(targets, summary, report)
    for v, r in targets.items():
        if 'normalized_mae' not in r: continue
        # Independently group row losses to check repeated-patient aggregation.
        part = data[data.variable == v]
        patient_rows = []
        for p, group in part.groupby('patient'):
            patient_rows.append(dict(component=group.component.iloc[0], **{
                a: float(np.mean(np.abs(group[a].to_numpy()-group.truth.to_numpy()))) for a in ARMS}))
        independent = pd.DataFrame(patient_rows)
        for level, expected in [('patient', independent[list(ARMS)].mean()),
                                ('component', independent.groupby('component')[list(ARMS)].mean().mean())]:
            for a in ARMS:
                np.testing.assert_allclose(expected[a], r['normalized_mae'][level][a], atol=1e-12, rtol=1e-12)
        differences.append(dict(variable=v, full_better_small=all(r['normalized_mae'][l]['full_raw'] <
            r['normalized_mae'][l]['small_raw'] for l in ('patient', 'component')),
            full_better_native=all(r['normalized_mae'][l]['full_raw'] < r['normalized_mae'][l]['native']
                                  for l in ('patient', 'component'))))
    result = dict(models_replayed=replayed, exact_saved_prediction_parity=True,
        summary_roundoff_max=roundoff,
        independent_loss_aggregation_verified=True, targets_verified=len(differences),
        full_better_small_both_means=sum(d['full_better_small'] for d in differences),
        full_better_native_both_means=sum(d['full_better_native'] for d in differences),
        mean_tree_leaves={a: float(np.mean([r[a]['mean'] for r in leaves])) for a in ('full_raw', 'small_raw')})
    write_json(output, dict(summary=result, fitted_tree_complexity=leaves,
        descriptive_direction=differences, checked_files=len(checked),
        bindings={str(OUTPUT): sha256(OUTPUT), str(Path(__file__)): sha256(Path(__file__))},
        posthoc_diagnostic='Tree complexity and mean directions only; no new gate, test selection or claim of pure sample-count causality.',
        runtime_modified=False, promotion_allowed=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
