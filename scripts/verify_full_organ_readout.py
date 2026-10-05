"""Independently aggregate E25 losses and replay saved selection predictions."""
from contextlib import ExitStack, closing
import json
import math
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_full_organ_readout import (
    DEST, PARAMETERS, READERS, ROOT, paired_data, require_model,
    verify_full_grid, write_json,
)
from whole_body_jepa.multisource_intervals import sha256

ARMS = ('body', 'organs', 'shuffled', 'expanded', 'raw', 'native')
CONTROLS = ('body', 'shuffled', 'expanded', 'raw', 'native')
ASSESSMENT = Path('artifacts/native_reconstruction_full_organ_readout_assessment_20261003.json')
OUTPUT = Path('artifacts/native_reconstruction_full_organ_readout_verification_20261003.json')


def group_means(keys, values):
    """Sorted equal-unit means without the scorer's pandas grouping path."""
    unique, index, counts = np.unique(keys, return_inverse=True, return_counts=True)
    # Sequential accumulation drifts for large source groups; close arm means
    # amplify that rounding in paired tests. Keep independent, accurate sums.
    ordered = values[np.argsort(index, kind='stable')]
    groups = np.split(ordered, np.cumsum(counts)[:-1])
    sums = np.array([[math.fsum(column) for column in group.T] for group in groups],
                    dtype=np.float64)
    return unique, sums / counts[:, None]


def independent_assessment(frame, variables, repetitions=5000):
    if (len(variables) != 53 or len(set(variables)) != 53 or frame.empty
            or not set(frame.variable) <= set(variables)
            or frame[['patient', 'component', 'variable', 'fold']].isna().any().any()
            or frame.duplicated(['patient', 'variable', 'fold']).any()
            or not set(frame.fold) <= set(range(5))
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid complete-inventory prediction frame')
    links = frame[['patient', 'component']].drop_duplicates()
    if links.patient.duplicated().any():
        raise ValueError('A patient changes source component')
    component_by_patient = dict(links.itertuples(index=False, name=None))
    targets, screens = {}, {'organs_minus_'+c: [] for c in CONTROLS}
    for target_index, variable in enumerate(variables):
        rows = frame.loc[frame.variable == variable]
        if rows.empty:
            targets[variable] = dict(status='unsupported')
            continue
        errors = abs(rows[list(ARMS)].to_numpy()-rows.truth.to_numpy()[:, None])
        people, losses = group_means(rows.patient.to_numpy(), errors)
        components, component_losses = group_means(
            np.array([component_by_patient[p] for p in people]), losses)
        comparisons = {}
        seed = 20261003 + target_index*1000
        for i, control in enumerate(CONTROLS):
            col = ARMS.index(control)
            stats = dict(patient=paired_summary(losses[:, 1]-losses[:, col], seed+i, repetitions),
                         component=paired_summary(component_losses[:, 1]-component_losses[:, col],
                                                  seed+100+i, repetitions))
            passed = len(people) >= 30 and len(components) >= 3
            for stat in stats.values():
                passed = passed and (stat['ci95'] is not None and stat['ci95'][1] < 0
                    and stat['p_less'] is not None and stat['p_less'] < .05/(53*5))
            key = 'organs_minus_'+control
            comparisons[key] = dict(paired=stats, screen_pass=bool(passed))
            if passed:
                screens[key].append(variable)
        targets[variable] = dict(patients=len(people), components=len(components),
            query_occurrences=len(rows), folds=sorted(map(int, rows.fold.unique())),
            comparisons=comparisons, normalized_mae={
                level: dict(zip(ARMS, map(float, matrix.mean(axis=0))))
                for level, matrix in [('patient', losses), ('component', component_losses)]})
    supported = [v for v in variables if 'normalized_mae' in targets[v]]
    incremental = [v for v in supported if all(v in screens['organs_minus_'+c] for c in CONTROLS[:3])]
    summary = dict(variables=len(supported), query_occurrences=len(frame),
        patients=len(component_by_patient), variable_folds=len(frame[['variable', 'fold']].drop_duplicates()),
        screens=screens, joint_incremental=incremental,
        joint_all_controls=[v for v in incremental if all(v in screens['organs_minus_'+c] for c in CONTROLS[3:])],
        normalized_macro={level: {a: float(np.mean([targets[v]['normalized_mae'][level][a] for v in supported]))
                                 for a in ARMS} for level in ('patient', 'component')})
    return targets, summary


def compare_results(actual, expected):
    """Only floating-point roundoff is tolerated; screens and counts stay exact."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError('Changed result keys')
        return max((compare_results(v, expected[k]) for k, v in actual.items()), default=0.)
    if isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError('Changed result list')
        return max((compare_results(a, e) for a, e in zip(actual, expected)), default=0.)
    if isinstance(actual, float):
        if type(expected) not in (float, int) or not np.isfinite([actual, expected]).all():
            raise ValueError('Invalid numeric result')
        difference = abs(actual-expected)
        if difference > 64*np.finfo(np.float64).eps*max(1., abs(actual), abs(expected)):
            raise ValueError('Statistics exceed floating-point roundoff')
        return difference
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('Changed discrete result, support or screen')
    return 0.


def replay_selection(bundle, raw_model, frame, ids, data, variable, fold, components, shape):
    if (bundle['target'] != variable or bundle['parameters'] != PARAMETERS
            or set(bundle['models']) != set(ARMS[:4]) or ids != sorted(set(ids))):
        raise ValueError('Changed reader bundle or patient ordering')
    expected = pd.DataFrame(dict(patient=ids, component=[components[p] for p in ids],
        variable=variable, fold=fold, truth=data['truth'], native=data['prediction']))
    pd.testing.assert_frame_equal(frame[expected.columns].reset_index(drop=True), expected)
    projection = np.random.default_rng(20261003+fold).normal(size=(64, int(np.prod(shape))))/8
    np.testing.assert_array_equal(bundle['projection'], projection)
    body, organs, raw = [data[k] for k in ('body', 'organs', 'raw')]
    if body.shape != (len(ids), 64) or raw.shape != (len(ids), 265) or organs.shape != (len(ids), int(np.prod(shape))):
        raise ValueError('Changed reconstruction feature layout')
    # Deliberately assemble inputs without calling the trainer's features().
    inputs = dict(body=np.hstack([body, raw[:, 106:]]),
        organs=np.hstack([body, organs, raw[:, 106:]]),
        shuffled=np.hstack([body, organs[np.r_[len(ids)-1, np.arange(len(ids)-1)]], raw[:, 106:]]),
        expanded=np.hstack([body, body@projection, raw[:, 106:]]), raw=raw)
    if not all(np.isfinite(x).all() for x in inputs.values()):
        raise ValueError('Nonfinite replay inputs')
    for arm, x in inputs.items():
        model = raw_model if arm == 'raw' else bundle['models'][arm]
        require_model(model, x.shape[1])
        values = model.predict(x)
        if not np.isfinite(values).all() or not np.isfinite(frame[arm]).all():
            raise ValueError('Nonfinite saved or replayed prediction')
        np.testing.assert_allclose(values, frame[arm], atol=1e-12, rtol=1e-12)
    return len(inputs)


def main():
    if OUTPUT.exists():
        raise ValueError('Completed E25 verification already exists')
    if not ASSESSMENT.exists():
        raise ValueError('Full E25 assessment required before verification')
    report = json.loads(ASSESSMENT.read_text())
    checked = {}
    verify_bound_files(report['bindings'], checked)
    grid, variables, shape, bindings = verify_full_grid()
    for path, digest in bindings.items():
        if report['bindings'].get(path) != digest:
            raise ValueError('Assessment does not bind the complete extraction grid')
    if report['runtime_modified'] is not False or report['promotion_allowed'] is not False:
        raise ValueError('Research assessment changed runtime or promotion policy')
    roles_all = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if links.isna().any().any() or links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous source provenance')
    components = dict(links.itertuples(index=False, name=None))
    frames, replayed = [], 0
    with threadpool_limits(limits=4):
        for fold in range(5):
            rp = Path(f'artifacts/native_reconstruction_full_organ_readers_fold{fold}_20261003.json')
            if report['bindings'].get(str(rp)) != sha256(rp):
                raise ValueError('Unbound reader fold report')
            fitted = json.loads(rp.read_text())
            if (fitted['complete'] is not True or fitted['contract']['variables'] != variables
                    or set(fitted['targets']) != set(variables)):
                raise ValueError('Incomplete reader inventory')
            with ExitStack() as stack:
                ref = stack.enter_context(closing(sqlite3.connect(
                    (ROOT/f'nonlinear_probe_20261001/fold{fold}_selection.sqlite').resolve().as_uri()+'?mode=ro', uri=True)))
                organ_db = stack.enter_context(closing(sqlite3.connect(
                    (DEST/f'fold{fold}_selection.sqlite').resolve().as_uri()+'?mode=ro', uri=True)))
                for c, variable in enumerate(variables):
                    meta = fitted['targets'][variable]
                    if variable not in grid[fold, 'selection']['support']:
                        if meta != dict(status='unsupported', fit_patients=0, selection_patients=0):
                            raise ValueError('Unsupported target misreported')
                        continue
                    dest = READERS/f'fold{fold}'
                    mp, pp = dest/f'{variable}.joblib', dest/f'{variable}.parquet'
                    raw_path = ROOT/f'nonlinear_probe_20261001/fold{fold}_probes/{variable}.joblib'
                    for path, digest in [(mp, meta['model_sha256']), (pp, meta['prediction_sha256']),
                                         (raw_path, meta['raw_model_sha256'])]:
                        if path != raw_path and report['bindings'].get(str(path)) != digest:
                            raise ValueError('Unbound saved reader or predictions')
                        verify_bound_files({str(path): digest}, checked)
                    ids, data = paired_data(ref, organ_db, c, shape, roles_all[fold], 'selection')
                    if len(ids) != grid[fold, 'selection']['support'][variable]['expected']:
                        raise ValueError('Incomplete target query support')
                    frame = pd.read_parquet(pp)
                    replayed += replay_selection(joblib.load(mp), joblib.load(raw_path)['models']['raw_probe'],
                        frame, ids, data, variable, fold, components, shape)
                    frames.append(frame)
            print('Independently replayed full organ readers', fold, flush=True)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != sum(grid[f, 'selection']['expected'] for f in range(5)) or len(frames) != 47*5:
        raise ValueError('Incomplete selection grid')
    targets, summary = independent_assessment(data, variables)
    roundoff = compare_results(dict(targets=targets, summary=summary),
                               dict(targets=report['targets'], summary=report['summary']))
    paths = [ASSESSMENT, Path(__file__), Path('scripts/score_additional_reconstruction.py'),
             Path('scripts/train_full_organ_readout.py')]
    write_json(OUTPUT, dict(models_replayed=replayed, target_folds=len(frames),
        query_occurrences=len(data), independent_statistical_verification_complete=True,
        summary_roundoff_max=roundoff, summary=summary, checked_files=len(checked),
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Independent selection replay and loss aggregation with the fixed statistical primitive. '
                   'Not independent data, forecasting, or release promotion. Fit replay belongs to the bound assessment.'))
    print(json.dumps(dict(models_replayed=replayed, summary_roundoff_max=roundoff), indent=2), flush=True)


if __name__ == '__main__':
    main()
