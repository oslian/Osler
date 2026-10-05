"""Descriptive source-only error decomposition; no deployable correction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_nonlinear_source_components import patient_losses
from scripts.diagnose_source_group_residuals import crossfit_offset, patient_half
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


ARMS = ('native', 'body_probe', 'raw_probe', 'process_summary', 'population', 'trained_shuffle')
BANDS = ('central', 'intermediate', 'tail')


def decompose_errors(frame):
    # This also rejects duplicate patient/fold rows, mixed components and test rows.
    original = patient_losses(frame)
    frame = frame.reset_index(drop=True).copy()
    if not np.isfinite(frame[['truth', *ARMS]]).all().all():
        raise ValueError('Nonfinite errors')
    weights = 1 / frame.groupby('patient').patient.transform('size').to_numpy()
    mass = float(weights.sum())
    distance = abs(frame.truth.to_numpy())
    band = np.where(distance <= 1, 'central', np.where(distance <= 3, 'intermediate', 'tail'))
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    differences = errors.body_probe.to_numpy()-errors.native.to_numpy()
    positive = np.maximum(differences, 0)
    total_positive = float(weights @ positive)
    bands = {}
    for name in BANDS:
        keep = band == name
        w = weights[keep]
        subtotal = float(w.sum())
        bands[name] = dict(patient_equivalent_mass=subtotal, mass_fraction=subtotal/mass,
            normalized_mae={arm: float(w @ errors.loc[keep, arm].to_numpy()/subtotal) if subtotal else None for arm in ARMS},
            body_minus_native_contribution=float(w @ differences[keep]/mass),
            share_of_positive_body_regression=float(w @ positive[keep]/total_positive) if total_positive else None)
    raw_delta = float(original.body_probe.mean()-original.native.mean())
    if not np.isclose(sum(b['body_minus_native_contribution'] for b in bands.values()), raw_delta, atol=1e-10):
        raise ValueError('Band contributions do not reproduce patient-equal difference')

    corrected = pd.DataFrame(np.nan, index=frame.index, columns=ARMS)
    groups = []
    for (fold, component), part in frame.groupby(['fold', 'component'], sort=True):
        halves = np.array([patient_half(p) for p in part.patient])
        statistics = {}
        for arm in ARMS:
            prediction = part[arm].to_numpy()
            truth = part.truth.to_numpy()
            shifted = crossfit_offset(truth, prediction, halves)
            if shifted is not None:
                corrected.loc[part.index, arm] = abs(shifted-truth)
            statistics[arm] = dict(mean_signed_error=float(np.mean(prediction-truth)),
                median_signed_error=float(np.median(prediction-truth)),
                normalized_mae=float(np.mean(abs(prediction-truth))),
                crossfit_offset_mae=float(np.mean(abs(shifted-truth))) if shifted is not None else None)
        groups.append(dict(fold=int(fold), component=str(component), patients=len(part),
            half_counts=[int(sum(halves == h)) for h in (0, 1)], arms=statistics))
    eligible = corrected.notna().all(axis=1).to_numpy()
    if eligible.any():
        # Same occurrence support and same patient weights for every arm, before/after.
        eligible_weights = weights[eligible]
        denom = float(eligible_weights.sum())
        matched = dict(patient_equivalent_mass=denom, mass_fraction=denom/mass,
            before={a: float(eligible_weights @ errors.loc[eligible, a].to_numpy()/denom) for a in ARMS},
            after={a: float(eligible_weights @ corrected.loc[eligible, a].to_numpy()/denom) for a in ARMS})
    else:
        matched = None
    component_means = original.groupby('component')[list(ARMS)].mean().mean()
    return dict(patients=len(original), occurrences=len(frame), bands=bands, groups=groups,
        normalized_mae=dict(patient=original[list(ARMS)].mean().to_dict(), component=component_means.to_dict()),
        matched_local_offset=matched)


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921/nonlinear_probe_20261001')
    output = Path('artifacts/native_reconstruction_nonlinear_source_errors_20261003.json')
    if output.exists():
        raise ValueError('Completed diagnostic already exists')
    parent_path = Path('artifacts/native_reconstruction_nonlinear_source_components_20261003.json')
    parent = json.loads(parent_path.read_text())
    checked = {}
    verify_bound_files(parent['bindings'], checked)
    assessment_path = Path('artifacts/native_reconstruction_nonlinear_placebo_assessment_20261003.json')
    assessment = json.loads(assessment_path.read_text())
    results = {}
    for i, target in enumerate(sorted(parent['targets'])):
        frames = []
        for fold in range(5):
            p = root/f'placebo_20261003/fold{fold}/{target}.parquet'
            if str(p) not in parent['bindings'] or sha256(p) != parent['bindings'][str(p)]:
                raise ValueError('Unbound prediction file')
            frames.append(pd.read_parquet(p, filters=[('role', '==', 'selection')]))
        data = pd.concat(frames, ignore_index=True)
        result = decompose_errors(data)
        expected = assessment['results']['selection'][target]
        if result['patients'] != expected['unique_patients']:
            raise ValueError('Patient support changed')
        for level in ('patient', 'component'):
            for arm in ARMS:
                if not np.isclose(result['normalized_mae'][level][arm], expected['normalized_mae'][level][arm], rtol=1e-10, atol=1e-10):
                    raise ValueError('Aggregate parity failure')
        results[target] = result
        print('Source error decomposition', i+1, 'of', len(parent['targets']), flush=True)
    if len(results) != 47:
        raise ValueError('Incomplete target family')
    summary = dict(patient_mean_harm_vs_native=[t for t,r in results.items() if r['normalized_mae']['patient']['body_probe'] > r['normalized_mae']['patient']['native']],
        matched_offset_still_harm_vs_native=[t for t,r in results.items() if r['matched_local_offset'] is not None and r['matched_local_offset']['after']['body_probe'] > r['matched_local_offset']['after']['native']])
    paths = [Path(__file__), parent_path, assessment_path, Path('scripts/diagnose_nonlinear_source_components.py'), Path('scripts/diagnose_source_group_residuals.py')]
    output.write_text(json.dumps(dict(summary=summary, targets=results,
        bindings={str(p):sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Source-only masked-current reconstruction, all47 targets. Bands use absolute fit-normalized truth: <=1, (1,3], >3; not clinical normality or runtime routing. Patient weights sum to one across eligible overlapping folds. Tail contribution is descriptive; positive-regression share is not share of total error. Offset uses disjoint patient hash halves within known fold/component and treats all arms equally; not transferable calibration. Singleton/one-half groups remain uncorrected with missing corrected statistics; matched comparison reports its reduced mass. No significance, outer tuning, encoder update or new gate.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
