"""Describe E29 failure positions without fitting or changing any gate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_organ_assimilation import CONTROLS, DEST, validate_identities
from scripts.train_organ_binding import write_json
from whole_body_jepa.multisource_intervals import sha256


def failure_positions(results):
    contrasts, unsupported = {}, []
    scored = {v: r for v, r in results.items() if r['status'] == 'scored'}
    unsupported = sorted(set(results) - set(scored))
    for control in CONTROLS:
        rows = {}
        for variable, result in scored.items():
            fit = result.get('fit_patient_mae', {})
            fit_delta = None if control not in fit else fit['immediate'] - fit[control]
            selected = result['normalized_mae']['patient']
            selected_delta = selected['immediate'] - selected[control]
            component = result['normalized_mae']['component']
            component_delta = component['immediate'] - component[control]
            values = [selected_delta, component_delta]
            if fit_delta is not None:
                values.append(fit_delta)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite diagnostic contrast')
            if fit_delta is None:
                stage = 'fit_comparison_unavailable'
            elif fit_delta >= 0:
                stage = 'no_fit_mean_gain'
            elif selected_delta >= 0:
                stage = 'fit_gain_reversed_on_selection'
            else:
                stage = 'fit_and_selection_mean_gain'
            rows[variable] = dict(fit_delta=fit_delta, selection_delta=selected_delta,
                component_delta=component_delta, stage=stage,
                minimum_support=result['patients'] >= 30 and result['components'] >= 3,
                patient_component_direction_disagrees=(selected_delta < 0) != (component_delta < 0),
                original_screen_pass=result['comparisons'][control]['screen_pass'])
        contrasts[control] = dict(targets=rows, counts={
            stage: sum(r['stage'] == stage for r in rows.values()) for stage in (
                'fit_comparison_unavailable', 'no_fit_mean_gain',
                'fit_gain_reversed_on_selection', 'fit_and_selection_mean_gain')},
            selection_mean_gain=sum(r['selection_delta'] < 0 for r in rows.values()),
            direction_disagreement=sum(r['patient_component_direction_disagrees'] for r in rows.values()))
    return dict(scored=len(scored), unsupported=unsupported,
        minimum_support=sum(r['patients'] >= 30 and r['components'] >= 3 for r in scored.values()),
        contrasts=contrasts)


def support(frame):
    validate_identities(frame)
    if (frame.zero_gap.isna().any() or frame.event_count.isna().any()
            or not pd.api.types.is_bool_dtype(frame.zero_gap.dtype)
            or not pd.api.types.is_integer_dtype(frame.event_count.dtype)
            or (frame.event_count < 0).any()):
        raise ValueError('Invalid event support')
    return dict(occurrences=len(frame), patients=int(frame.patient.nunique()),
        zero_gap_occurrences=int(frame.zero_gap.sum()),
        zero_observed_event_occurrences=int((frame.event_count == 0).sum()),
        single_observed_event_occurrences=int((frame.event_count == 1).sum()),
        single_observed_event_zero_gap_occurrences=int(((frame.event_count == 1) & frame.zero_gap).sum()))


def correction_summary(reports):
    result = {}
    for channel in ('immediate', 'delayed', 'body'):
        result[channel] = {}
        for role in ('fit', 'selection'):
            rows = [r['diagnostics'][channel][role] for r in reports]
            for r in rows:
                if (r['coordinates'] <= 0 or not 0 <= r['saturated'] <= r['coordinates']
                        or not np.isfinite([r['absolute_sum'], r['maximum']]).all()
                        or r['absolute_sum'] < 0 or r['maximum'] < 0):
                    raise ValueError('Invalid correction summary')
            count = sum(r['coordinates'] for r in rows)
            saturation = sum(r['saturated'] for r in rows)
            result[channel][role] = dict(coordinates=count, saturated=saturation,
                saturated_fraction=saturation/count,
                mean_absolute_correction=sum(r['absolute_sum'] for r in rows)/count,
                maximum_correction=max(r['maximum'] for r in rows))
    return result


def require_verification(assessment_path, verification):
    if verification.get('complete') is not True:
        raise ValueError('Complete independent verification required')
    if verification.get('bindings', {}).get(str(assessment_path)) != sha256(assessment_path):
        raise ValueError('Verification does not bind this assessment')
    for path, digest in verification['bindings'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed verification dependency')


def main():
    ap = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    vp = Path('artifacts/native_organ_assimilation_verification_20261004.json')
    output = Path('artifacts/native_organ_assimilation_diagnosis_20261005.json')
    if output.exists():
        raise ValueError('Completed diagnosis exists; do not overwrite')
    assessment, verification = json.loads(ap.read_text()), json.loads(vp.read_text())
    require_verification(ap, verification)
    bindings = {str(p): sha256(p) for p in (ap, vp, Path(__file__))}
    frames = {'fit': [], 'selection': []}
    reports = []
    for fold in range(5):
        rp = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
        if sha256(rp) != assessment['bindings'][str(rp)]:
            raise ValueError('Changed fold report')
        report = json.loads(rp.read_text())
        reports.append(report)
        bindings[str(rp)] = sha256(rp)
        for role in frames:
            path = DEST/f'fold{fold}'/f'{role}_predictions.parquet'
            if sha256(path) != report['prediction_hashes'][role]:
                raise ValueError('Changed predictions')
            frames[role].append(pd.read_parquet(path))
            bindings[str(path)] = sha256(path)
    result = dict(bindings=bindings, positions=failure_positions(assessment['results']),
        support={role: support(pd.concat(rows, ignore_index=True)) for role, rows in frames.items()},
        corrections=correction_summary(reports), runtime_modified=False, promotion_allowed=False,
        limitation='Descriptive means, not significance or causal attribution. Support counts are repeated '
        'query occurrences. Correction summaries reuse bound training hooks, including zero/absent steps, '
        'and are not independently recomputed. Raw fitting comparison is unavailable in this artifact.')
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}, indent=2))


if __name__ == '__main__':
    main()
