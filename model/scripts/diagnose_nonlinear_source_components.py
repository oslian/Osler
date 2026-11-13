"""Source-only conditional error localization for the completed nonlinear probe."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_coverage_component_influence import influence
from scripts.diagnose_coverage_variance import decompose
from scripts.diagnose_source_group_residuals import patient_half
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.score_nonlinear_probe import ARMS
from scripts.score_nonlinear_placebo import verify_rows
from whole_body_jepa.multisource_intervals import sha256


CONTROLS = ('native', 'process_summary', 'population', 'trained_shuffle')


def patient_losses(frame):
    arms = [*ARMS, 'trained_shuffle']
    if (frame.empty or frame.target.nunique() != 1 or not frame.role.eq('selection').all()
            or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *arms]]).all().all()):
        raise ValueError('Invalid source paired errors')
    error = frame[arms].subtract(frame.truth, axis=0).abs()
    error['patient'] = frame.patient.to_numpy()
    error = error.groupby('patient', sort=True)[arms].mean()
    groups = frame.drop_duplicates('patient').set_index('patient').component
    error['component'] = groups.reindex(error.index)
    return error


def diagnose(errors, reference, seed, repetitions=1000):
    if errors.index.duplicated().any() or reference not in CONTROLS:
        raise ValueError('Invalid patient/contrast')
    delta = errors.body_probe-errors[reference]
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite paired differences')
    groups = []
    for i, (component, values) in enumerate(delta.groupby(errors.component, sort=True)):
        stats = paired_summary(values.to_numpy(), seed+i, repetitions)
        halves = np.array([patient_half(p) for p in values.index])
        half_means = [float(values.to_numpy()[halves==h].mean()) if (halves==h).any() else None for h in (0,1)]
        counts = [int((halves==h).sum()) for h in (0,1)]
        groups.append(dict(component=str(component), patients=len(values), mean=stats['mean'],
            mean_variance=float(values.var(ddof=1)/len(values)) if len(values)>1 else None,
            nominal_bootstrap_ci95=stats['ci95'], half_counts=counts, half_means=half_means,
            supported_both_half_harm=bool(min(counts)>=15 and all(v>0 for v in half_means)),
            supported_both_half_benefit=bool(min(counts)>=15 and all(v<0 for v in half_means))))
    if len(groups)<2:
        raise ValueError('Insufficient component diagnostic')
    inf = influence([dict(patients=g['patients'], real_minus_reference=g['mean']) for g in groups], 'reference')
    return dict(groups=groups, influence=inf,
        variance=decompose([g['mean'] for g in groups], [np.nan if g['mean_variance'] is None else g['mean_variance'] for g in groups]),
        mean_harm_groups=sum(g['mean']>0 for g in groups),
        mean_benefit_groups=sum(g['mean']<0 for g in groups),
        supported_nominal_harm_groups=sum(g['patients']>=30 and g['nominal_bootstrap_ci95'] is not None and g['nominal_bootstrap_ci95'][0]>0 for g in groups),
        supported_nominal_benefit_groups=sum(g['patients']>=30 and g['nominal_bootstrap_ci95'] is not None and g['nominal_bootstrap_ci95'][1]<0 for g in groups),
        supported_both_half_harm_groups=sum(g['supported_both_half_harm'] for g in groups),
        supported_both_half_benefit_groups=sum(g['supported_both_half_benefit'] for g in groups))


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921/nonlinear_probe_20261001')
    output = Path('artifacts/native_reconstruction_nonlinear_source_components_20261003.json')
    if output.exists():
        raise ValueError('Existing completed diagnostic')
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    placebo_path = Path('artifacts/native_reconstruction_nonlinear_placebo_assessment_20261003.json')
    source, placebo = [json.loads(p.read_text()) for p in (source_path,placebo_path)]
    paths = [Path(__file__), source_path, placebo_path,
        Path('scripts/diagnose_coverage_component_influence.py'), Path('scripts/diagnose_coverage_variance.py'),
        Path('scripts/diagnose_source_group_residuals.py'), Path('scripts/score_additional_reconstruction.py'),
        Path('scripts/score_nonlinear_placebo.py'), Path('scripts/score_monitored_continuation_outer.py')]
    checked = {}
    verify_bound_files(source['bindings'], checked)
    targets = sorted(source['targets'])
    if len(targets)!=47:
        raise ValueError('Changed target family')
    frames = []
    for fold in range(5):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
        report = json.loads(rp.read_text())
        verify_bound_files(report['bindings'], checked)
        if sorted(report['targets'])!=targets or report['fold']!=fold:
            raise ValueError('Incomplete control family')
        paths.append(rp)
        for target in targets:
            p = root/f'placebo_20261003/fold{fold}/{target}.parquet'
            if sha256(p)!=report['targets'][target]['prediction_sha256']:
                raise ValueError('Changed paired predictions')
            # Only selection rows are read; no outer outcomes enter diagnosis.
            frame = pd.read_parquet(p, filters=[('role','==','selection')])
            prior = pd.read_parquet(root/f'fold{fold}_probes/{target}.parquet')
            verify_rows(frame,prior,'selection',fold)
            if len(frame)!=report['targets'][target]['query_counts']['selection']:
                raise ValueError('Incomplete source query grid')
            frames.append(frame); paths.append(p)
    data = pd.concat(frames,ignore_index=True)
    if len(data)!=source['summary']['query_occurrences']:
        raise ValueError('Changed full source grid')
    results = {}
    for i,target in enumerate(targets):
        error = patient_losses(data.loc[data.target==target])
        expected = placebo['results']['selection'][target]
        grouped = error.groupby('component')
        if len(error)!=expected['unique_patients'] or len(grouped)!=expected['components']:
            raise ValueError('Changed patient/component counts')
        for level,means in (('patient',error.select_dtypes('number').mean()),
                            ('component',grouped.mean(numeric_only=True).mean())):
            for arm in (*ARMS,'trained_shuffle'):
                if not np.isclose(means[arm],expected['normalized_mae'][level][arm],atol=1e-10,rtol=1e-10):
                    raise ValueError('Original paired weighting changed')
        comparisons = {c:diagnose(error,c,20261003+10000*i+1000*j) for j,c in enumerate(CONTROLS)}
        results[target] = dict(patients=len(error),components=len(grouped),comparisons=comparisons)
        print('Source nonlinear component diagnosis',i+1,'of47',flush=True)
    summary = {}
    for c in CONTROLS:
        rows = {t:r['comparisons'][c] for t,r in results.items()}
        summary[c] = dict(
            targets_with_supported_nominal_harm=[t for t,r in rows.items() if r['supported_nominal_harm_groups']],
            targets_with_supported_both_half_harm=[t for t,r in rows.items() if r['supported_both_half_harm_groups']],
            all_component_means_beneficial=[t for t,r in rows.items() if r['mean_benefit_groups']==len(r['groups'])],
            negative_mean_after_any_one_component_removal=[t for t,r in rows.items() if r['influence']['leave_one_mean_max']<0],
            targets_with_positive_excess_variance=[t for t,r in rows.items() if r['variance']['excess_variance'] is not None and r['variance']['excess_variance']>0])
    output.write_text(json.dumps(dict(summary=summary,targets=results,
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Source-only current reconstruction, all47 targets/four references. Conditional descriptive component precision/influence, not new significance gates. Nominal subgroup CIs are not multiplicity-corrected discoveries. Patient losses first average overlapping folds. Shared fits induce dependence; half signs and excess variance are not independent replications or sole-cause proofs. All groups retained; no fitting, corrected outputs, outer-based selection or threshold relaxation.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
