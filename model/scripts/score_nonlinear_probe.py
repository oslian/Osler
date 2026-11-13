"""All-fold fixed probe assessment; overlapping queries count once per patient."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import FAMILY_SIZE, paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256


ARMS = ('native', 'body_probe', 'raw_probe', 'process_summary', 'population')
CONTRASTS = (
    ('body_probe', 'native'), ('raw_probe', 'native'),
    ('process_summary', 'native'), ('raw_probe', 'body_probe'),
    ('raw_probe', 'process_summary'), ('body_probe', 'process_summary'),
    ('body_probe', 'population'), ('raw_probe', 'population'),
)


def verify_probe_frame(frame, baseline, roles, fold, target, location, scale, components):
    if (frame.empty or frame.patient.duplicated().any()
            or not (frame.fold == fold).all() or not (frame.target == target).all()
            or baseline.patient.duplicated().any()
            or set(frame.patient) != set(baseline.patient)):
        raise ValueError('Incomplete or duplicate query identity')
    if (not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()
            or not np.isfinite([location, scale]).all() or scale <= 0):
        raise ValueError('Invalid normalized predictions')
    if any(roles[p] != 'selection' for p in frame.patient):
        raise ValueError('Wrong patient role')
    if not np.array_equal(frame.component.to_numpy(), frame.patient.map(components).to_numpy()):
        raise ValueError('Changed component provenance')
    expected = baseline.set_index('patient').loc[frame.patient]
    for column in ('truth', 'population'):
        normalized = (expected[column].to_numpy() - location) / scale
        if not np.allclose(frame[column], normalized, atol=1e-10, rtol=1e-10):
            raise ValueError('Changed truth or population reference')


def patient_errors(frame):
    """Average paired losses across a patient's folds, not their predictions."""
    if (frame.empty or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or frame.target.nunique() != 1
            or not np.isfinite(frame[['truth', *ARMS]]).all().all()):
        raise ValueError('Invalid paired target rows')
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    errors['patient'] = frame.patient.to_numpy()
    unique = errors.groupby('patient', sort=True)[list(ARMS)].mean()
    groups = frame.drop_duplicates('patient').set_index('patient').component
    unique['component'] = groups.reindex(unique.index)
    return unique


def assess_target(frame, seed, repetitions=5000):
    errors = patient_errors(frame)
    grouped = errors.groupby('component')[list(ARMS)].mean()
    comparisons = {}
    for index, (candidate, reference) in enumerate(CONTRASTS):
        delta = errors[candidate] - errors[reference]
        group_delta = grouped[candidate] - grouped[reference]
        paired = dict(patient=paired_summary(delta, seed + index, repetitions),
                      component=paired_summary(group_delta, seed + 100 + index, repetitions))
        sufficient = len(errors) >= 30 and len(grouped) >= 3
        def passes(family):
            return bool(sufficient and all(s['mean'] < 0 and s['p_less'] is not None
                        and s['p_less'] < .05 / family for s in paired.values()))
        comparisons[candidate + '_minus_' + reference] = dict(
            paired=paired, per_contrast_47_screen=passes(FAMILY_SIZE),
            all_8_contrasts_47_screen=passes(FAMILY_SIZE * len(CONTRASTS)),
            component_mean_harms=int((group_delta > 0).sum()))
    return dict(query_occurrences=len(frame), unique_patients=len(errors), components=len(grouped),
                normalized_mae=dict(patient=errors[list(ARMS)].mean().to_dict(),
                                    component=grouped.mean().to_dict()),
                comparisons=comparisons,
                per_component=[dict(component=str(c), patients=int((errors.component == c).sum()),
                                    normalized_mae=row.to_dict()) for c, row in grouped.iterrows()])


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    private = root / 'nonlinear_probe_20261001'
    output = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    if output.exists():
        raise ValueError('Existing fixed assessment')
    sp, fp = root / 'scaling_20260922.pt', root / 'partitions_20260922.pt'
    pp = root / 'provenance.parquet'
    scaling, plan = torch.load(sp, weights_only=False), torch.load(fp, weights_only=False)
    links = pd.read_parquet(pp)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    original_path = Path('artifacts/native_reconstruction_full_continuation_selection_20260930.json')
    original = json.loads(original_path.read_text())
    checked = {}
    verify_bound_files(original['bindings'], checked)
    targets = sorted(original['folds']['0'])
    if len(targets) != FAMILY_SIZE:
        raise ValueError('Changed target family')
    paths = [Path(__file__), Path('scripts/score_additional_reconstruction.py'),
             Path('scripts/score_monitored_continuation_outer.py'),
             Path('scripts/train_nonlinear_probe.py'), sp, fp, pp, original_path]
    all_frames, folds = [], {}
    for fold in range(5):
        report_path = Path(f'artifacts/native_reconstruction_nonlinear_training_fold{fold}_20261001.json')
        report = json.loads(report_path.read_text())
        if (report['fold'] != fold or report['parameters'] != PARAMETERS
                or sorted(report['targets']) != targets or report['promotion_allowed'] is not False):
            raise ValueError('Changed fitted probe family')
        verify_bound_files(report['bindings'], checked)
        base_path = root / f'coverage_selection_20260926/fold{fold}.parquet'
        base = pd.read_parquet(base_path)
        proof_path = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_selection_20261001.json')
        proof = json.loads(proof_path.read_text())
        if proof['fold'] != fold or proof['role'] != 'selection':
            raise ValueError('Wrong extraction proof')
        verify_bound_files(proof['bindings'], checked)
        dest = private / f'fold{fold}_probes'
        contract_path = dest / 'contract.json'
        contract = json.loads(contract_path.read_text())
        if contract != dict(fold=fold, parameters=PARAMETERS, bindings=report['bindings']):
            raise ValueError('Wrong fitted contract')
        paths.extend((report_path, base_path, proof_path, contract_path))
        frames, rows = [], {}
        for target in targets:
            path, model_path = dest / f'{target}.parquet', dest / f'{target}.joblib'
            result = report['targets'][target]
            if (result['target'] != target or result['fold'] != fold
                    or result['contract_sha256'] != sha256(contract_path)
                    or result['prediction_sha256'] != sha256(path)
                    or result['model_sha256'] != sha256(model_path)):
                raise ValueError('Changed model or prediction artifact')
            frame = pd.read_parquet(path)
            c = scaling['variables'].index(target)
            scaler = scaling['scalers'][fold]
            verify_probe_frame(frame, base.loc[base.target == target], plan['folds'][fold], fold,
                               target, scaler['location'][c], scaler['scale'][c], components)
            if len(frame) != result['selection_patients']:
                raise ValueError('Changed query count')
            error = patient_errors(frame)
            means = dict(patient=error[list(ARMS)].mean().to_dict(),
                         component=error.groupby('component')[list(ARMS)].mean().mean().to_dict())
            for level in means:
                for arm in ARMS:
                    if not np.isclose(means[level][arm], result['normalized_mae'][level][arm], atol=1e-10, rtol=1e-10):
                        raise ValueError('Changed saved mean')
                if not np.isclose(means[level]['native'], original['folds'][str(fold)][target]['normalized_mae'][level]['real10'], atol=1e-10, rtol=1e-10):
                    raise ValueError('Original readout reproduction failed')
            rows[target] = dict(patients=len(frame), normalized_mae=means)
            paths.extend((path, model_path))
            frames.append(frame)
        full = pd.concat(frames, ignore_index=True)
        if len(full) != proof['examples'] or full.patient.nunique() != proof['patients']:
            raise ValueError('Incomplete full fold grid')
        folds[str(fold)] = rows
        all_frames.append(full)
        print('Verified full fixed source probes', fold, len(full), flush=True)
    data = pd.concat(all_frames, ignore_index=True)
    results = {}
    for i, target in enumerate(targets):
        results[target] = assess_target(data.loc[data.target == target], 20261001 + 1000 * i)
        print('Scored all-fold target', i + 1, 'of', FAMILY_SIZE, flush=True)
    contrast_names = [c + '_minus_' + r for c, r in CONTRASTS]
    summary = dict(query_occurrences=len(data), unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient', 'target']].drop_duplicates()),
        normalized_target_macro={level: {arm: float(np.mean([
            result['normalized_mae'][level][arm] for result in results.values()]))
            for arm in ARMS} for level in ('patient', 'component')},
        all_contrasts_screen_passes={name: [t for t, r in results.items()
            if r['comparisons'][name]['all_8_contrasts_47_screen']] for name in contrast_names},
        fold_macro={f: {level: {arm: float(np.mean([r['normalized_mae'][level][arm]
            for r in rows.values()])) for arm in ARMS} for level in ('patient', 'component')}
            for f, rows in folds.items()})
    report = dict(summary=summary, targets=results, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='All47/five source folds. Current masked reconstruction, reused development, not future prediction. Average paired normalized absolute losses within patient across eligible folds, not predictions or independent replicated patients. Uncertainty conditional on overlapping fitted models; component means retained including small groups. Existing47-family per-contrast screens plus stricter8x47 descriptive family screen. No runtime, clinical, causal, new lockbox or exact-capacity claim. Raw summaries omit detailed sequence order. No encoder information-loss proof from readout differences alone.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
