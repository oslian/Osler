"""Frozen all-target descriptive assessment; no fitting or promotion."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import torch

from whole_body_jepa.multisource_intervals import sha256


ARMS = ('real', 'process', 'population')
FAMILY_SIZE = 47


def paired_summary(delta, seed, repetitions=5000):
    values = np.asarray(delta, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Invalid paired differences')
    mean = float(values.mean())
    if len(values) < 2:
        return dict(n=len(values), mean=mean, p_less=None, ci95=None)
    p = (0.0 if mean < 0 else 1.0) if np.ptp(values) == 0 else float(
        ttest_1samp(values, 0, alternative='less').pvalue)
    rng = np.random.default_rng(seed)
    samples = []
    # Bound memory without changing ordinary paired-unit resampling.
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        samples.extend(values[rng.integers(len(values), size=(count, len(values)))].mean(1))
    return dict(n=len(values), mean=mean, p_less=p,
                ci95=np.quantile(samples, [.025, .975]).tolist())


def summarize_target(frame, seed, repetitions=5000):
    if frame.empty or frame.patient.duplicated().any():
        raise ValueError('Empty target or duplicated patient')
    values = frame[['truth', 'fit_scale', *ARMS]].to_numpy()
    if not np.isfinite(values).all() or (frame.fit_scale <= 0).any():
        raise ValueError('Invalid prediction or scale')
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    normalized = errors.div(frame.fit_scale, axis=0)
    grouped = errors.groupby(frame.component).mean()
    normalized_grouped = normalized.groupby(frame.component).mean()
    comparisons = {}
    for i, arm in enumerate(ARMS[1:]):
        comparisons[arm] = {
            'patient': paired_summary(errors.real - errors[arm], seed + i, repetitions),
            'component': paired_summary(grouped.real - grouped[arm], seed + 100 + i, repetitions),
        }
    passed = len(frame) >= 30 and len(grouped) >= 3 and all(
        stat['mean'] < 0 and stat['p_less'] is not None and stat['p_less'] < .05 / FAMILY_SIZE
        for comparison in comparisons.values() for stat in comparison.values())
    groups = []
    for component, part in frame.groupby('component', sort=True):
        err = errors.loc[part.index]
        groups.append(dict(component=str(component), patients=len(part),
            fold=int(part.fold.iloc[0]), raw_mae=err.mean().to_dict(),
            normalized_mae=normalized.loc[part.index].mean().to_dict(),
            real_minus_process=float((err.real - err.process).mean()),
            real_minus_population=float((err.real - err.population).mean())))
    return dict(patients=len(frame), components=len(grouped),
        components_with_30_patients=int((frame.groupby('component').size() >= 30).sum()),
        raw_mae=dict(patient=errors.mean().to_dict(), component=grouped.mean().to_dict()),
        normalized_mae=dict(patient=normalized.mean().to_dict(), component=normalized_grouped.mean().to_dict()),
        comparisons=comparisons, per_component=groups, descriptive_screen_pass=bool(passed))


def verify_frame(frame, scaling, roles, fold):
    people = list(scaling['people']); variables = list(scaling['variables'])
    pi = {p: i for i, p in enumerate(people)}; vi = {v: i for i, v in enumerate(variables)}
    centers = np.asarray(scaling['centers'])
    expected = np.isfinite(centers) & np.array([roles[p] == 'test' for p in people])[:, None]
    actual = np.zeros_like(expected)
    ix = np.array([pi[p] for p in frame.patient]); iy = np.array([vi[v] for v in frame.target])
    actual[ix, iy] = True
    if frame.duplicated(['patient', 'target']).any() or not np.array_equal(actual, expected):
        raise ValueError('Incomplete or duplicate outer identity')
    if not (frame.fold == fold).all() or not np.allclose(frame.truth, centers[ix, iy], rtol=1e-6, atol=1e-6):
        raise ValueError('Outer label or role mismatch')
    if not np.isfinite(frame[['truth', 'fit_scale', *ARMS]].to_numpy()).all() or (frame.fit_scale <= 0).any():
        raise ValueError('Invalid prediction or scale')


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    out = Path('artifacts/native_reconstruction_additional_evaluation_20260924.json')
    if out.exists():
        raise ValueError('Existing assessment')
    fp = root / 'partitions_20260922.pt'; sp = root / 'scaling_20260922.pt'
    plan = torch.load(fp, weights_only=False); scaling = torch.load(sp, weights_only=False)
    provenance_path = root / 'provenance.parquet'
    links = pd.read_parquet(provenance_path)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    paths = [Path(__file__), fp, sp, provenance_path,
             Path('RECONSTRUCTION_ADDITIONAL_EVALUATION_PROTOCOL_20260924.md')]
    checked = {}; frames = []
    for fold in range(5):
        p = root / f'outer_20260924/fold{fold}.parquet'; mp = p.with_suffix('.json')
        meta = json.loads(mp.read_text())
        if sha256(p) != meta['prediction_sha256'] or meta['fold'] != fold:
            raise ValueError('Changed outer predictions')
        for name, digest in meta['bindings'].items():
            if name not in checked:
                checked[name] = sha256(Path(name))
            if checked[name] != digest:
                raise ValueError('Changed inference dependency')
        frame = pd.read_parquet(p)
        verify_frame(frame, scaling, plan['folds'][fold], fold)
        if not np.array_equal(frame.component.to_numpy(), frame.patient.map(components).to_numpy()):
            raise ValueError('Wrong component provenance')
        if len(frame) != meta['examples'] or frame.patient.nunique() != meta['patients']:
            raise ValueError('Wrong output counts')
        paths.extend((p, mp)); frames.append(frame)
        print('Verified fold', fold, len(frame), flush=True)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['patient', 'target']).any() or data.groupby('patient').fold.nunique().max() != 1:
        raise ValueError('Cross-fold identity duplication')
    if data.groupby('component').fold.nunique().max() != 1:
        raise ValueError('Cross-fold component overlap')
    if len(data) != int(np.isfinite(np.asarray(scaling['centers'])).sum()):
        raise ValueError('Incomplete total coverage')
    targets = sorted(data.target.unique())
    if len(targets) != FAMILY_SIZE:
        raise ValueError('Changed target family')
    results = {}
    for i, target in enumerate(targets):
        results[target] = summarize_target(data.loc[data.target == target], 20260924 + 1000 * i)
        print('Scored target', i + 1, 'of', FAMILY_SIZE, flush=True)
    errors = data[list(ARMS)].subtract(data.truth, axis=0).abs().div(data.fit_scale, axis=0)
    summary = dict(pairs=len(data), patients=int(data.patient.nunique()), targets=len(targets),
        components=int(data.component.nunique()),
        descriptive_passes=[t for t, r in results.items() if r['descriptive_screen_pass']],
        normalized_target_macro={level: {a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
            for a in ARMS} for level in ('patient', 'component')},
        normalized_patient_macro=errors.groupby(data.patient).mean().mean().to_dict(),
        patient_macro_definition='Equal patients after averaging available targets within each patient; target composition varies.')
    report = dict(summary=summary, targets=results, bindings={str(p):sha256(p) for p in paths},
        missing_cases=dict(accepted_prediction_missing=0, candidate_anchor_rejected_no_fresh_label=5467,
            candidate_anchor_rejected_no_independent_history=4429),
        runtime_modified=False, promotion_allowed=False,
        limitation='Masked current-value reconstruction. Conditional descriptive crossfit inference only; overlapping training and explicitly reused fold2 weights. Not untouched external validation, forecasting, calibration or promotion.')
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
