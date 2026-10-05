"""Frozen all-target outer assessment of the unchanged nonlinear source probes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import FAMILY_SIZE, verify_frame
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.score_nonlinear_probe import ARMS, CONTRASTS, assess_target
from whole_body_jepa.multisource_intervals import sha256


def verify_outer_probe_frame(frame, base, native, fold):
    keys = ['patient', 'target']
    if any(x.duplicated(keys).any() for x in (frame, base, native)):
        raise ValueError('Duplicate outer identity')
    a, b, c = [x.set_index(keys).sort_index() for x in (frame, base, native)]
    if (not a.index.equals(b.index) or not a.index.equals(c.index)
            or not a.fold.eq(fold).all() or not b.fold.eq(fold).all()
            or not c.fold.eq(fold).all() or not a.component.equals(b.component)):
        raise ValueError('Changed outer identity or provenance')
    if not np.isfinite(a[['truth', *ARMS]]).all().all() or not b.fit_scale.gt(0).all():
        raise ValueError('Nonfinite outer predictions')
    # Normalized differences remove the location without reconstructing it here.
    for arm, values in (('native', c.real10), ('population', b.population)):
        expected = (values - b.truth) / b.fit_scale
        if not np.allclose(a[arm] - a.truth, expected, rtol=1e-10, atol=1e-10):
            raise ValueError('Changed native/reference residual')


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    private = root / 'nonlinear_probe_20261001/outer_20261002'
    output = Path('artifacts/native_reconstruction_nonlinear_outer_assessment_20261002.json')
    if output.exists():
        raise ValueError('Existing outer assessment')
    fp, sp = root / 'partitions_20260922.pt', root / 'scaling_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds']
    scaling = torch.load(sp, weights_only=False)
    old_path = Path('artifacts/native_reconstruction_continuation_outer_20260929.json')
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    old, source = json.loads(old_path.read_text()), json.loads(source_path.read_text())
    targets = sorted(source['targets'])
    if len(targets) != FAMILY_SIZE:
        raise ValueError('Changed target family')
    checked = {}
    for report in (old, source):
        verify_bound_files(report['bindings'], checked)
    paths = [Path(__file__), fp, sp, old_path, source_path,
             Path('scripts/score_nonlinear_probe.py'), Path('scripts/score_additional_reconstruction.py'),
             Path('scripts/score_monitored_continuation_outer.py'),
             Path('RECONSTRUCTION_NONLINEAR_OUTER_PROTOCOL_20261002.md')]
    frames, folds = [], {}
    for fold in range(5):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_outer_fold{fold}_20261002.json')
        report = json.loads(rp.read_text())
        verify_bound_files(report['bindings'], checked)
        path, features = private / f'fold{fold}.parquet', private / f'fold{fold}_features.sqlite'
        if ((report['fold'], report['role']) != (fold, 'test') or report['promotion_allowed'] is not False
                or sha256(path) != report['prediction_sha256'] or sha256(features) != report['feature_sha256']):
            raise ValueError('Wrong outer artifact identity')
        bp = root / f'coverage_outer_20260927/fold{fold}.parquet'
        npth = root / f'continuation_outer_20260929/fold{fold}.parquet'
        base, native, frame = pd.read_parquet(bp), pd.read_parquet(npth), pd.read_parquet(path)
        verify_frame(base, scaling, roles[fold], fold)
        verify_outer_probe_frame(frame, base, native, fold)
        if (len(frame) != report['examples'] or frame.patient.nunique() != report['patients']
                or frame.target.nunique() != report['targets']
                or sorted(set(targets) - set(frame.target)) != report['unsupported_targets']):
            raise ValueError('Incomplete outer support')
        vi = {v: i for i, v in enumerate(scaling['variables'])}
        ix = frame.target.map(vi).to_numpy()
        scaler = scaling['scalers'][fold]
        b = base.set_index(['patient', 'target']).loc[list(zip(frame.patient, frame.target))]
        if not np.allclose(frame.truth, (b.truth.to_numpy() - np.asarray(scaler['location'])[ix]) /
                           np.asarray(scaler['scale'])[ix], rtol=1e-10, atol=1e-10):
            raise ValueError('Wrong normalized labels')
        rows = {}
        for target in targets:
            part = frame.loc[frame.target == target]
            if part.empty:
                rows[target] = None
                continue
            error = part[list(ARMS)].subtract(part.truth, axis=0).abs()
            rows[target] = dict(patients=len(part), normalized_mae=dict(
                patient=error.mean().to_dict(), component=error.groupby(part.component).mean().mean().to_dict()))
        folds[str(fold)] = rows
        paths.extend((rp, path, features, bp, npth))
        frames.append(frame)
        print('Verified frozen outer fold', fold, len(frame), flush=True)
    data = pd.concat(frames, ignore_index=True)
    if (data.duplicated(['patient', 'target']).any() or data.groupby('patient').fold.nunique().max() != 1
            or data.groupby('component').fold.nunique().max() != 1
            or len(data) != int(np.isfinite(scaling['centers']).sum()) or sorted(data.target.unique()) != targets):
        raise ValueError('Incomplete independent outer grid')
    results = {}
    for i, target in enumerate(targets):
        result = assess_target(data.loc[data.target == target], 20261002 + 1000 * i)
        for level in ('patient', 'component'):
            expected = old['targets'][target]['base_assessment']['normalized_mae'][level]['real']
            if not np.isclose(result['normalized_mae'][level]['native'], expected, rtol=1e-10, atol=1e-10):
                raise ValueError('Original outer aggregate changed')
        results[target] = result
        print('Scored frozen outer target', i + 1, 'of', FAMILY_SIZE, flush=True)
    names = [c + '_minus_' + r for c, r in CONTRASTS]
    passes = {name: [t for t, r in results.items() if r['comparisons'][name]['all_8_contrasts_47_screen']]
              for name in names}
    summary = dict(pairs=len(data), patients=int(data.patient.nunique()), targets=len(results),
        normalized_target_macro={level: {arm: float(np.mean([r['normalized_mae'][level][arm]
            for r in results.values()])) for arm in ARMS} for level in ('patient', 'component')},
        all_contrasts_screen_passes=passes,
        joint_incremental_and_controls={arm: sorted(set(passes[arm + '_minus_native']) &
            set(passes[arm + '_minus_process_summary']) & set(passes[arm + '_minus_population']))
            for arm in ('body_probe', 'raw_probe')},
        source_outer_screen_overlap={name: sorted(set(passes[name]) &
            set(source['summary']['all_contrasts_screen_passes'][name])) for name in names})
    output.write_text(json.dumps(dict(summary=summary, targets=results, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='All47/five-fold frozen outer current reconstruction. No head/target selection or refit after source assessment. Every patient and component belongs to one outer fold; training fits overlap. 47-family plus8x47 descriptive screens, not complete runtime promotion. Reused development, not untouched external validation; no future forecast or intervals. Matched tree settings but differing feature dimensions and native loss/optimizer; gains do not isolate encoder information loss.'),
        indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
