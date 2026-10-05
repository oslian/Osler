"""Fixed epoch-10 versus epoch-8 paired assessment on the original outer grid."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import (
    FAMILY_SIZE, paired_summary, summarize_target, verify_frame,
)
from whole_body_jepa.multisource_intervals import sha256


def verify_bound_files(bindings, checked):
    if not bindings:
        raise ValueError('Missing inference dependencies')
    for name, digest in bindings.items():
        path = Path(name).resolve()
        if path not in checked:
            checked[path] = sha256(path)
        if checked[path] != digest:
            raise ValueError('Changed inference dependency: ' + name)


def verify_prediction_pair(base, fresh, meta, fold, components):
    if (meta['fold'] != fold or meta['role'] != 'test'
            or meta['source_fold'] != (0 if fold == 2 else fold)):
        raise ValueError('Invalid continued prediction role or fitted fold')
    if meta['candidate_only'] is not True or meta['promotion_allowed'] is not False:
        raise ValueError('Unexpected promotion status')
    pd.testing.assert_frame_equal(base[['patient', 'target', 'fold']],
                                  fresh[['patient', 'target', 'fold']])
    if len(fresh) != meta['examples'] or fresh.patient.nunique() != meta['patients']:
        raise ValueError('Incomplete continued inference')
    if not np.array_equal(base.component.to_numpy(), base.patient.map(components).to_numpy()):
        raise ValueError('Wrong component provenance')
    if not np.isfinite(fresh[['real10', 'process10']]).all().all():
        raise ValueError('Nonfinite continued prediction')


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    output = Path('artifacts/native_reconstruction_continuation_outer_20260929.json')
    if output.exists():
        raise ValueError('Existing outer assessment')
    partitions = root / 'partitions_20260922.pt'
    scaling_path = root / 'scaling_20260922.pt'
    plan = torch.load(partitions, weights_only=False)
    scaling = torch.load(scaling_path, weights_only=False)
    provenance_path = root / 'provenance.parquet'
    links = pd.read_parquet(provenance_path)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    prior_path = Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json')
    paths = [Path(__file__), partitions, scaling_path,
             Path('scripts/score_additional_reconstruction.py'), provenance_path, prior_path]
    checked = {}
    frames = []
    for fold in range(5):
        base_path = root / f'coverage_outer_20260927/fold{fold}.parquet'
        new_path = root / f'continuation_outer_20260929/fold{fold}.parquet'
        base_meta = json.loads(base_path.with_suffix('.json').read_text())
        new_meta = json.loads(new_path.with_suffix('.json').read_text())
        if sha256(base_path) != base_meta['prediction_sha256'] or sha256(new_path) != new_meta['prediction_sha256']:
            raise ValueError('Changed paired prediction file')
        verify_bound_files(new_meta['bindings'], checked)
        base = pd.read_parquet(base_path)
        fresh = pd.read_parquet(new_path)
        verify_frame(base, scaling, plan['folds'][fold], fold)
        verify_prediction_pair(base, fresh, new_meta, fold, components)
        variable_indices = {variable: i for i, variable in enumerate(scaling['variables'])}
        expected_scale = np.asarray(scaling['scalers'][fold]['scale'])[
            base.target.map(variable_indices).to_numpy()]
        if not np.array_equal(base.fit_scale.to_numpy(), expected_scale):
            raise ValueError('Wrong fit-only normalization')
        frame = base.rename(columns={'real': 'real8', 'process': 'process8'}).copy()
        frame['real'] = fresh.real10.to_numpy()
        frame['process'] = fresh.process10.to_numpy()
        if not np.isfinite(frame[['real', 'process']]).all().all():
            raise ValueError('Nonfinite continued prediction')
        frames.append(frame)
        paths.extend((base_path, base_path.with_suffix('.json'), new_path,
                      new_path.with_suffix('.json')))
        print('Verified fold', fold, len(frame), flush=True)
    data = pd.concat(frames, ignore_index=True)
    if (data.duplicated(['patient', 'target']).any()
            or data.groupby('patient').fold.nunique().max() != 1
            or len(data) != int(np.isfinite(scaling['centers']).sum())):
        raise ValueError('Incomplete total outer grid')
    if data.groupby('component').fold.nunique().max() != 1:
        raise ValueError('Component split violation')
    results = {}
    for number, (target, frame) in enumerate(data.groupby('target', sort=True), 1):
        assessment = summarize_target(frame, 20260929 + 1000 * number)
        new_error = (frame.real - frame.truth).abs()
        old_error = (frame.real8 - frame.truth).abs()
        contrast = new_error - old_error
        groups = contrast.groupby(frame.component).mean()
        paired = dict(patient=paired_summary(contrast, 20260929 + number),
                      component=paired_summary(groups, 20261929 + number))
        incremental = len(frame) >= 30 and len(groups) >= 3 and all(
            stat['p_less'] is not None and stat['p_less'] < .05 / FAMILY_SIZE
            and stat['mean'] < 0 for stat in paired.values()
        )
        results[target] = dict(base_screen=assessment['descriptive_screen_pass'],
                               incremental_screen=bool(incremental),
                               comparisons=paired, base_assessment=assessment,
                               raw_mae=dict(new_patient=float(new_error.mean()),
                                            old_patient=float(old_error.mean()),
                                            new_component=float(new_error.groupby(frame.component).mean().mean()),
                                            old_component=float(old_error.groupby(frame.component).mean().mean())),
                               component_harms=int((groups > 0).sum()),
                               component_count=len(groups))
        print('Scored', number, 'of', FAMILY_SIZE, flush=True)
    if len(results) != FAMILY_SIZE:
        raise ValueError('Incomplete target family')
    original = json.loads(prior_path.read_text())
    prior = set(original['summary']['base_screen_passes'])
    passes = {target for target, result in results.items() if result['base_screen']}
    summary = dict(pairs=len(data), patients=int(data.patient.nunique()),
                   base_screen_passes=sorted(passes),
                   retained=sorted(prior & passes),
                   lost=sorted(prior - passes), added=sorted(passes - prior),
                   incremental_screen_passes=sorted(
                       target for target, result in results.items() if result['incremental_screen']
                   ),
                   joint_screen_passes=sorted(target for target, result in results.items()
                                              if result['base_screen'] and result['incremental_screen']),
                   normalized_target_macro={level: {
                       arm: float(np.mean([
                           result['base_assessment']['normalized_mae'][level][arm]
                           for result in results.values()
                       ])) for arm in ('real', 'process', 'population')
                   } for level in ('patient', 'component')})
    report = dict(summary=summary, targets=results,
                  bindings={str(path): sha256(path) for path in paths},
                  promotion_allowed=False, runtime_modified=False,
                  limitation='Conditional current-value reconstruction assessment on reused development data. Fold2 shares training with fold0, and other fits overlap. Passing descriptive screens is not fresh external confirmation or forecast promotion.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
