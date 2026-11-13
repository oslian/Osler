"""Reload E44 ablations and independently aggregate every saved query."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import audit_tangent_spectrum as audit
from scripts.verify_tangent_bridge import scalar_results
from scripts.verify_frozen_bridge import compare, mean

OUTPUT = Path('artifacts/native_tangent_spectrum_verification_20261005.json')


def verify_saved_rows(frame, original, alpha):
    audit.base.validate_identities(frame)
    pd.testing.assert_frame_equal(frame[list(original)], original, check_exact=True)
    required = [*audit.VARIANTS, 'strong_slope', 'weak_slope', 'null_slope', 'roundoff_slope']
    if not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError('Nonfinite saved ablation')
    error = 0.
    for row in frame.itertuples():
        difference = abs(row.strong_slope + row.weak_slope + row.null_slope
                         + row.roundoff_slope - row.tangent_slope)
        if difference > 1e-7 * (1 + abs(row.tangent_slope)):
            raise ValueError('Saved tangent decomposition changed')
        error = max(error, difference)
        for name in ('strong', 'weak'):
            expected = row.tangent_base + alpha * getattr(row, f'{name}_slope')
            if getattr(row, f'{name}_linear') != expected:
                raise ValueError('Saved linear ablation changed')
    return error


def scalar_variants(frame, inventory):
    return {v: scalar_results(frame.assign(tangent_step=frame[v]), inventory)
            for v in audit.VARIANTS}


def macro_summary(results):
    output = {}
    for role, variants in results.items():
        output[role] = {}
        for variant, targets in variants.items():
            scored = [r for r in targets.values() if r['status'] == 'scored']
            output[role][variant] = dict(scored_targets=len(scored), **{
                metric: {level: mean([r[metric][level]['tangent_step'] for r in scored])
                         for level in ('patient', 'component')}
                for metric in ('mae', 'smooth_l1')})
    return output


def main():
    if OUTPUT.exists():
        raise ValueError('Completed verification exists')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = json.loads(audit.OUTPUT.read_text())
    if not report['complete'] or report['runtime_modified'] or report['promotion_allowed']:
        raise ValueError('Invalid E44 report state')
    audit.base.check_bindings(report['bindings'])
    paths = [audit.OUTPUT, Path(__file__), Path('tests/test_tangent_spectrum_verification.py'),
             Path('scripts/verify_tangent_bridge.py'), Path('scripts/verify_frozen_bridge.py')]
    data = {r: [] for r in audit.base.ROLES}
    replays = []
    for fold in range(5):
        variables, examples, bound, original, *_ = audit.parent.previous.prepare_existing(fold)
        dest = audit.DEST / f'fold{fold}'
        old = audit.parent.DEST / f'fold{fold}'
        info = json.loads((dest / 'complete.json').read_text())
        source = json.loads((old / 'complete.json').read_text())
        if not info['complete'] or info['alpha'] != source['accepted_alpha']:
            raise ValueError('Changed fixed step')
        audit.base.check_bindings(info['bindings'])
        audit.base.check_bindings(info['outputs'])
        parts = torch.load(dest / 'parts.pt', weights_only=True)
        direction = torch.load(old / 'direction.pt', weights_only=True).double()
        if set(parts) != {'strong', 'weak', 'null', 'roundoff'}:
            raise ValueError('Incomplete parameter decomposition')
        for part in parts.values():
            if part.shape != direction.shape or not torch.isfinite(part).all():
                raise ValueError('Invalid saved direction')
        # Parts were independently cast to float32 before saving as float64.
        torch.testing.assert_close(sum(parts.values()), direction, atol=1e-12, rtol=1e-12)
        for name, part in parts.items():
            energy = (info['spectrum']['cast_roundoff_norm2'] if name == 'roundoff'
                      else info['spectrum']['energy'][name])
            np.testing.assert_allclose(float(part @ part), energy,
                                       atol=1e-8, rtol=1e-6)
        models = {}
        for name in ('strong', 'weak'):
            model = audit.base.AssimilatedReconstruction(original, 'immediate').eval().requires_grad_(False)
            with torch.no_grad():
                model.bridge.projection.weight.copy_(
                    (info['alpha'] * parts[name]).reshape_as(model.bridge.projection.weight))
            audit.base.assert_frozen(model, original)
            models[name] = model
        for role in audit.base.ROLES:
            frame = pd.read_parquet(dest / f'{role}.parquet')
            prior = pd.read_parquet(old / f'{role}.parquet')
            slope_error = verify_saved_rows(frame, prior, info['alpha'])
            rows = audit.role_rows(role, frame, examples, bound, original, variables, fold)
            maximum = 0.
            count = 0
            with torch.inference_mode():
                for count, example in enumerate(rows, 1):
                    for name, model in models.items():
                        # Main audit used explicit direct_prediction; this uses forward.
                        actual = float(model(example))
                        error = abs(actual - frame[f'{name}_actual'].iloc[count-1])
                        maximum = max(maximum, error)
                        if error > 1e-7:
                            raise ValueError('Reloaded ablation replay mismatch')
            if count != len(frame):
                raise ValueError('Incomplete saved replay')
            data[role].append(frame)
            replays.append(dict(fold=fold, role=role, rows=count, max_error=maximum,
                                slope_sum_max_error=slope_error))
            print('E44 verified', fold, role, count, maximum, flush=True)
        paths.extend([dest / 'complete.json', *map(Path, info['outputs'])])
    maximum = 0.
    actual = {}
    for role, frames in data.items():
        actual[role] = scalar_variants(pd.concat(frames, ignore_index=True), variables)
        maximum = max(maximum, compare(actual[role], report['results'][role]))
    audit.base.check_bindings(report['bindings'])
    audit.base.write_json(OUTPUT, dict(complete=True, replay=replays,
        scalar_max_error=maximum, summary=macro_summary(actual),
        bindings={str(p): audit.base.sha256(p) for p in paths},
        runtime_modified=False, promotion_allowed=False))
    print('E44 independent verification complete', maximum, flush=True)


if __name__ == '__main__':
    main()
