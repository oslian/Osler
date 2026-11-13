"""E29 saved-model verification with a separate explicit update schedule."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.organ_assimilation import CHANNELS
from scripts.train_organ_assimilation import (
    AssimilatedReconstruction, CONTROLS, DEST, NAMES, SEED, assess, old_frames, validate_identities,
)
from scripts.train_organ_binding import prepare, select_arm, write_json
from scripts.verify_organ_binding import verify_statistics
from scripts.verify_full_organ_readout import compare_results
from whole_body_jepa.multisource_intervals import sha256


def direct_prediction(candidate, example, bound):
    e = select_arm(example, bound, 'binding')
    layer = candidate.model.trajectory.layer
    bodies, organs, masks, times = [e[k] for k in ('body', 'organs', 'presence', 'times')]
    state = layer.initialize(bodies.new_zeros(1, layer.body_dim),
        bodies.new_zeros(1, layer.organ_count, layer.organ_dim), bodies.new_zeros(1, layer.organ_count))
    pending = organs.new_zeros(1, layer.organ_count * layer.organ_dim)
    for i, timestamp in enumerate(times):
        elapsed = (timestamp-times[0]).to(bodies.dtype).reshape(1) - state.elapsed_hours
        predicted = layer.advance(state, elapsed)
        observed = layer.update(predicted, bodies[i:i+1], organs[i:i+1], masks[i:i+1], elapsed)
        innovation = (observed.organs-predicted.organs).reshape(1, -1)
        visible = bool((masks[i] > 0).any())
        if visible:
            if candidate.channel == 'immediate':
                feature = innovation
            elif candidate.channel == 'delayed':
                feature = pending
            else:
                feature = (observed.body-predicted.body) @ candidate.bridge.body_expansion
            correction = torch.nn.functional.linear(feature, candidate.bridge.projection.weight).tanh() * candidate.bridge.limit
            state = replace(observed, body=observed.body + correction)
            pending = innovation
        else:
            state = observed
    state = layer.advance(state, (times.new_tensor(e['anchor'])-times[-1]).to(bodies.dtype).reshape(1))
    return float(candidate.model.readout(state.body)[0, e['target']])


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    ap = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    output = Path('artifacts/native_organ_assimilation_verification_20261004.json')
    if output.exists():
        raise ValueError('Completed verification exists')
    assessment = json.loads(ap.read_text())
    bindings = {str(p): sha256(p) for p in (ap, Path(__file__))}
    errors, frames, fit_frames = {}, [], []
    for fold in range(5):
        variables, examples, bound, original, parent, extraction, old_dest = prepare(fold)
        baseline, _ = old_frames(fold, examples, variables, old_dest)
        dest = DEST/f'fold{fold}'
        rp = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
        if sha256(rp) != assessment['bindings'][str(rp)]:
            raise ValueError('Changed E29 fold report')
        report = json.loads(rp.read_text())
        contract = json.loads((dest/'contract.json').read_text())
        if report['contract'] != contract or contract['parent'] != parent or contract['extraction'] != extraction:
            raise ValueError('Changed E29 contract')
        for path, digest in contract['bindings'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('Changed E29 dependency')
        roles = {}
        for role in examples:
            path = dest/f'{role}_predictions.parquet'
            if sha256(path) != report['prediction_hashes'][role]:
                raise ValueError('Changed E29 predictions')
            roles[role] = pd.read_parquet(path)
            validate_identities(roles[role])
            pd.testing.assert_frame_equal(roles[role][list(baseline[role])], baseline[role], check_exact=True)
            bindings[str(path)] = sha256(path)
        fold_errors = {}
        for channel in CHANNELS:
            path = dest/f'{channel}.pt'
            info = report['fits'][channel]
            if (info['sha256'] != sha256(path) or info['channel'] != channel
                    or info['contract_sha256'] != sha256(dest/'contract.json')):
                raise ValueError('Changed E29 checkpoint')
            candidate = AssimilatedReconstruction(original, channel).eval()
            candidate.load_state_dict(torch.load(path, weights_only=False), strict=True)
            for key, value in original.trajectory.jepa.state_dict().items():
                torch.testing.assert_close(candidate.model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
            fold_errors[channel] = {}
            with torch.no_grad():
                for role, rows in examples.items():
                    actual = np.array([direct_prediction(candidate, e, b) for e, b in zip(rows, bound[role])])
                    expected = roles[role][channel].to_numpy()
                    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
                    fold_errors[channel][role] = float(np.max(abs(actual-expected)))
            bindings[str(path)] = sha256(path)
        errors[str(fold)] = fold_errors
        frames.append(roles['selection']); fit_frames.append(roles['fit'])
        bindings[str(rp)] = sha256(rp)
        print('E29 verified', fold, fold_errors, flush=True)
    data, fitting = pd.concat(frames, ignore_index=True), pd.concat(fit_frames, ignore_index=True)
    validate_identities(data)
    validate_identities(fitting)
    results, roundoff = {}, 0.
    for i, variable in enumerate(variables):
        selected = data[data.variable == variable]
        actual = assess(selected, SEED+i*1000) if len(selected) else dict(status='unsupported')
        expected = dict(assessment['results'][variable])
        expected_fit = expected.pop('fit_patient_mae', None)
        roundoff = max(roundoff, verify_statistics(actual, expected))
        f = fitting[fitting.variable == variable]
        actual_fit = None
        if len(f):
            names = [n for n in NAMES if n != 'raw']
            actual_fit = f[names].subtract(f.truth, axis=0).abs().groupby(f.patient).mean().mean().to_dict()
        roundoff = max(roundoff, compare_results(actual_fit, expected_fit))
        results[variable] = actual
    scored = {v: r for v, r in results.items() if r['status'] == 'scored'}
    summary = dict(patients=int(data.patient.nunique()), occurrences=len(data), targets=len(scored),
        normalized_macro={level: {n: float(np.mean([r['normalized_mae'][level][n] for r in scored.values()]))
                                  for n in NAMES} for level in ('patient', 'component')},
        screens={c: [v for v, r in scored.items() if r['comparisons'][c]['screen_pass']] for c in CONTROLS})
    summary['mechanism_increment'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS[:3])))
    summary['full_screen'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS)))
    roundoff = max(roundoff, compare_results(summary, assessment['summary']))
    write_json(output, dict(bindings=bindings, saved_prediction_errors=errors,
        maximum_mean_roundoff=roundoff, complete=True, runtime_modified=False,
        promotion_allowed=False, limitation='Independent schedule, shared verified native update and statistics primitives.'))


if __name__ == '__main__':
    main()
