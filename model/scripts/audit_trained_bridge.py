"""Localize a trained bridge's inference effect without refitting weights."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_organ_assignment import pools
from scripts.diagnose_organ_assimilation import require_verification
from scripts.encoder_fold_scaling import transform_example
from scripts.organ_assimilation import CHANNELS
from scripts.probe_assimilation_transfer import DEST as TRANSFER, ROOT
from scripts.train_content_retention import replay
from scripts.train_organ_assimilation import AssimilatedReconstruction, DEST as TRAINED, validate_identities
from scripts.train_organ_binding import prepare, select_arm, space_required, write_json
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.reconstruction_cache import ReconstructionCache

PROTOCOL = Path('RECONSTRUCTION_BRIDGE_ABLATION_PROTOCOL_20261005.md')
DEST = ROOT/'trained_bridge_ablation_20261005'
OUTPUT = Path('artifacts/native_trained_bridge_ablation_20261005.json')
ROLES = ('fit', 'selection', 'same_source')


def disable_bridge(candidate):
    result = deepcopy(candidate).eval()
    result.requires_grad_(False)
    with torch.no_grad():
        result.bridge.projection.weight.zero_()
    for key, value in candidate.state_dict().items():
        actual = result.state_dict()[key]
        expected = torch.zeros_like(value) if key == 'bridge.projection.weight' else value
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return result


def summarize(frame, inventory):
    validate_identities(frame)
    columns = ['truth', 'unbridged', 'frozen', *CHANNELS, *[c+'_off' for c in CHANNELS]]
    if 'raw' in frame and not np.isfinite(frame.raw.to_numpy()).all():
        raise ValueError('Nonfinite raw reference')
    if (frame.empty or frame.duplicated(['patient', 'fold']).any()
            or (frame.groupby('patient').component.nunique() > 1).any()
            or not set(frame.variable) <= set(inventory)
            or not np.isfinite(frame[columns].to_numpy()).all()):
        raise ValueError('Invalid paired rows')
    results = {}
    for variable in inventory:
        f = frame[frame.variable == variable]
        if f.empty:
            results[variable] = dict(status='unsupported')
            continue
        predictions = columns[1:] + (['raw'] if 'raw' in f else [])
        errors = f[predictions].subtract(f.truth, axis=0).abs().groupby(f.patient).mean()
        link = f.drop_duplicates('patient').set_index('patient').component
        groups = errors.groupby(link.reindex(errors.index)).mean()
        contrasts = {}
        for channel in CHANNELS:
            contrasts[channel] = {}
            for name, values in (('patient', errors), ('component', groups)):
                bridge = values[channel] - values[channel+'_off']
                remaining = values[channel+'_off'] - values.unbridged
                total = values[channel] - values.unbridged
                np.testing.assert_allclose(total, bridge+remaining, atol=1e-12, rtol=1e-12)
                contrasts[channel][name] = dict(bridge=float(bridge.mean()),
                    remaining=float(remaining.mean()), total=float(total.mean()))
        results[variable] = dict(status='scored', patients=len(errors), occurrences=len(f),
            components=len(groups), normalized_mae=dict(patient=errors.mean().to_dict(),
            component=groups.mean().to_dict()), contrasts=contrasts)
    return results


def bound_hashes(paths):
    return {str(p): sha256(p) for p in paths}


def check_bindings(bindings):
    for path, digest in bindings.items():
        if sha256(Path(path)) != digest:
            raise ValueError('Changed dependency: '+str(path))


def same_source_examples(frame, original, variables, fold):
    scaler = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)['scalers'][fold]
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    try:
        with torch.inference_mode():
            for row in frame.itertuples(index=False):
                e = transform_example(source.get(row.patient, row.context), scaler)
                if variables[row.context] != row.variable or float(e['truth_normalized']) != row.truth:
                    raise ValueError('Changed same-source identity/truth')
                clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, row.variable)
                body, organs, presence = encode_native_organ_observations(original.trajectory.jepa, variables, clean)
                native, bound = pools(original.trajectory.jepa, variables, clean)
                torch.testing.assert_close(native, organs, atol=1e-6, rtol=0)
                yield dict(body=body, organs=bound, presence=presence, times=times,
                    anchor=float(e['anchor_hour']), target=row.context)
    finally:
        source.close()


def run_fold(fold, assessments):
    variables, examples, bound, original, _, _, _ = prepare(fold)
    paths = [PROTOCOL, Path(__file__), Path('tests/test_trained_bridge.py')]
    reports = {}
    for label, pattern in (('train', 'native_organ_assimilation_fold{}_20261004.json'),
                           ('transfer', 'native_assimilation_transfer_fold{}_20261005.json')):
        path = Path('artifacts')/pattern.format(fold)
        if sha256(path) != assessments[label]['bindings'][str(path)]:
            raise ValueError('Changed parent fold')
        reports[label] = json.loads(path.read_text())
        paths.append(path)
    check_bindings(reports['train']['contract']['bindings'])
    check_bindings(reports['transfer']['bindings'])
    models, off_models = {}, {}
    for channel in CHANNELS:
        path = TRAINED/f'fold{fold}'/f'{channel}.pt'
        if sha256(path) != reports['train']['fits'][channel]['sha256']:
            raise ValueError('Changed trained checkpoint')
        model = AssimilatedReconstruction(original, channel).eval()
        model.load_state_dict(torch.load(path, weights_only=False), strict=True)
        model.requires_grad_(False)
        models[channel], off_models[channel] = model, disable_bridge(model)
        paths.append(path)
    frames = {}
    for role in ROLES:
        path = (TRANSFER/f'fold{fold}'/'predictions.parquet' if role == 'same_source'
                else TRAINED/f'fold{fold}'/f'{role}_predictions.parquet')
        digest = (reports['transfer']['prediction_sha256'] if role == 'same_source'
                  else reports['train']['prediction_hashes'][role])
        if sha256(path) != digest:
            raise ValueError('Changed saved predictions')
        frames[role] = pd.read_parquet(path)
        validate_identities(frames[role])
        if len(frames[role]) != (2048 if role == 'fit' else 1024):
            raise ValueError('Changed role size')
        paths.append(path)
    if any(set(frames[a].patient) & set(frames[b].patient)
           for a, b in (('fit', 'selection'), ('fit', 'same_source'), ('selection', 'same_source'))):
        raise ValueError('Patient role overlap')
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = Path(f'artifacts/native_trained_bridge_ablation_fold{fold}_20261005.json')
    bindings = bound_hashes(paths)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['bindings'] != bindings:
            raise ValueError('Changed completed ablation contract')
        check_bindings(meta['outputs'])
        return {r: pd.read_parquet(dest/f'{r}.parquet') for r in ROLES}, meta_path, variables
    parity = {r: {c: dict(on=0., off=0.) for c in CHANNELS} for r in ROLES}
    with torch.inference_mode():
        for role, frame in frames.items():
            if role == 'same_source':
                rows = same_source_examples(frame, original, variables, fold)
            else:
                expected = [(e['patient'], e['component'], e['target'], e['truth']) for e in examples[role]]
                actual = list(frame[['patient', 'component', 'context', 'truth']].itertuples(index=False, name=None))
                if expected != actual:
                    raise ValueError('Changed cached identities')
                rows = (select_arm(e, b, 'binding') for e, b in zip(examples[role], bound[role]))
            off_values = {c: [] for c in CHANNELS}
            count = 0
            for i, e in enumerate(rows):
                for c in CHANNELS:
                    on, off = float(models[c](e)), float(off_models[c](e))
                    native = float(models[c].model.readout(replay(models[c].model.trajectory.layer, e))[0, e['target']])
                    np.testing.assert_allclose(on, float(frame.iloc[i][c]), atol=1e-7, rtol=0)
                    np.testing.assert_allclose(off, native, atol=1e-7, rtol=0)
                    if not np.isfinite([on, off, native]).all():
                        raise ValueError('Nonfinite replay')
                    parity[role][c]['on'] = max(parity[role][c]['on'], abs(on-float(frame.iloc[i][c])))
                    parity[role][c]['off'] = max(parity[role][c]['off'], abs(off-native))
                    off_values[c].append(off)
                count += 1
                if count % 512 == 0:
                    print('E33 replay', fold, role, count, len(frame), flush=True)
            if count != len(frame):
                raise ValueError('Incomplete replay')
            for c in CHANNELS:
                frame[c+'_off'] = off_values[c]
            summarize(frame, variables)
    # No checkpoint tensors may be changed by the audit, including buffers.
    for c, model in models.items():
        saved = torch.load(TRAINED/f'fold{fold}'/f'{c}.pt', weights_only=False)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
            expected = torch.zeros_like(value) if key == 'bridge.projection.weight' else value
            torch.testing.assert_close(off_models[c].state_dict()[key], expected, atol=0, rtol=0)
    check_bindings(bindings)
    outputs = {}
    space_required()
    for role, frame in frames.items():
        path = dest/f'{role}.parquet'
        tmp = path.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(path)
        outputs[str(path)] = sha256(path)
    write_json(meta_path, dict(bindings=bindings, outputs=outputs, parity=parity,
        results={r: summarize(f, variables) for r, f in frames.items()},
        runtime_modified=False, promotion_allowed=False))
    return frames, meta_path, variables


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    if OUTPUT.exists():
        raise ValueError('Completed audit exists')
    assessments, bindings = {}, bound_hashes([PROTOCOL, Path(__file__)])
    for label, stem, date in (('train', 'native_organ_assimilation', '20261004'),
                               ('transfer', 'native_assimilation_transfer', '20261005')):
        suffix = '_assessment' if label == 'train' else ''
        ap = Path(f'artifacts/{stem}{suffix}_{date}.json')
        vp = Path(f'artifacts/{stem}_verification_{date}.json')
        require_verification(ap, json.loads(vp.read_text()))
        assessments[label] = json.loads(ap.read_text())
        bindings.update(bound_hashes([ap, vp]))
    frames = {r: [] for r in ROLES}
    for fold in range(5):
        result, path, variables = run_fold(fold, assessments)
        for role in ROLES:
            frames[role].append(result[role])
        bindings[str(path)] = sha256(path)
    results = {r: summarize(pd.concat(fs, ignore_index=True), variables) for r, fs in frames.items()}
    write_json(OUTPUT, dict(results=results, bindings=bindings,
        occurrences={r: sum(map(len, fs)) for r, fs in frames.items()},
        runtime_modified=False, promotion_allowed=False,
        limitation='Fixed-checkpoint inference-path ablation, not isolated training causality or promotion.'))
    print('E33 complete', {r: sum(map(len, fs)) for r, fs in frames.items()}, flush=True)


if __name__ == '__main__':
    main()
