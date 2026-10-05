"""Fixed E29 same-body assimilation policy comparison; no serving integration."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.organ_assimilation import CHANNELS, OrganAssimilation, replay_assimilated
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_organ_binding import prepare, raw_control, select_arm, space_required, write_json
from scripts.train_content_retention import ROOT
from whole_body_jepa.multisource_intervals import sha256


DEST = ROOT / 'organ_assimilation_20261004'
PROTOCOL = Path('RECONSTRUCTION_ORGAN_ASSIMILATION_PROTOCOL_20261004.md')
SEED = 20261004
CONTROLS = ('delayed', 'body', 'unbridged', 'frozen', 'raw')
NAMES = ('immediate', *CONTROLS)


def validate_identities(frame):
    keys = ['patient', 'component', 'variable', 'fold', 'context']
    if (not set(keys) <= set(frame) or frame[keys].isna().any().any()
            or any(not isinstance(v, str) or not v.strip()
                   for key in ('patient', 'component', 'variable') for v in frame[key])
            or not pd.api.types.is_integer_dtype(frame.fold.dtype)
            or not pd.api.types.is_integer_dtype(frame.context.dtype)
            or not frame.fold.isin(range(5)).all() or not frame.context.isin(range(53)).all()
            or frame.duplicated(['patient', 'fold']).any()
            or (frame.groupby('patient').component.nunique() > 1).any()
            or (frame.groupby('variable').context.nunique() > 1).any()
            or (frame.groupby('context').variable.nunique() > 1).any()):
        raise ValueError('Invalid or missing patient/context identities')


class AssimilatedReconstruction(nn.Module):
    def __init__(self, original, channel):
        super().__init__()
        if channel not in CHANNELS:
            raise ValueError('Unknown assimilation channel')
        self.channel = channel
        self.model = deepcopy(original).eval()
        self.bridge = OrganAssimilation(self.model.trajectory.layer)

    def forward(self, example):
        state = replay_assimilated(self.model.trajectory.layer, self.bridge, example, self.channel)
        return self.model.readout(state.body)[0, example['target']]


def fit_arm(original, examples, bound, channel):
    if not examples or len(examples) != len(bound):
        raise ValueError('Aligned nonempty fitting contexts required')
    torch.manual_seed(SEED)
    candidate = AssimilatedReconstruction(original, channel).eval()
    for p in candidate.parameters():
        p.requires_grad_(False)
    parameters = [*candidate.model.trajectory.layer.parameters(),
                  *candidate.model.readout.parameters(), *candidate.bridge.parameters()]
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(SEED)
    log = []
    for epoch in range(2):
        losses = []
        for index in rng.permutation(len(examples)):
            e = select_arm(examples[index], bound[index], 'binding')
            prediction = candidate(e)
            loss = F.smooth_l1_loss(prediction, prediction.new_tensor(e['truth']))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fitting loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        log.append(dict(epoch=epoch+1, steps=len(losses), loss=float(np.mean(losses))))
        print('E29 trained', channel, log[-1], flush=True)
    for key, value in original.trajectory.jepa.state_dict().items():
        torch.testing.assert_close(candidate.model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
    return candidate, log


def predict(candidate, examples, bound):
    if len(examples) != len(bound):
        raise ValueError('Prediction alignment mismatch')
    summary = dict(coordinates=0, saturated=0, absolute_sum=0., maximum=0.)
    def observe(_module, _inputs, output):
        values = torch.tanh(output.detach()).abs()
        summary['coordinates'] += values.numel()
        summary['saturated'] += int((values >= .99).sum())
        summary['absolute_sum'] += float(values.sum()) * candidate.bridge.limit
        summary['maximum'] = max(summary['maximum'], float(values.max()) * candidate.bridge.limit)
    hook = candidate.bridge.projection.register_forward_hook(observe)
    try:
        with torch.no_grad():
            predictions = np.array([float(candidate(select_arm(e, b, 'binding')))
                                    for e, b in zip(examples, bound)])
    finally:
        hook.remove()
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite prediction')
    return predictions, summary


def old_frames(fold, examples, variables, old_dest):
    report_path = Path(f'artifacts/native_organ_binding_fold{fold}_20261004.json')
    assessment = json.loads(Path('artifacts/native_organ_binding_assessment_20261004.json').read_text())
    if sha256(report_path) != assessment['bindings'][str(report_path)]:
        raise ValueError('Changed E27 report')
    report = json.loads(report_path.read_text())
    if report['contract'] != json.loads((old_dest/'contract.json').read_text()):
        raise ValueError('Changed E27 baseline contract')
    frames = {}
    bindings = {str(report_path): sha256(report_path)}
    for role, rows in examples.items():
        path = old_dest/f'{role}_predictions.parquet'
        if sha256(path) != report['prediction_hashes'][role]:
            raise ValueError('Changed E27 predictions')
        frame = pd.read_parquet(path)
        validate_identities(frame)
        expected = [(e['patient'], e['component'], variables[e['target']], fold, e['target'], e['truth'], e['frozen']) for e in rows]
        if list(frame[['patient', 'component', 'variable', 'fold', 'context', 'truth', 'frozen']].itertuples(index=False, name=None)) != expected:
            raise ValueError('Changed E27 identities or truth')
        base = frame[['patient', 'component', 'context', 'variable', 'fold', 'truth', 'frozen']].copy()
        base['unbridged'] = frame.binding
        base['zero_gap'] = [e['anchor'] == float(e['times'][-1]) for e in rows]
        base['event_count'] = [int((e['presence'] > 0).any(dim=1).sum()) for e in rows]
        frames[role] = base
        bindings[str(path)] = sha256(path)
    raw, raw_bindings = raw_control(fold, examples['selection'], variables)
    frames['selection']['raw'] = raw
    bindings.update(raw_bindings)
    return frames, bindings


def assess(frame, seed, repetitions=5000):
    validate_identities(frame)
    if (frame.empty or frame.variable.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *NAMES]].to_numpy()).all()):
        raise ValueError('Invalid paired assessment')
    losses = frame[list(NAMES)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = losses.groupby(links.reindex(losses.index)).mean()
    comparisons = {}
    for i, control in enumerate(CONTROLS):
        stats = dict(patient=paired_summary(losses.immediate-losses[control], seed+i, repetitions),
                     component=paired_summary(groups.immediate-groups[control], seed+100+i, repetitions))
        passed = len(losses) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*5) for s in stats.values())
        comparisons[control] = dict(paired=stats, screen_pass=bool(passed))
    return dict(status='scored', patients=len(losses), components=len(groups), occurrences=len(frame),
                normalized_mae=dict(patient=losses.mean().to_dict(), component=groups.mean().to_dict()),
                comparisons=comparisons)


def run_fold(fold):
    variables, examples, bound, original, parent, extraction, old_dest = prepare(fold)
    frames, bindings = old_frames(fold, examples, variables, old_dest)
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), PROTOCOL, Path('scripts/organ_assimilation.py'),
             Path('whole_body_jepa/physiology_state.py'), old_dest/'contract.json']
    bindings.update({str(p): sha256(p) for p in paths})
    contract = dict(fold=fold, parent=parent, extraction=extraction, bindings=bindings,
                    seed=SEED, arms=list(CHANNELS), passes=2, lr=1e-4, weight_decay=.01)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed E29 contract')
    if not cp.exists():
        write_json(cp, contract)
    fits, diagnostics = {}, {}
    for channel in CHANNELS:
        path, mp = dest/f'{channel}.pt', dest/f'{channel}.json'
        if mp.exists():
            info = json.loads(mp.read_text())
            if (info['sha256'] != sha256(path) or info['contract_sha256'] != sha256(cp)
                    or info['channel'] != channel):
                raise ValueError('Changed E29 saved arm')
            candidate = AssimilatedReconstruction(original, channel).eval()
            candidate.load_state_dict(torch.load(path, weights_only=False), strict=True)
        else:
            space_required()
            candidate, log = fit_arm(original, examples['fit'], bound['fit'], channel)
            space_required()
            tmp = path.with_suffix('.tmp')
            torch.save(candidate.state_dict(), tmp); tmp.replace(path)
            info = dict(sha256=sha256(path), contract_sha256=sha256(cp), channel=channel,
                        training=log, parameters=sum(p.numel() for p in candidate.parameters()))
            write_json(mp, info)
        for key, value in original.trajectory.jepa.state_dict().items():
            torch.testing.assert_close(candidate.model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
        diagnostics[channel] = {}
        for role in examples:
            frames[role][channel], diagnostics[channel][role] = predict(candidate, examples[role], bound[role])
        fits[channel] = info
    if len({i['parameters'] for i in fits.values()}) != 1:
        raise ValueError('Unmatched learned parameter counts')
    hashes = {}
    for role, frame in frames.items():
        path = dest/f'{role}_predictions.parquet'
        tmp = path.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(path)
        hashes[role] = sha256(path)
    report_path = Path(f'artifacts/native_organ_assimilation_fold{fold}_20261004.json')
    write_json(report_path, dict(contract=contract, fits=fits, diagnostics=diagnostics,
        prediction_hashes=hashes, runtime_modified=False, promotion_allowed=False))
    print('E29 fold complete', fold, flush=True)
    return frames, variables, {str(report_path): sha256(report_path)}


def main():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    output = Path('artifacts/native_organ_assimilation_assessment_20261004.json')
    if output.exists():
        raise ValueError('Completed assessment exists; do not repeat')
    frames, fit_frames, bindings = [], [], {}
    for fold in range(5):
        data, variables, bound = run_fold(fold)
        frames.append(data['selection']); fit_frames.append(data['fit']); bindings.update(bound)
    data = pd.concat(frames, ignore_index=True)
    fit = pd.concat(fit_frames, ignore_index=True)
    validate_identities(data)
    validate_identities(fit)
    results = {}
    for i, variable in enumerate(variables):
        selected = data[data.variable == variable]
        results[variable] = assess(selected, SEED+i*1000) if len(selected) else dict(status='unsupported')
        fitted = fit[fit.variable == variable]
        if len(fitted):
            names = [n for n in NAMES if n != 'raw']
            losses = fitted[names].subtract(fitted.truth, axis=0).abs().groupby(fitted.patient).mean()
            results[variable]['fit_patient_mae'] = losses.mean().to_dict()
    scored = {v: r for v, r in results.items() if r['status'] == 'scored'}
    summary = dict(patients=int(data.patient.nunique()), occurrences=len(data), targets=len(scored),
        normalized_macro={level: {n: float(np.mean([r['normalized_mae'][level][n] for r in scored.values()]))
                                  for n in NAMES} for level in ('patient', 'component')},
        screens={c: [v for v, r in scored.items() if r['comparisons'][c]['screen_pass']] for c in CONTROLS})
    summary['mechanism_increment'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS[:3])))
    summary['full_screen'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS)))
    write_json(output, dict(summary=summary, results=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False, independent_verification_complete=False,
        limitation='Reused bounded development; assimilation policy not pure timing or forecast.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
