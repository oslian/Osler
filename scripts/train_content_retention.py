"""Fixed E21 state-objective intervention; private rows/weights, no promotion."""
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from scripts.probe_observed_content import ROOT, extract, last_observed
from scripts.encoder_fold_scaling import transform_example
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.score_additional_reconstruction import paired_summary
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache

DEST = ROOT / 'content_training_20261003'
PROTOCOL = Path('RECONSTRUCTION_CONTENT_TRAINING_PROTOCOL_20261003.md')
ARMS = ('primary', 'real', 'shuffled')
SEED = 20261003


def replay(layer, example):
    body, organs, presence, times = [example[k] for k in ('body', 'organs', 'presence', 'times')]
    if (not len(times) or times.shape != (len(body),) or not torch.isfinite(times).all()
            or not np.isfinite(example['anchor']) or (times > example['anchor']).any()
            or (times[1:] <= times[:-1]).any()):
        raise ValueError('Invalid cached event times')
    state = layer.initialize(body.new_zeros((1, layer.body_dim)),
        body.new_zeros((1, layer.organ_count, layer.organ_dim)), body.new_zeros((1, layer.organ_count)))
    relative = (times-times[0]).to(body.dtype)
    for i in range(len(times)):
        delta = relative[i:i+1]-state.elapsed_hours
        state = layer.advance(state, delta)
        state = layer.update(state, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
    gap = (times.new_tensor(example['anchor'])-times[-1]).to(body.dtype).reshape(1)
    return layer.advance(state, gap).body


def shuffle_observed(values, masks):
    if (values.ndim != 2 or masks.shape != values.shape or masks.dtype != torch.bool
            or not torch.isfinite(values[masks]).all()):
        raise ValueError('Invalid observed labels')
    eligible = masks.sum(0) >= 2
    shuffled = values.clone()
    for c in torch.where(eligible)[0]:
        rows = torch.where(masks[:, c])[0]
        shuffled[rows, c] = values[rows.roll(1), c]
    return shuffled, masks & eligible[None], eligible


def content_loss(head, state, values, mask, detached):
    if not mask.any():
        return head.weight.sum()*0
    prediction = head(state.detach() if detached else state)[0]
    return F.smooth_l1_loss(prediction[mask], values[mask])


def build_fold(fold):
    variables, data, parent, metadata = extract(fold)
    dest = DEST / f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    contract = dict(fold=fold, parent=parent, extraction=metadata, torch=torch.__version__,
        bindings={str(p): sha256(p) for p in (Path(__file__), PROTOCOL)},
        seed=SEED, passes=2, warmup_passes=1, auxiliary_weight=.25,
        learning_rate=1e-4, warmup_learning_rate=1e-3, weight_decay=.01)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed E21 contract')
    if not cp.exists(): cp.write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
    scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)
    schema = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
    sf = 0 if fold == 2 else fold
    ck = torch.load(ROOT/f'monitored_continuation_20260929/fold{sf}_real_epoch10.pt', weights_only=False)
    original = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    original.load_state_dict(ck['state_dict'], strict=True)
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    examples, extraction = {}, {}
    try:
        for role, d in data.items():
            path, mp = dest/f'{role}.joblib', dest/f'{role}.json'
            if mp.exists():
                m = json.loads(mp.read_text())
                if m['sha256'] != sha256(path) or m['contract_sha256'] != sha256(cp):
                    raise ValueError('Changed event cache')
                rows = joblib.load(path)
            else:
                rows, errors = [], []
                for i, (patient, c) in enumerate(zip(d['patient'], d['context'])):
                    c = int(c)
                    e = transform_example(source.get(patient, c), scaling['scalers'][fold])
                    # Validate the original history before target-only events are removed.
                    last_observed(e['tokens'], e['event_hours'], e['anchor_hour'], len(variables))
                    if (e['event_hours'][1:] <= e['event_hours'][:-1]).any():
                        raise ValueError('Unordered original history')
                    clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, variables[c])
                    present, positions = last_observed(clean, times, e['anchor_hour'], len(variables))
                    if present[c] or not np.array_equal(present, d['raw'][i, 106:159].astype(bool)):
                        raise ValueError('Changed target mask')
                    columns = np.flatnonzero(present)
                    np.testing.assert_array_equal(clean[positions[present], columns].numpy(), d['raw'][i, columns])
                    with torch.no_grad():
                        body, organs, presence = encode_native_organ_observations(original.trajectory.jepa, variables, clean)
                        row = dict(body=body.detach().clone(), organs=organs.detach().clone(),
                            presence=presence.detach().clone(), times=times.clone(), anchor=float(e['anchor_hour']),
                            target=c, truth=float(e['truth_normalized']),
                            values=torch.tensor(d['raw'][i, :53]), mask=torch.tensor(present),
                            patient=patient, component=d['component'][i])
                        state = replay(original.trajectory.layer, row)
                        np.testing.assert_allclose(state[0].numpy(), d['body'][i], atol=1e-6, rtol=0)
                        prediction = original.readout(state)[0, c]
                        direct = original(e['tokens'], e['event_hours'], e['anchor_hour'], variables[c])[0]
                        torch.testing.assert_close(prediction, direct, atol=1e-7, rtol=0)
                        row['frozen'] = float(prediction)
                        errors.append(float(np.max(np.abs(state[0].numpy()-d['body'][i]))))
                    rows.append(row)
                joblib.dump(rows, path)
                m = dict(sha256=sha256(path), contract_sha256=sha256(cp), contexts=len(rows),
                         original_state_max_error=max(errors), native_predictions_reproduced=len(rows))
                mp.write_text(json.dumps(m, indent=2, allow_nan=False)+'\n')
            if [(r['patient'], r['target']) for r in rows] != list(zip(d['patient'], map(int, d['context']))):
                raise ValueError('Cached identity mismatch')
            examples[role], extraction[role] = rows, m
            print('E21 cached', fold, role, len(rows), flush=True)
    finally:
        source.close()
    return variables, examples, original, contract, extraction, dest


def fit_arm(original, examples, arm):
    if arm not in ARMS: raise ValueError('Unknown arm')
    torch.manual_seed(SEED)
    model = deepcopy(original).eval()
    for p in model.parameters(): p.requires_grad_(False)
    head = nn.Linear(model.trajectory.layer.body_dim, len(model.trajectory.variables))
    values = torch.stack([e['values'] for e in examples]).to(head.weight.dtype)
    masks = torch.stack([e['mask'] for e in examples])
    shuffled, masks, eligible = shuffle_observed(values, masks)
    labels = shuffled if arm == 'shuffled' else values
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    for i in rng.permutation(len(examples)):
        with torch.no_grad(): state = replay(model.trajectory.layer, examples[i])
        loss = content_loss(head, state, labels[i], masks[i], True)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True); optimizer.step()
    params = list(model.trajectory.layer.parameters())+list(model.readout.parameters())+list(head.parameters())
    for p in params: p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    log = []
    for epoch in range(2):
        losses = []
        for i in rng.permutation(len(examples)):
            e = examples[i]
            state = replay(model.trajectory.layer, e)
            prediction = model.readout(state)[0, e['target']]
            primary = F.smooth_l1_loss(prediction, prediction.new_tensor(e['truth']))
            auxiliary = content_loss(head, state, labels[i], masks[i], arm == 'primary')
            loss = primary+.25*auxiliary
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
            losses.append([float(primary.detach()), float(auxiliary.detach())])
        log.append(dict(epoch=epoch+1, primary=float(np.mean(losses, 0)[0]),
                        auxiliary=float(np.mean(losses, 0)[1]), steps=len(losses)))
        print('E21 trained', arm, log[-1], flush=True)
    for key, value in original.trajectory.jepa.state_dict().items():
        if not torch.equal(value, model.trajectory.jepa.state_dict()[key]):
            raise ValueError('Frozen encoder changed')
    return model, head.eval(), eligible, log


def predict_rows(model, head, examples, eligible, fold, variables):
    rows = []
    with torch.no_grad():
        for e in examples:
            state = replay(model.trajectory.layer, e)
            point, content = model.readout(state)[0], head(state)[0]
            base = dict(patient=e['patient'], component=e['component'], fold=fold, context=e['target'])
            rows.append(dict(base, variable=variables[e['target']], task='masked', truth=e['truth'],
                             frozen=e['frozen'], prediction=float(point[e['target']])))
            for c in torch.where(e['mask'] & eligible)[0].tolist():
                rows.append(dict(base, variable=variables[c], task='content', truth=float(e['values'][c]),
                                 prediction=float(content[c])))
    return pd.DataFrame(rows)


def assess(frame, seed, repetitions=5000):
    if frame.empty or frame.task.nunique() != 1 or frame.variable.nunique() != 1:
        raise ValueError('One nonempty task and variable required')
    arms = (*ARMS, 'frozen') if frame.task.iloc[0] == 'masked' else ARMS
    if (frame.duplicated(['patient', 'fold']).any() or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *arms]].to_numpy()).all()):
        raise ValueError('Invalid scored pairs')
    patients = frame[list(arms)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = patients.groupby(links.reindex(patients.index)).mean()
    pairs = [('real', 'primary'), ('real', 'shuffled')]
    if 'frozen' in arms: pairs += [(a, 'frozen') for a in ARMS]
    comparisons = {}
    for i, (a, b) in enumerate(pairs):
        stats = dict(patient=paired_summary(patients[a]-patients[b], seed+i, repetitions),
                     component=paired_summary(groups[a]-groups[b], seed+100+i, repetitions))
        passed = b != 'frozen' and len(patients) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*2*2) for s in stats.values())
        comparisons[a+'_minus_'+b] = dict(paired=stats, screen_pass=bool(passed))
    return dict(status='scored', patients=len(patients), components=len(groups),
        query_occurrences=len(frame), folds=sorted(map(int, frame.fold.unique())),
        normalized_mae=dict(patient=patients.mean().to_dict(), component=groups.mean().to_dict()),
        comparisons=comparisons)


def run_fold(fold):
    variables, examples, original, contract, extraction, dest = build_fold(fold)
    frame, fits, support = None, {}, None
    cp = dest/'contract.json'
    for arm in ARMS:
        path, meta = dest/f'{arm}.pt', dest/f'{arm}.json'
        if meta.exists():
            info = json.loads(meta.read_text())
            if info['sha256'] != sha256(path) or info['contract_sha256'] != sha256(cp):
                raise ValueError('Changed fitted arm')
            saved = torch.load(path, weights_only=False)
            model = deepcopy(original).eval(); model.load_state_dict(saved['model'], strict=True)
            head = nn.Linear(model.trajectory.layer.body_dim, len(variables)).eval()
            head.load_state_dict(saved['head'], strict=True); eligible = saved['eligible']
        else:
            model, head, eligible, log = fit_arm(original, examples['fit'], arm)
            torch.save(dict(model=model.state_dict(), head=head.state_dict(), eligible=eligible), path)
            info = dict(sha256=sha256(path), contract_sha256=sha256(cp), training=log,
                        encoder_unchanged=True)
            meta.write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')
        if support is not None and not torch.equal(support, eligible): raise ValueError('Unequal arm support')
        support = eligible
        pred = predict_rows(model, head, examples['selection'], eligible, fold, variables)
        if frame is None: frame = pred.drop(columns='prediction')
        pd.testing.assert_frame_equal(frame[pred.columns.drop('prediction')], pred.drop(columns='prediction'))
        frame[arm] = pred.prediction
        fits[arm] = info
    private = dest/'predictions.parquet'; frame.to_parquet(private, index=False)
    report = Path(f'artifacts/native_reconstruction_content_training_fold{fold}_20261003.json')
    report.write_text(json.dumps(dict(contract=contract, extraction=extraction, fits=fits,
        prediction_sha256=sha256(private), eligible_variables=[v for v, ok in zip(variables, support) if ok],
        query_occurrences=len(frame), runtime_modified=False, promotion_allowed=False), indent=2, allow_nan=False)+'\n')
    return frame, variables, {str(report): sha256(report)}


def main():
    out = Path('artifacts/native_reconstruction_content_training_assessment_20261003.json')
    if out.exists(): raise ValueError('Completed E21 assessment already exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    frames, bindings = [], {}
    for fold in range(5):
        frame, variables, bound = run_fold(fold); frames.append(frame); bindings.update(bound)
    data = pd.concat(frames, ignore_index=True)
    results, summary = {}, {}
    for task in ('masked', 'content'):
        results[task] = {}
        for i, variable in enumerate(variables):
            selected = data[(data.task == task) & (data.variable == variable)]
            results[task][variable] = assess(selected, SEED+i*1000) if len(selected) else dict(status='unsupported')
        supported = {v: r for v, r in results[task].items() if r['status'] == 'scored'}
        arms = (*ARMS, 'frozen') if task == 'masked' else ARMS
        summary[task] = dict(supported=len(supported),
            normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in supported.values()]))
                for a in arms} for level in ('patient', 'component')},
            screens={key: [v for v, r in supported.items() if r['comparisons'][key]['screen_pass']]
                for key in ('real_minus_primary', 'real_minus_shuffled')})
    out.write_text(json.dumps(dict(summary=summary, results=results, bindings=bindings,
        patients=int(data.patient.nunique()), query_occurrences=len(data), runtime_modified=False,
        promotion_allowed=False, limitation='Bounded reused development; known-content retention plus masked inference, not forecast.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
