"""E27 paired state adaptation; never modifies serving weights or old artifacts."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_organ_assignment import pools
from scripts.encoder_fold_scaling import transform_example
from scripts.probe_observed_content import DEST as OBSERVED
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_content_retention import ROOT, build_fold, replay
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.reconstruction_cache import ReconstructionCache

DEST = ROOT / 'organ_binding_20261004'
PROTOCOL = Path('RECONSTRUCTION_ORGAN_BINDING_PROTOCOL_20261004.md')
ARMS = ('native', 'binding', 'postpool')
CONTROLS = ('native', 'postpool', 'frozen', 'raw')
SEED = 20261004


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def space_required():
    if shutil.disk_usage('.').free < 10 * 1024**3:
        raise ValueError('Less than 10 GiB free; preserve data and stop')


def select_arm(example, bound, arm):
    if arm not in ARMS:
        raise ValueError('Unknown binding arm')
    organs = example['organs']
    if bound.shape != organs.shape or not torch.isfinite(bound).all():
        raise ValueError('Invalid bound proposals')
    if arm == 'binding':
        organs = bound
    elif arm == 'postpool':
        organs = F.silu(organs)
    return dict(example, organs=organs)


def validate_roles(examples, roles):
    sets = {}
    for role, rows in examples.items():
        people = [row['patient'] for row in rows]
        if not people or len(set(people)) != len(people) or any(roles.get(p) != role for p in people):
            raise ValueError('Invalid patient roles or duplicate contexts')
        sets[role] = set(people)
    if sets['fit'] & sets['selection']:
        raise ValueError('Fitting/selection overlap')


def prepare(fold):
    variables, examples, original, parent, extraction, old_dest = build_fold(fold)
    roles = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds'][fold]
    validate_roles(examples, roles)
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), PROTOCOL, Path('scripts/audit_organ_assignment.py'),
             Path('scripts/train_content_retention.py'), Path('scripts/probe_observed_content.py'),
             Path('scripts/score_additional_reconstruction.py'),
             Path('scripts/score_monitored_continuation_outer.py'), old_dest/'contract.json']
    for role in examples:
        paths.extend((old_dest/f'{role}.joblib', old_dest/f'{role}.json'))
    contract = dict(fold=fold, variables=variables, parent=parent, extraction=extraction,
                    arms=ARMS, seed=SEED, passes=2, lr=1e-4, weight_decay=.01,
                    bindings={str(p): sha256(p) for p in paths})
    # JSON-normalize tuple-valued configuration before resume comparisons.
    contract = json.loads(json.dumps(contract))
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed binding experiment contract')
    if not cp.exists():
        write_json(cp, contract)
    scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)['scalers'][fold]
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    bound, metadata = {}, {}
    try:
        for role, rows in examples.items():
            path, mp = dest/f'{role}_binding.joblib', dest/f'{role}_binding.json'
            if mp.exists():
                info = json.loads(mp.read_text())
                if info['sha256'] != sha256(path) or info['contract_sha256'] != sha256(cp):
                    raise ValueError('Changed binding cache')
                saved = joblib.load(path)
            else:
                space_required()
                saved = []
                parity = 0.
                for i, row in enumerate(rows):
                    e = transform_example(source.get(row['patient'], row['target']), scaling)
                    if (e['truth_normalized'], e['anchor_hour']) != (row['truth'], row['anchor']):
                        raise ValueError('Changed truth/anchor')
                    clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, variables[row['target']])
                    torch.testing.assert_close(times, row['times'], atol=0, rtol=0)
                    with torch.no_grad():
                        manual, candidate = pools(original.trajectory.jepa, variables, clean)
                    torch.testing.assert_close(manual, row['organs'], atol=1e-6, rtol=0)
                    if not torch.isfinite(candidate).all():
                        raise ValueError('Nonfinite binding proposals')
                    parity = max(parity, float((manual-row['organs']).abs().max()))
                    saved.append(dict(patient=row['patient'], target=row['target'], organs=candidate.detach()))
                    if (i+1) % 512 == 0:
                        print('E27 proposals', fold, role, i+1, len(rows), flush=True)
                space_required()
                tmp = path.with_suffix('.tmp')
                joblib.dump(saved, tmp); tmp.replace(path)
                info = dict(sha256=sha256(path), contract_sha256=sha256(cp), contexts=len(saved),
                            original_proposal_max_error=parity)
                write_json(mp, info)
            if [(x['patient'], x['target']) for x in saved] != [(x['patient'], x['target']) for x in rows]:
                raise ValueError('Binding cache identity mismatch')
            bound[role] = [x['organs'] for x in saved]
            metadata[role] = info
    finally:
        source.close()
    return variables, examples, bound, original, contract, metadata, dest


def fit_arm(original, examples, bound, arm):
    if len(examples) != len(bound) or not examples or arm not in ARMS:
        raise ValueError('Aligned nonempty fitting data and known arm required')
    torch.manual_seed(SEED)
    model = deepcopy(original).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.trajectory.layer.parameters()) + list(model.readout.parameters())
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    rng = np.random.default_rng(SEED)
    logs = []
    for epoch in range(2):
        losses = []
        for index in rng.permutation(len(examples)):
            e = select_arm(examples[index], bound[index], arm)
            state = replay(model.trajectory.layer, e)
            prediction = model.readout(state)[0, e['target']]
            loss = F.smooth_l1_loss(prediction, prediction.new_tensor(e['truth']))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite primary loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append(dict(epoch=epoch+1, steps=len(losses), loss=float(np.mean(losses))))
        print('E27 trained', arm, logs[-1], flush=True)
    for key, value in original.trajectory.jepa.state_dict().items():
        torch.testing.assert_close(model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
    return model, logs


def predict(model, examples, bound, arm):
    if len(examples) != len(bound):
        raise ValueError('Prediction context alignment mismatch')
    values = []
    with torch.no_grad():
        for e, b in zip(examples, bound):
            state = replay(model.trajectory.layer, select_arm(e, b, arm))
            values.append(float(model.readout(state)[0, e['target']]))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    return np.asarray(values)


def raw_control(fold, examples, variables):
    """Reuse strong full-fit raw readers; check direct predictions against saved rows."""
    dest = ROOT/f'nonlinear_probe_20261001/fold{fold}_probes'
    contract_path = dest/'contract.json'
    parent = json.loads(contract_path.read_text())
    if parent['fold'] != fold or parent['parameters'] != PARAMETERS:
        raise ValueError('Wrong full-fit raw contract')
    verify_bound_files(parent['bindings'], {})
    data_path = OBSERVED/f'fold{fold}/selection.joblib'
    d = joblib.load(data_path)
    if list(zip(d['patient'], map(int, d['context']))) != [(e['patient'], e['target']) for e in examples]:
        raise ValueError('Raw context mismatch')
    values = np.full(len(examples), np.nan)
    bindings = {str(contract_path): sha256(contract_path), str(data_path): sha256(data_path)}
    for col in sorted(set(e['target'] for e in examples)):
        variable = variables[col]
        model_path, rows_path, meta_path = [dest/f'{variable}.{ext}' for ext in ('joblib', 'parquet', 'json')]
        info = json.loads(meta_path.read_text())
        if ((info['fold'], info['target'], info['contract_sha256']) != (fold, variable, sha256(contract_path))
                or sha256(model_path) != info['model_sha256'] or sha256(rows_path) != info['prediction_sha256']):
            raise ValueError('Changed raw artifact')
        bundle = joblib.load(model_path)
        if bundle['target'] != variable or bundle['parameters'] != PARAMETERS:
            raise ValueError('Wrong raw model')
        ids = np.array([i for i, e in enumerate(examples) if e['target'] == col])
        actual = bundle['models']['raw_probe'].predict(d['raw'][ids])
        saved = pd.read_parquet(rows_path).set_index('patient').loc[[examples[i]['patient'] for i in ids]]
        np.testing.assert_allclose(saved.truth, [examples[i]['truth'] for i in ids], atol=0, rtol=0)
        np.testing.assert_allclose(saved.native, [examples[i]['frozen'] for i in ids], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(saved.component, [examples[i]['component'] for i in ids])
        np.testing.assert_allclose(saved.raw_probe, actual, atol=1e-12, rtol=1e-12)
        values[ids] = actual
        bindings.update({str(p): sha256(p) for p in (model_path, rows_path, meta_path)})
    if not np.isfinite(values).all():
        raise ValueError('Incomplete raw control')
    return values, bindings


def assess(frame, seed, repetitions=5000):
    names = (*ARMS, 'frozen', 'raw')
    if (frame.empty or frame.variable.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *names]].to_numpy()).all()):
        raise ValueError('Invalid paired assessment')
    losses = frame[list(names)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = losses.groupby(links.reindex(losses.index)).mean()
    comparisons = {}
    for i, control in enumerate(CONTROLS):
        stats = dict(patient=paired_summary(losses.binding-losses[control], seed+i, repetitions),
                     component=paired_summary(groups.binding-groups[control], seed+100+i, repetitions))
        passed = len(losses) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(53*4) for s in stats.values())
        comparisons[control] = dict(paired=stats, screen_pass=bool(passed))
    return dict(status='scored', patients=len(losses), components=len(groups), occurrences=len(frame),
                normalized_mae=dict(patient=losses.mean().to_dict(), component=groups.mean().to_dict()),
                comparisons=comparisons)


def run_fold(fold):
    variables, examples, bound, original, contract, extraction, dest = prepare(fold)
    cp = dest/'contract.json'
    frames = {role: pd.DataFrame([dict(patient=e['patient'], component=e['component'],
        context=e['target'], variable=variables[e['target']], fold=fold, truth=e['truth'],
        frozen=e['frozen']) for e in rows]) for role, rows in examples.items()}
    fits = {}
    for arm in ARMS:
        path, mp = dest/f'{arm}.pt', dest/f'{arm}.json'
        if mp.exists():
            info = json.loads(mp.read_text())
            if info['sha256'] != sha256(path) or info['contract_sha256'] != sha256(cp):
                raise ValueError('Changed trained arm')
            model = deepcopy(original).eval()
            model.load_state_dict(torch.load(path, weights_only=False), strict=True)
        else:
            model, log = fit_arm(original, examples['fit'], bound['fit'], arm)
            space_required()
            tmp = path.with_suffix('.tmp'); torch.save(model.state_dict(), tmp); tmp.replace(path)
            info = dict(sha256=sha256(path), contract_sha256=sha256(cp), training=log,
                        encoder_unchanged=True, parameters=sum(p.numel() for p in model.parameters()))
            write_json(mp, info)
        for role in examples:
            frames[role][arm] = predict(model, examples[role], bound[role], arm)
        fits[arm] = info
    frames['selection']['raw'], bindings = raw_control(fold, examples['selection'], variables)
    prediction_hashes = {}
    for role, frame in frames.items():
        path = dest/f'{role}_predictions.parquet'
        tmp = path.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(path)
        prediction_hashes[role] = sha256(path)
    output = Path(f'artifacts/native_organ_binding_fold{fold}_20261004.json')
    write_json(output, dict(contract=contract, extraction=extraction, fits=fits, raw_bindings=bindings,
        prediction_hashes=prediction_hashes, runtime_modified=False, promotion_allowed=False))
    print('E27 fold complete', fold, flush=True)
    return frames['selection'], variables, {str(output): sha256(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    output = Path('artifacts/native_organ_binding_assessment_20261004.json')
    if output.exists():
        raise ValueError('Completed assessment exists; do not repeat')
    frames, bindings = [], {}
    for fold in range(5):
        if args.cache_only:
            prepare(fold)
        else:
            frame, variables, bound = run_fold(fold)
            frames.append(frame); bindings.update(bound)
    if args.cache_only:
        print('E27 all five paired caches prepared; training not executed', flush=True)
        return
    data = pd.concat(frames, ignore_index=True)
    results = {}
    for i, variable in enumerate(variables):
        selected = data[data.variable == variable]
        results[variable] = assess(selected, SEED+i*1000) if len(selected) else dict(status='unsupported')
    scored = {v: r for v, r in results.items() if r['status'] == 'scored'}
    summary = dict(patients=int(data.patient.nunique()), occurrences=len(data), targets=len(scored),
        normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in scored.values()]))
                                 for a in (*ARMS, 'frozen', 'raw')} for level in ('patient', 'component')},
        screens={c: [v for v, r in scored.items() if r['comparisons'][c]['screen_pass']] for c in CONTROLS})
    summary['mechanism_increment'] = sorted(set(summary['screens']['native']) & set(summary['screens']['postpool']))
    summary['full_screen'] = sorted(set.intersection(*(set(summary['screens'][c]) for c in CONTROLS)))
    write_json(output, dict(summary=summary, results=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Bounded reused development and frozen event encoder; not forecast or promotion.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
