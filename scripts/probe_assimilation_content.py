"""Locate known-content accessibility at immediate assimilation versus later state."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.probe_observed_content import DEST as PARENT, ROOT, extract, last_observed
from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.bounded_site_sampling import bounded_weights
from scripts.encoder_fold_scaling import transform_example
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache

DEST = ROOT / 'assimilation_content_20261003'
PROTOCOL = Path('RECONSTRUCTION_ASSIMILATION_PROTOCOL_20261003.md')
ARMS = tuple(f'{r}_{s}' for r in ('ridge', 'hgb') for s in ('event', 'post', 'body'))
CONTRASTS = tuple((f'{r}_{a}', f'{r}_{b}') for r in ('ridge', 'hgb')
                  for a, b in (('event', 'post'), ('post', 'body')))


def trace_content(model, tokens, times, anchor, target):
    """Read snapshots only; the original state transition and update are unchanged."""
    n = len(model.trajectory.variables)
    last_observed(tokens, times, anchor, n)
    if (times[1:] <= times[:-1]).any():
        raise ValueError('Event times must strictly increase')
    clean, times = mask_native_target(tokens, times, model.trajectory.variables, target)
    present, positions = last_observed(clean, times, anchor, n)
    trajectory = model.trajectory
    body, organs, presence = encode_native_organ_observations(trajectory.jepa, trajectory.variables, clean)
    layer = trajectory.layer
    state = layer.initialize(clean.new_zeros((1, layer.body_dim)),
        clean.new_zeros((1, layer.organ_count, layer.organ_dim)), clean.new_zeros((1, layer.organ_count)))
    relative = (times-times[0]).to(clean.dtype)
    post = clean.new_zeros((n, layer.body_dim))
    for i in range(len(times)):
        delta = relative[i].reshape(1)-state.elapsed_hours
        state = layer.advance(state, delta)
        state = layer.update(state, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
        columns = np.flatnonzero(present & (positions == i))
        if len(columns):
            post[columns] = state.body[0]
    gap = (torch.as_tensor(anchor, device=times.device, dtype=times.dtype)-times[-1]).to(clean.dtype).reshape(1)
    final = layer.advance(state, gap).body[0]
    event = clean.new_zeros((n, layer.body_dim))
    unique = np.unique(positions[present])
    # Reproduce E19's batch shape; state replay above retains its original full batch.
    if len(unique):
        encoded, _, _ = encode_native_organ_observations(trajectory.jepa, trajectory.variables, clean[unique])
        event[np.flatnonzero(present)] = encoded[np.searchsorted(unique, positions[present])]
        batch_error = float((encoded-body[unique]).abs().max())
    else:
        batch_error = 0.
    return dict(post=post, final=final, event=event, present=present, positions=positions,
                event_batch_error=batch_error)


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.duplicated(['patient', 'fold']).any() or frame.variable.nunique()!=1
            or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid assimilation pairs')
    error = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    patients = error.groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = patients.groupby(links.reindex(patients.index)).mean()
    comparisons = {}
    for i, (a, b) in enumerate(CONTRASTS):
        stats = dict(patient=paired_summary(patients[a]-patients[b], seed+i, repetitions),
                     component=paired_summary(groups[a]-groups[b], seed+100+i, repetitions))
        passed = len(patients)>=30 and len(groups)>=3 and all(s['ci95'] is not None
            and s['ci95'][1]<0 and s['p_less'] is not None and s['p_less']<.05/(53*4) for s in stats.values())
        comparisons[a+'_minus_'+b] = dict(paired=stats, screen_pass=bool(passed))
    return dict(patients=len(patients), components=len(groups), query_occurrences=len(frame),
        folds=sorted(map(int, frame.fold.unique())),
        normalized_mae=dict(patient=patients.mean().to_dict(), component=groups.mean().to_dict()),
        comparisons=comparisons)


def run_fold(fold):
    variables, data, parent_contract, metadata = extract(fold)
    parent_report_path = Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
    parent_report = json.loads(parent_report_path.read_text())
    if parent_report['contract'] != parent_contract or parent_report['extraction'] != metadata:
        raise ValueError('E19 extraction changed')
    dest = DEST / f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    contract = dict(fold=fold, parent_sha256=sha256(parent_report_path), parent_contract=parent_contract,
        bindings={str(p): sha256(p) for p in (Path(__file__), PROTOCOL)}, scope='all_E19_supported')
    cp = dest / 'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed stage contract')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=2)+'\n')
    scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)
    schema = torch.load('models/whole_body_joint_jepa.pt', weights_only=False)
    sf = 0 if fold == 2 else fold
    checkpoint = torch.load(ROOT/f'monitored_continuation_20260929/fold{sf}_real_epoch10.pt', weights_only=False)
    model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    stages, extraction = {}, {}
    try:
        for role in ('fit', 'selection'):
            d = data[role]
            path, mp = dest/f'{role}.joblib', dest/f'{role}.json'
            if mp.exists():
                meta = json.loads(mp.read_text())
                if meta['contract_sha256'] != sha256(cp) or meta['sha256'] != sha256(path):
                    raise ValueError('Stage cache changed')
                post = joblib.load(path)
            else:
                rows = []
                max_event_error = max_final_error = max_event_batch_error = 0.
                with torch.inference_mode():
                    for i, (person, context) in enumerate(zip(d['patient'], d['context'])):
                        e = transform_example(source.get(person, int(context)), scaling['scalers'][fold])
                        traced = trace_content(model, e['tokens'], e['event_hours'], e['anchor_hour'], variables[int(context)])
                        np.testing.assert_array_equal(traced['present'], d['raw'][i, 106:159].astype(bool))
                        final, event = traced['final'].numpy(), traced['event'].numpy()
                        np.testing.assert_allclose(final, d['body'][i], atol=1e-6, rtol=1e-6)
                        np.testing.assert_allclose(event, d['event'][i], atol=1e-6, rtol=1e-6)
                        max_event_error = max(max_event_error, float(np.max(np.abs(event-d['event'][i]))))
                        max_event_batch_error = max(max_event_batch_error, traced['event_batch_error'])
                        max_final_error = max(max_final_error, float(np.max(np.abs(final-d['body'][i]))))
                        rows.append(traced['post'].numpy())
                        if (i+1)%512==0: print('Assimilation trace', fold, role, i+1, flush=True)
                post = np.stack(rows)
                joblib.dump(post, path)
                meta = dict(sha256=sha256(path), contract_sha256=sha256(cp),
                    contexts=len(post), max_event_error=max_event_error, max_final_error=max_final_error,
                    max_event_batch_error=max_event_batch_error)
                mp.write_text(json.dumps(meta, indent=2)+'\n')
            if post.shape != (len(d['patient']), 53, 64) or not np.isfinite(post).all():
                raise ValueError('Invalid stage features')
            stages[role] = post
            extraction[role] = meta
    finally:
        source.close()
    frames, results = [], {}
    with threadpool_limits(limits=4):
        for c, variable in enumerate(variables):
            old = parent_report['targets'][variable]
            if old['status'] != 'fitted':
                results[variable] = old
                continue
            oldp = PARENT/f'fold{fold}/{variable}.parquet'
            if sha256(oldp) != old['prediction_sha256']:
                raise ValueError('Changed E19 prediction')
            frame = pd.read_parquet(oldp)
            x = {}
            for role, d in data.items():
                m = d['raw'][:, 106+c].astype(bool)
                x[role] = np.c_[stages[role][m, c], d['raw'][m, 106:]]
            f, q = data['fit'], data['selection']
            fm, qm = f['raw'][:, 106+c].astype(bool), q['raw'][:, 106+c].astype(bool)
            if (frame.patient.tolist() != q['patient'][qm].tolist()
                    or not np.array_equal(frame.truth, q['raw'][qm, c])
                    or not np.array_equal(frame.context, q['context'][qm])):
                raise ValueError('Changed scored context')
            y = f['raw'][fm, c]
            paths = {ext: dest/f'{variable}.{ext}' for ext in ('joblib', 'parquet', 'json')}
            if paths['json'].exists():
                info = json.loads(paths['json'].read_text())
                if (info['contract_sha256'] != sha256(cp) or info['model_sha256'] != sha256(paths['joblib'])
                        or info['prediction_sha256'] != sha256(paths['parquet'])):
                    raise ValueError('Changed saved post reader')
                model_bundle = joblib.load(paths['joblib'])
                saved = pd.read_parquet(paths['parquet'])
                pd.testing.assert_frame_equal(saved[frame.columns], frame)
                frame = saved
            else:
                weights = bounded_weights(f['component'][fm].tolist())
                scaler = StandardScaler().fit(x['fit'], sample_weight=weights)
                ridge = Ridge(alpha=100., solver='cholesky').fit(scaler.transform(x['fit']), y, sample_weight=weights)
                hgb = HistGradientBoostingRegressor(**PARAMETERS).fit(x['fit'], y, sample_weight=weights)
                model_bundle = dict(scaler=scaler, ridge=ridge, hgb=hgb)
                frame['ridge_post'] = ridge.predict(scaler.transform(x['selection']))
                frame['hgb_post'] = hgb.predict(x['selection'])
                joblib.dump(model_bundle, paths['joblib'])
                frame.to_parquet(paths['parquet'], index=False)
                info = dict(status='fitted', support=old['support'], contract_sha256=sha256(cp),
                    model_sha256=sha256(paths['joblib']), prediction_sha256=sha256(paths['parquet']))
                paths['json'].write_text(json.dumps(info, indent=2)+'\n')
            info = dict(info, numerics=check_scaler(model_bundle['scaler']))
            if not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all():
                raise ValueError('Nonfinite predictions')
            np.testing.assert_allclose(frame.ridge_post, model_bundle['ridge'].predict(
                model_bundle['scaler'].transform(x['selection'])), atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(frame.hgb_post, model_bundle['hgb'].predict(x['selection']), atol=1e-12, rtol=1e-12)
            frames.append(frame)
            results[variable] = info
    report_path = Path(f'artifacts/native_reconstruction_assimilation_fold{fold}_20261003.json')
    report_path.write_text(json.dumps(dict(contract=contract, extraction=extraction, targets=results,
        promotion_allowed=False, runtime_modified=False), indent=2, allow_nan=False)+'\n')
    print('Assimilation fold complete', fold, len(frames), flush=True)
    return frames, report_path


def main():
    output = Path('artifacts/native_reconstruction_assimilation_assessment_20261003.json')
    if output.exists(): raise ValueError('Completed assimilation assessment exists')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    frames, bindings = [], {}
    for fold in range(5):
        rows, report = run_fold(fold)
        frames.extend(rows)
        bindings[str(report)] = sha256(report)
    data = pd.concat(frames, ignore_index=True)
    results = {v: assess(data[data.variable==v], 20261003+i*1000)
               for i, v in enumerate(sorted(data.variable.unique()))}
    summary = dict(variables=len(results), variable_folds=len(frames), query_occurrences=len(data),
        patients=int(data.patient.nunique()),
        normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
            for a in ARMS} for level in ('patient', 'component')},
        screens={a+'_minus_'+b: [v for v, r in results.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
                 for a, b in CONTRASTS})
    if len(frames)!=177 or len(data)!=88122 or len(results)!=36:
        raise ValueError('Original supported family not reproduced')
    output.write_text(json.dumps(dict(summary=summary, targets=results, bindings=bindings,
        promotion_allowed=False, runtime_modified=False,
        limitation='Observed-content developmental diagnostic, not missing-value inference or irreversible-loss proof.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
