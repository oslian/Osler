"""Separate known-content readability under later dynamics and assimilation."""
from dataclasses import fields, replace
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

from scripts.audit_scaled_kernel_numerics import check_scaler
from scripts.bounded_site_sampling import bounded_weights
from scripts.encoder_fold_scaling import transform_example
from scripts.probe_assimilation_content import DEST as ASSIMILATION
from scripts.probe_observed_content import DEST as OBSERVED, ROOT, last_observed
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_content_retention import build_fold
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.reconstruction_cache import ReconstructionCache

DEST = ROOT / 'state_suffix_20261003'
PROTOCOL = Path('RECONSTRUCTION_SUFFIX_PROTOCOL_20261003.md')
ARMS = tuple(f'{r}_{s}' for r in ('ridge', 'hgb') for s in ('post', 'advance', 'update', 'body'))
CONTRASTS = tuple((f'{r}_{a}', f'{r}_{b}') for r in ('ridge', 'hgb') for a, b in
                 (('post', 'advance'), ('post', 'update'), ('advance', 'body'), ('update', 'body')))


def append_state(batch, one):
    if batch is None:
        return one
    return replace(one, **{f.name: torch.cat((getattr(batch, f.name), getattr(one, f.name)), 0)
                           for f in fields(one)})


def suffix_states(layer, example, positions):
    """Branch only after the last observation; preserve original time segmentation."""
    body, organs, presence, times = [example[k] for k in ('body', 'organs', 'presence', 'times')]
    positions = np.asarray(positions)
    if (not len(times) or times.shape != (len(body),) or not torch.isfinite(times).all()
            or not np.isfinite(example['anchor']) or (times > example['anchor']).any()
            or (times[1:] <= times[:-1]).any() or positions.ndim != 1
            or positions.dtype.kind not in 'iu' or (positions < -1).any() or (positions >= len(times)).any()):
        raise ValueError('Invalid suffix history or observation positions')
    state = layer.initialize(body.new_zeros((1, layer.body_dim)),
        body.new_zeros((1, layer.organ_count, layer.organ_dim)), body.new_zeros((1, layer.organ_count)))
    relative = (times-times[0]).to(body.dtype)
    post = body.new_zeros((len(positions), layer.body_dim))
    advance = update = None
    order = []
    for i in range(len(times)):
        delta = relative[i:i+1]-state.elapsed_hours
        state = layer.advance(state, delta)
        state = layer.update(state, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
        if advance is not None:
            count = len(order)
            advance = layer.advance(advance, delta.expand(count))
            # Clock progression is not latent evolution; the gate still gets real delta.
            update = replace(update, elapsed_hours=state.elapsed_hours.expand(count))
            update = layer.update(update, body[i:i+1].expand(count, -1),
                organs[i:i+1].expand(count, -1, -1), presence[i:i+1].expand(count, -1), delta.expand(count))
        columns = np.flatnonzero(positions == i)
        if len(columns):
            post[columns] = state.body[0]
            advance = append_state(advance, state)
            update = append_state(update, state)
            order.append(i)
    gap = (times.new_tensor(example['anchor'])-times[-1]).to(body.dtype).reshape(1)
    final = layer.advance(state, gap)
    result = dict(post=post, body=final.body[0])
    a, u = torch.zeros_like(post), torch.zeros_like(post)
    present = positions >= 0
    if advance is not None:
        advance = layer.advance(advance, gap.expand(len(order)))
        update = replace(update, elapsed_hours=final.elapsed_hours.expand(len(order)))
        indices = np.searchsorted(order, positions[present])
        a[present], u[present] = advance.body[indices], update.body[indices]
    result.update(advance=a, update=u)
    return result


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.duplicated(['patient', 'fold']).any() or frame.variable.nunique() != 1
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid suffix pairs')
    patient = frame[list(ARMS)].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    component = patient.groupby(links.reindex(patient.index)).mean()
    comparisons = {}
    for i, (a, b) in enumerate(CONTRASTS):
        stats = dict(patient=paired_summary(patient[a]-patient[b], seed+i, repetitions),
                     component=paired_summary(component[a]-component[b], seed+100+i, repetitions))
        passed = len(patient) >= 30 and len(component) >= 3 and all(s['ci95'] is not None
            and s['ci95'][1] < 0 and s['p_less'] is not None and s['p_less'] < .05/(53*8) for s in stats.values())
        comparisons[a+'_minus_'+b] = dict(paired=stats, screen_pass=bool(passed))
    means = dict(patient=patient.mean().to_dict(), component=component.mean().to_dict())
    return dict(patients=len(patient), components=len(component), query_occurrences=len(frame),
        folds=sorted(map(int, frame.fold.unique())), normalized_mae=means, comparisons=comparisons,
        descriptive_interaction={level: {r: m[r+'_body']-m[r+'_advance']-m[r+'_update']+m[r+'_post']
            for r in ('ridge', 'hgb')} for level, m in means.items()})


def run_fold(fold):
    variables, examples, original, parent, encoded_meta, encoded_dir = build_fold(fold)
    pp = Path(f'artifacts/native_reconstruction_assimilation_fold{fold}_20261003.json')
    prior = json.loads(pp.read_text())
    observed_report = json.loads(Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json').read_text())
    if prior['contract']['parent_contract'] != parent['parent']:
        raise ValueError('Different parent roles or model')
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    contract = dict(fold=fold, parent_sha256=sha256(pp), encoded_contract=parent,
        bindings={str(p): sha256(p) for p in (Path(__file__), PROTOCOL)}, parameters=PARAMETERS)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed suffix contract')
    if not cp.exists(): cp.write_text(json.dumps(contract, indent=2)+'\n')
    scaling = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)
    source = ReconstructionCache(ROOT/'full_histories_20260922.sqlite')
    data, stages, extraction = {}, {}, {}
    try:
        for role, rows in examples.items():
            oldpath, postpath = OBSERVED/f'fold{fold}/{role}.joblib', ASSIMILATION/f'fold{fold}/{role}.joblib'
            if (sha256(oldpath) != observed_report['extraction'][role]['sha256']
                    or sha256(postpath) != prior['extraction'][role]['sha256']):
                raise ValueError('Changed original stage cache')
            d, oldpost = joblib.load(oldpath), joblib.load(postpath)
            data[role] = d
            path, mp = dest/f'{role}.joblib', dest/f'{role}.json'
            if mp.exists():
                meta = json.loads(mp.read_text())
                if meta['contract_sha256'] != sha256(cp) or meta['sha256'] != sha256(path):
                    raise ValueError('Changed suffix cache')
                stage = joblib.load(path)
            else:
                results, post_errors, body_errors = [], [], []
                with torch.no_grad():
                    for i, row in enumerate(rows):
                        e = transform_example(source.get(row['patient'], row['target']), scaling['scalers'][fold])
                        clean, times = mask_native_target(e['tokens'], e['event_hours'], variables, variables[row['target']])
                        present, positions = last_observed(clean, times, e['anchor_hour'], len(variables))
                        torch.testing.assert_close(times, row['times'], atol=0, rtol=0)
                        np.testing.assert_array_equal(present, row['mask'].numpy())
                        np.testing.assert_array_equal(clean[positions[present], np.flatnonzero(present)], row['values'][present])
                        if row['anchor'] != e['anchor_hour'] or (row['patient'], row['target']) != (d['patient'][i], d['context'][i]):
                            raise ValueError('Changed context')
                        r = suffix_states(original.trajectory.layer, row, positions)
                        np.testing.assert_allclose(r['post'].numpy(), oldpost[i], atol=1e-6, rtol=0)
                        np.testing.assert_allclose(r['body'].numpy(), d['body'][i], atol=1e-6, rtol=0)
                        post_errors.append(float(np.max(np.abs(r['post'].numpy()-oldpost[i]))))
                        body_errors.append(float(np.max(np.abs(r['body'].numpy()-d['body'][i]))))
                        results.append({k: r[k].numpy() for k in ('advance', 'update')})
                        if (i+1)%512 == 0: print('Suffix', fold, role, i+1, flush=True)
                stage = {k: np.stack([r[k] for r in results]) for k in ('advance', 'update')}
                joblib.dump(stage, path)
                meta = dict(sha256=sha256(path), contract_sha256=sha256(cp),
                    encoded_sha256=encoded_meta[role]['sha256'], contexts=len(rows),
                    post_max_error=max(post_errors), final_max_error=max(body_errors))
                mp.write_text(json.dumps(meta, indent=2)+'\n')
            for values in stage.values():
                if values.shape != oldpost.shape or not np.isfinite(values).all():
                    raise ValueError('Invalid suffix features')
            stages[role], extraction[role] = stage, meta
    finally:
        source.close()
    frames, results = [], {}
    with threadpool_limits(limits=4):
        for c, variable in enumerate(variables):
            old = prior['targets'][variable]
            if old['status'] != 'fitted':
                results[variable] = old
                continue
            oldp = ASSIMILATION/f'fold{fold}/{variable}.parquet'
            if sha256(oldp) != old['prediction_sha256']: raise ValueError('Changed original predictions')
            frame = pd.read_parquet(oldp)
            f, q = data['fit'], data['selection']
            fm, qm = f['raw'][:, 106+c].astype(bool), q['raw'][:, 106+c].astype(bool)
            if frame.patient.tolist() != q['patient'][qm].tolist() or not np.array_equal(frame.truth, q['raw'][qm, c]):
                raise ValueError('Changed scoring rows')
            paths = {s: dest/f'{variable}.{s}' for s in ('joblib', 'parquet', 'json')}
            if paths['json'].exists():
                info = json.loads(paths['json'].read_text())
                if (info['contract_sha256'] != sha256(cp) or info['model_sha256'] != sha256(paths['joblib'])
                        or info['prediction_sha256'] != sha256(paths['parquet'])):
                    raise ValueError('Changed saved suffix reader')
                saved = pd.read_parquet(paths['parquet'])
                pd.testing.assert_frame_equal(saved[frame.columns], frame)
                frame = saved
            else:
                bundle, numerics = {}, {}
                weights = bounded_weights(f['component'][fm].tolist())
                for name in ('advance', 'update'):
                    x = np.c_[stages['fit'][name][fm, c], f['raw'][fm, 106:]]
                    z = np.c_[stages['selection'][name][qm, c], q['raw'][qm, 106:]]
                    scaler = StandardScaler().fit(x, sample_weight=weights)
                    numerics[name] = check_scaler(scaler)
                    ridge = Ridge(alpha=100., solver='cholesky').fit(scaler.transform(x), f['raw'][fm, c], sample_weight=weights)
                    hgb = HistGradientBoostingRegressor(**PARAMETERS).fit(x, f['raw'][fm, c], sample_weight=weights)
                    bundle[name] = dict(scaler=scaler, ridge=ridge, hgb=hgb)
                    frame['ridge_'+name] = ridge.predict(scaler.transform(z))
                    frame['hgb_'+name] = hgb.predict(z)
                if not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all(): raise ValueError('Nonfinite predictions')
                joblib.dump(bundle, paths['joblib']); frame.to_parquet(paths['parquet'], index=False)
                info = dict(status='fitted', support=old['support'], contract_sha256=sha256(cp), numerics=numerics,
                    model_sha256=sha256(paths['joblib']), prediction_sha256=sha256(paths['parquet']))
                paths['json'].write_text(json.dumps(info, indent=2)+'\n')
            frames.append(frame); results[variable] = info
    report = Path(f'artifacts/native_reconstruction_suffix_fold{fold}_20261003.json')
    report.write_text(json.dumps(dict(contract=contract, extraction=extraction, targets=results,
        runtime_modified=False, promotion_allowed=False), indent=2, allow_nan=False)+'\n')
    print('Suffix fold complete', fold, len(frames), flush=True)
    return frames, report


def summarize(data):
    results = {v: assess(data[data.variable == v], 20261003+i*1000)
               for i, v in enumerate(sorted(data.variable.unique()))}
    summary = dict(variables=len(results), variable_folds=int(data[['variable', 'fold']].drop_duplicates().shape[0]),
        query_occurrences=len(data), patients=int(data.patient.nunique()),
        normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
            for a in ARMS} for level in ('patient', 'component')},
        screens={a+'_minus_'+b: [v for v, r in results.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
                 for a, b in CONTRASTS})
    return results, summary


def main():
    output = Path('artifacts/native_reconstruction_suffix_assessment_20261003.json')
    if output.exists(): raise ValueError('Completed suffix assessment exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    frames, bindings = [], {}
    for fold in range(5):
        rows, report = run_fold(fold); frames.extend(rows); bindings[str(report)] = sha256(report)
    data = pd.concat(frames, ignore_index=True)
    results, summary = summarize(data)
    if (len(frames), len(data), len(results)) != (177, 88122, 36):
        raise ValueError('E20 supported family not reproduced')
    output.write_text(json.dumps(dict(summary=summary, targets=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Known-content suffix intervention in reused development; not masked inference or forecast.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
