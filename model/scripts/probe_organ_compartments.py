"""Read all compartments of one frozen masked-patient state, without promotion."""
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
from scripts.probe_observed_content import DEST as OBSERVED, ROOT
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_content_retention import build_fold
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256

DEST = ROOT / 'organ_compartments_20261003'
PROTOCOL = Path('RECONSTRUCTION_ORGAN_READOUT_PROTOCOL_20261003.md')
KINDS = ('body', 'organs', 'shuffled', 'expanded', 'raw')
ARMS = tuple(f'{r}_{k}' for r in ('ridge', 'hgb') for k in KINDS)
CONTRASTS = tuple((f'{r}_organs', f'{r}_{k}') for r in ('ridge', 'hgb')
                  for k in ('body', 'shuffled', 'expanded', 'raw'))


def replay_full(layer, example):
    body, organs, presence, times = [example[k] for k in ('body', 'organs', 'presence', 'times')]
    if (not len(times) or times.shape != (len(body),) or not torch.isfinite(times).all()
            or not np.isfinite(example['anchor']) or (times > example['anchor']).any()
            or (times[1:] <= times[:-1]).any()):
        raise ValueError('Invalid compartment event times')
    state = layer.initialize(body.new_zeros((1, layer.body_dim)),
        body.new_zeros((1, layer.organ_count, layer.organ_dim)), body.new_zeros((1, layer.organ_count)))
    relative = (times-times[0]).to(body.dtype)
    for i in range(len(times)):
        delta = relative[i:i+1]-state.elapsed_hours
        state = layer.advance(state, delta)
        state = layer.update(state, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
    gap = (times.new_tensor(example['anchor'])-times[-1]).to(body.dtype).reshape(1)
    return layer.advance(state, gap)


def donor_indices(patient, target):
    patient, target = np.asarray(patient), np.asarray(target)
    if patient.ndim != 1 or target.shape != patient.shape or len(set(patient)) != len(patient):
        raise ValueError('One context per distinct patient required')
    donor = np.full(len(patient), -1, dtype=int)
    for c in np.unique(target):
        rows = np.flatnonzero(target == c)
        if len(rows) < 2:
            continue
        ordered = rows[np.argsort(patient[rows], kind='stable')]
        donor[ordered] = np.roll(ordered, 1)
    return donor


def features(data, rows, kind, projection):
    if kind not in KINDS:
        raise ValueError('Unknown representation')
    body, raw = data['body'][rows], data['raw'][rows]
    if kind == 'raw':
        result = raw
    elif kind == 'body':
        result = np.c_[body, raw[:, 106:]]
    else:
        if kind == 'organs':
            extra = data['organs'][rows]
        elif kind == 'expanded':
            extra = body @ projection
        else:
            donors = data['donor'][rows]
            if (donors < 0).any() or not np.array_equal(data['target'][rows], data['target'][donors]):
                raise ValueError('No independent same-mask organ donor')
            extra = data['organs'][donors]
        result = np.c_[body, extra, raw[:, 106:]]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite compartment features')
    return result


def assess(frame, seed, repetitions=5000):
    if (frame.empty or frame.variable.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS, 'native']].to_numpy()).all()):
        raise ValueError('Invalid compartment pairs')
    patient = frame[[*ARMS, 'native']].subtract(frame.truth, axis=0).abs().groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    component = patient.groupby(links.reindex(patient.index)).mean()
    comparisons = {}
    for i, (a, b) in enumerate(CONTRASTS):
        stats = dict(patient=paired_summary(patient[a]-patient[b], seed+i, repetitions),
                     component=paired_summary(component[a]-component[b], seed+100+i, repetitions))
        passed = len(patient) >= 30 and len(component) >= 3 and all(s['ci95'] is not None
            and s['ci95'][1] < 0 and s['p_less'] is not None and s['p_less'] < .05/(53*8)
            for s in stats.values())
        comparisons[a+'_minus_'+b] = dict(paired=stats, screen_pass=bool(passed))
    return dict(patients=len(patient), components=len(component), query_occurrences=len(frame),
        folds=sorted(map(int, frame.fold.unique())),
        normalized_mae=dict(patient=patient.mean().to_dict(), component=component.mean().to_dict()),
        comparisons=comparisons)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run_fold(fold):
    variables, examples, original, parent, encoded_meta, _ = build_fold(fold)
    report_path = Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
    observed = json.loads(report_path.read_text())
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    bindings = {str(p): sha256(p) for p in (Path(__file__), PROTOCOL,
        Path('scripts/train_content_retention.py'), Path('scripts/train_nonlinear_probe.py'),
        Path('scripts/bounded_site_sampling.py'), Path('scripts/score_additional_reconstruction.py'),
        Path('whole_body_jepa/physiology_state.py'), report_path)}
    contract = dict(fold=fold, encoded_contract=parent, bindings=bindings, parameters=PARAMETERS,
                    seed=20261003+fold, ridge_alpha=100., variables=variables)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed compartment contract')
    if not cp.exists(): write_json(cp, contract)
    data, extraction = {}, {}
    with torch.no_grad():
        for role, rows in examples.items():
            op = OBSERVED/f'fold{fold}/{role}.joblib'
            if sha256(op) != observed['extraction'][role]['sha256']:
                raise ValueError('Changed raw/body parent cache')
            old = joblib.load(op)
            path, mp = dest/f'{role}.joblib', dest/f'{role}.json'
            if mp.exists():
                meta = json.loads(mp.read_text())
                if meta['sha256'] != sha256(path) or meta['contract_sha256'] != sha256(cp):
                    raise ValueError('Changed compartment feature cache')
                d = joblib.load(path)
            else:
                bs, os, body_errors, point_errors = [], [], [], []
                for i, row in enumerate(rows):
                    if (row['patient'], row['target']) != (old['patient'][i], old['context'][i]):
                        raise ValueError('Changed query identity')
                    if row['mask'][row['target']]:
                        raise ValueError('Target observation not masked')
                    state = replay_full(original.trajectory.layer, row)
                    b, o = state.body[0].numpy(), state.organs[0].numpy().reshape(-1)
                    np.testing.assert_allclose(b, old['body'][i], atol=1e-6, rtol=0)
                    prediction = float(original.readout(state.body)[0, row['target']])
                    np.testing.assert_allclose(prediction, row['frozen'], atol=1e-7, rtol=0)
                    body_errors.append(float(np.max(np.abs(b-old['body'][i]))))
                    point_errors.append(abs(prediction-row['frozen']))
                    bs.append(b); os.append(o)
                d = dict(body=np.stack(bs), organs=np.stack(os), raw=old['raw'],
                    patient=np.array([r['patient'] for r in rows]), target=np.array([r['target'] for r in rows]),
                    component=np.array([r['component'] for r in rows]), truth=np.array([r['truth'] for r in rows]),
                    native=np.array([r['frozen'] for r in rows]))
                d['donor'] = donor_indices(d['patient'], d['target'])
                joblib.dump(d, path)
                meta = dict(sha256=sha256(path), contract_sha256=sha256(cp), encoded=encoded_meta[role],
                    raw_sha256=sha256(op), contexts=len(rows), body_max_error=max(body_errors),
                    native_max_error=max(point_errors))
                write_json(mp, meta)
            if list(zip(d['patient'], d['target'])) != [(r['patient'], r['target']) for r in rows]:
                raise ValueError('Cached context mismatch')
            data[role], extraction[role] = d, meta
            print('Compartments extracted', fold, role, len(rows), flush=True)
    f, q = data['fit'], data['selection']
    if set(f['patient']) & set(q['patient']):
        raise ValueError('Patient role leakage')
    projection = np.random.default_rng(20261003+fold).normal(size=(64, f['organs'].shape[1]))/8.
    frames, results = [], {}
    for c, variable in enumerate(variables):
        fm, qm = np.flatnonzero(f['target'] == c), np.flatnonzero(q['target'] == c)
        support = dict(fit=len(fm), selection=len(qm))
        if len(fm) < 30 or len(qm) < 5:
            results[variable] = dict(status='insufficient_support', support=support)
            continue
        paths = {s: dest/f'{variable}.{s}' for s in ('joblib', 'parquet', 'json')}
        frame = pd.DataFrame(dict(patient=q['patient'][qm], component=q['component'][qm],
            variable=variable, fold=fold, truth=q['truth'][qm], native=q['native'][qm]))
        if paths['json'].exists():
            info = json.loads(paths['json'].read_text())
            if (info['contract_sha256'] != sha256(cp) or info['model_sha256'] != sha256(paths['joblib'])
                    or info['prediction_sha256'] != sha256(paths['parquet'])):
                raise ValueError('Changed fitted reader')
            saved = pd.read_parquet(paths['parquet'])
            pd.testing.assert_frame_equal(saved[frame.columns], frame)
            frame = saved
        else:
            bundle, numerics, fit_errors = {}, {}, {}
            weights = bounded_weights(f['component'][fm].tolist())
            for kind in KINDS:
                x, z = features(f, fm, kind, projection), features(q, qm, kind, projection)
                scaler = StandardScaler().fit(x, sample_weight=weights)
                numerics[kind] = check_scaler(scaler)
                ridge = Ridge(alpha=100., solver='cholesky').fit(scaler.transform(x), f['truth'][fm], sample_weight=weights)
                hgb = HistGradientBoostingRegressor(**PARAMETERS).fit(x, f['truth'][fm], sample_weight=weights)
                bundle[kind] = dict(scaler=scaler, ridge=ridge, hgb=hgb)
                for reader, model in (('ridge', ridge), ('hgb', hgb)):
                    frame[reader+'_'+kind] = model.predict(scaler.transform(z) if reader == 'ridge' else z)
                    pred_fit = model.predict(scaler.transform(x) if reader == 'ridge' else x)
                    fit_errors[reader+'_'+kind] = float(np.mean(np.abs(pred_fit-f['truth'][fm])))
            if not np.isfinite(frame[['truth', *ARMS, 'native']].to_numpy()).all():
                raise ValueError('Nonfinite predictions')
            joblib.dump(dict(models=bundle, projection=projection), paths['joblib'])
            frame.to_parquet(paths['parquet'], index=False)
            info = dict(status='fitted', support=support, contract_sha256=sha256(cp),
                numerics=numerics, fit_mae=fit_errors, model_sha256=sha256(paths['joblib']),
                prediction_sha256=sha256(paths['parquet']))
            write_json(paths['json'], info)
        frames.append(frame); results[variable] = info
        print('Compartments fitted', fold, variable, support, flush=True)
    report = Path(f'artifacts/native_reconstruction_organ_compartments_fold{fold}_20261003.json')
    write_json(report, dict(contract=contract, extraction=extraction, targets=results,
        runtime_modified=False, promotion_allowed=False))
    return frames, report, variables


def summarize(data, variables):
    results = {v: assess(data[data.variable == v], 20261003+i*1000) if (data.variable == v).any()
               else dict(status='insufficient_support') for i, v in enumerate(variables)}
    scored = {v: r for v, r in results.items() if 'normalized_mae' in r}
    screens = {a+'_minus_'+b: [v for v, r in scored.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
               for a, b in CONTRASTS}
    summary = dict(variables=len(scored), variable_folds=int(data[['variable', 'fold']].drop_duplicates().shape[0]),
        query_occurrences=len(data), patients=int(data.patient.nunique()),
        normalized_macro={level: {a: float(np.mean([r['normalized_mae'][level][a] for r in scored.values()]))
            for a in (*ARMS, 'native')} for level in ('patient', 'component')}, screens=screens)
    summary['joint_incremental'] = [v for v in scored if all(
        v in screens[f'{r}_organs_minus_{b}'] for r in ('ridge', 'hgb') for b in ('body', 'shuffled', 'expanded'))]
    summary['joint_over_raw'] = [v for v in summary['joint_incremental'] if all(
        v in screens[f'{r}_organs_minus_raw'] for r in ('ridge', 'hgb'))]
    return results, summary


def main():
    output = Path('artifacts/native_reconstruction_organ_compartments_assessment_20261003.json')
    if output.exists(): raise ValueError('Completed E23 assessment exists')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    frames, bindings = [], {}
    with threadpool_limits(limits=4):
        for fold in range(5):
            rows, report, variables = run_fold(fold)
            frames.extend(rows); bindings[str(report)] = sha256(report)
    data = pd.concat(frames, ignore_index=True)
    results, summary = summarize(data, variables)
    write_json(output, dict(summary=summary, targets=results, bindings=bindings,
        runtime_modified=False, promotion_allowed=False,
        limitation='Masked current inference on reused bounded development, not future or external promotion.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
