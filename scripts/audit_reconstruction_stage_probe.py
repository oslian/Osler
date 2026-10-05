"""Fixed source-only probes before and after persistent state aggregation."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.audit_reconstruction_raw_state import summarize
from scripts.bounded_site_sampling import bounded_weights
from scripts.cache_nonlinear_probe import decode_features
from scripts.encoder_fold_scaling import transform_example
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache

ROOT = Path('data/research_recovery/additional_development_domains_20260921')
DEST = ROOT/'stage_probe_20261003'
SOURCES = ('event_mean', 'event_last', 'body', 'raw', 'process')


def choose(ids, limit):
    if len(ids) != len(set(ids)) or limit <= 0:
        raise ValueError('Invalid sampling population')
    return sorted(ids, key=lambda p: (hashlib.sha256(
        ('stage-probe-20261003:'+str(p)).encode()).hexdigest(), str(p)))[:limit]


def event_summaries(jepa, variables, example):
    tokens, times = example['tokens'], example['event_hours']
    anchor = example['anchor_hour']
    if (times.shape != (len(tokens),) or not len(times)
            or not torch.isfinite(times).all() or not np.isfinite(anchor)
            or (times > anchor).any() or (times[1:] < times[:-1]).any()):
        raise ValueError('Invalid available-time history')
    clean, _ = mask_native_target(tokens, times, variables, variables[example['target_index']])
    with torch.inference_mode():
        body, _, _ = encode_native_organ_observations(jepa, variables, clean)
    if not torch.isfinite(body).all():
        raise ValueError('Nonfinite event representation')
    return body.mean(0).cpu().numpy(), body[-1].cpu().numpy()


def inputs(values):
    raw = np.asarray(values['raw'], dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 265:
        raise ValueError('Wrong raw schema')
    process = raw[:, 106:]
    result = dict(raw=raw, process=process)
    for name in ('event_mean', 'event_last', 'body'):
        state = np.asarray(values[name], dtype=float)
        if state.shape != (len(raw), 64):
            raise ValueError('Wrong stage schema')
        result[name] = np.c_[state, process]
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite features')
    return result


def fit_predict(fit, query, weights):
    x, q = inputs(fit), inputs(query)
    y = np.asarray(fit['truth'])
    predictions = {}
    with threadpool_limits(limits=4):
        for name in SOURCES:
            hgb = HistGradientBoostingRegressor(**PARAMETERS).fit(x[name], y, sample_weight=weights)
            scaler = StandardScaler().fit(x[name], sample_weight=weights)
            ridge = Ridge(alpha=100., solver='cholesky').fit(scaler.transform(x[name]), y,
                                                          sample_weight=weights)
            predictions['hgb_'+name] = hgb.predict(q[name])
            predictions['ridge_'+name] = ridge.predict(scaler.transform(q[name]))
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Nonfinite predictions')
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    dest = DEST/f'fold{args.fold}'
    dest.mkdir(parents=True, exist_ok=True)
    output = Path(f'artifacts/native_reconstruction_stage_fold{args.fold}_20261003.json')
    if output.exists():
        raise ValueError('Completed fold already exists')
    fp, sp = ROOT/'partitions_20260922.pt', ROOT/'scaling_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds'][args.fold]
    scaling = torch.load(sp, weights_only=False)
    variables, scaler = scaling['variables'], scaling['scalers'][args.fold]
    schema_path = Path('models/whole_body_joint_jepa.pt')
    schema = torch.load(schema_path, weights_only=False)
    source_fold = 0 if args.fold == 2 else args.fold
    weight_path = ROOT/f'monitored_continuation_20260929/fold{source_fold}_real_epoch10.pt'
    ck = torch.load(weight_path, weights_only=False)
    if (ck['fold'], ck['epoch'], ck['arm']) != (source_fold, 10, 'real') or variables != schema['variables']:
        raise ValueError('Changed frozen model')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema), variables, ('map',))).eval()
    model.load_state_dict(ck['state_dict'], strict=True)
    provenance = ROOT/'provenance.parquet'
    links = pd.read_parquet(provenance)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    history_path = ROOT/'full_histories_20260922.sqlite'
    source = ReconstructionCache(history_path)
    dbs = {}
    paths = [Path(__file__), Path('RECONSTRUCTION_STAGE_PROBE_PROTOCOL_20261003.md'),
             fp, sp, provenance, schema_path, weight_path, history_path,
             Path('whole_body_jepa/native_organ_observations.py'),
             Path('scripts/train_nonlinear_probe.py'), Path('scripts/encoder_fold_scaling.py')]
    verified = {str(p): sha256(p) for p in paths}
    for role in ('fit', 'selection'):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{args.fold}_{role}_20261001.json')
        report = json.loads(rp.read_text())
        cache = ROOT/f'nonlinear_probe_20261001/fold{args.fold}_{role}.sqlite'
        if (report['fold'], report['role'], report['targets']) != (args.fold, role, 47):
            raise ValueError('Changed original cache scope')
        for p in (weight_path, history_path, fp, sp):
            if verified[str(p)] != report['bindings'][str(p)]:
                raise ValueError('Changed extraction dependency')
        cache_hash = sha256(cache)
        if cache_hash != report['private_sha256']:
            raise ValueError('Changed original features')
        verified.update({str(rp): sha256(rp), str(cache): cache_hash})
        dbs[role] = sqlite3.connect(cache.resolve().as_uri()+'?mode=ro', uri=True)
    contract = dict(fold=args.fold, fit_limit=512, selection_limit=256,
                    parameters=PARAMETERS, ridge_alpha=100., bindings=verified)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    cp.write_text(json.dumps(contract, indent=2)+'\n')
    columns = [r[0] for r in dbs['fit'].execute('SELECT DISTINCT target FROM features ORDER BY target')]
    if len(columns) != 47 or columns != [r[0] for r in dbs['selection'].execute('SELECT DISTINCT target FROM features ORDER BY target')]:
        raise ValueError('Incomplete target family')
    results = {}
    try:
        for c in columns:
            target = variables[c]
            metadata, predicted = dest/f'{target}.json', dest/f'{target}.parquet'
            if metadata.exists():
                saved = json.loads(metadata.read_text())
                if saved['contract_sha256'] != sha256(cp) or saved['prediction_sha256'] != sha256(predicted):
                    raise ValueError('Changed saved output')
                results[target] = saved
                continue
            data, identities = {}, {}
            for role, limit in (('fit', 512), ('selection', 256)):
                eligible = [r[0] for r in dbs[role].execute('SELECT patient FROM features WHERE target=?', (c,))]
                ids = choose(eligible, limit)
                if not ids or any(roles[p] != role for p in ids):
                    raise ValueError('Wrong or empty patient role')
                rows = []
                for i, patient in enumerate(ids):
                    payload, digest = dbs[role].execute('SELECT payload,digest FROM features WHERE patient=? AND target=?', (patient,c)).fetchone()
                    original = decode_features(payload, digest)
                    example = transform_example(source.get(patient,c), scaler)
                    raw, _ = summarize(example['tokens'],example['event_hours'],example['anchor_hour'],variables,target)
                    if not np.array_equal(raw,original['raw']) or not np.isclose(example['truth_normalized'], original['truth'],atol=1e-12,rtol=0):
                        raise ValueError('Changed masked example')
                    if i == 0:
                        with torch.inference_mode():
                            native = float(model(example['tokens'],example['event_hours'],example['anchor_hour'],target)[0])
                        if not np.isclose(native, original['prediction'],atol=1e-7,rtol=0):
                            raise ValueError('Original forward mismatch')
                    mean,last = event_summaries(model.trajectory.jepa,variables,example)
                    rows.append(dict(original,event_mean=mean,event_last=last))
                data[role] = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
                identities[role] = ids
            if set(identities['fit']) & set(identities['selection']):
                raise ValueError('Overlapping roles')
            weights = bounded_weights([components[p] for p in identities['fit']])
            preds = fit_predict(data['fit'],data['selection'],weights)
            frame = pd.DataFrame(dict(patient=identities['selection'],
                component=[components[p] for p in identities['selection']], target=target,
                fold=args.fold, truth=data['selection']['truth'], **preds))
            tmp = predicted.with_suffix('.tmp')
            frame.to_parquet(tmp,index=False); tmp.replace(predicted)
            saved = dict(target=target,fold=args.fold,fit_patients=len(identities['fit']),
                selection_patients=len(frame),contract_sha256=sha256(cp),prediction_sha256=sha256(predicted),
                normalized_mae={k:float(np.mean(abs(v-frame.truth.to_numpy()))) for k,v in preds.items()})
            metadata.write_text(json.dumps(saved,indent=2)+'\n')
            results[target] = saved
            print('Stage probes',args.fold,len(results),'/47',target,flush=True)
    finally:
        source.close()
        for db in dbs.values(): db.close()
    output.write_text(json.dumps(dict(contract=contract,targets=results,runtime_modified=False,
        promotion_allowed=False,limitation='Bounded diagnostic. All folds required, no partial efficacy claim.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
