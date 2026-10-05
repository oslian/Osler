"""Fixed nonlinear raw/body/process probes; no source-query tuning."""

import argparse
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.cache_nonlinear_probe import decode_features
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


PARAMETERS = dict(loss='absolute_error', max_iter=100, learning_rate=.1,
                  max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.,
                  early_stopping=False, random_state=19)


def fit_probes(fit_body, fit_raw, truth, weights, query_body, query_raw):
    arrays = [np.asarray(x, dtype=float) for x in
              (fit_body, fit_raw, truth, weights, query_body, query_raw)]
    b, r, y, w, qb, qr = arrays
    if (b.shape != (len(y), 64) or r.shape != (len(y), 265) or y.ndim != 1
            or w.shape != y.shape or qb.ndim != 2 or qb.shape[1] != 64
            or qr.shape != (len(qb), 265) or not len(y) or not len(qb)
            or not all(np.isfinite(x).all() for x in arrays) or not (w > 0).all()):
        raise ValueError('Invalid matched probe arrays')
    process, qprocess = r.copy(), qr.copy()
    process[:, :106] = 0
    qprocess[:, :106] = 0
    models, predictions = {}, {}
    with threadpool_limits(limits=4):
        for name, x, q in (('body_probe', b, qb), ('raw_probe', r, qr),
                           ('process_summary', process, qprocess)):
            model = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=w)
            predictions[name] = model.predict(q)
            if not np.isfinite(predictions[name]).all() or model.n_iter_ != 100:
                raise ValueError('Incomplete/nonfinite probe fit')
            models[name] = model
    return models, predictions


def load_target(db, target):
    ids, values = [], []
    for patient, payload, digest in db.execute(
            'SELECT patient,payload,digest FROM features WHERE target=? ORDER BY patient', (target,)):
        ids.append(patient)
        values.append(decode_features(payload, digest))
    if not values or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate target support')
    return ids, {name: np.stack([row[name] for row in values]) for name in values[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    root = Path('data/research_recovery/additional_development_domains_20260921')
    private = root/'nonlinear_probe_20261001'
    dest = private/f'fold{args.fold}_probes'
    dest.mkdir(exist_ok=True)
    final_path = Path(f'artifacts/native_reconstruction_nonlinear_training_fold{args.fold}_20261001.json')
    if final_path.exists():
        raise ValueError('Existing completed probes')
    paths = [Path(__file__), Path('scripts/cache_nonlinear_probe.py'), Path('scripts/bounded_site_sampling.py'),
             Path('scripts/score_monitored_continuation_outer.py'), Path('RECONSTRUCTION_NONLINEAR_PROBE_PROTOCOL_20261001.md')]
    connections = {}
    checked = {}
    for role in ('fit', 'selection'):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{args.fold}_{role}_20261001.json')
        report = json.loads(rp.read_text())
        verify_bound_files(report['bindings'], checked)
        path = private/f'fold{args.fold}_{role}.sqlite'
        if (report['fold'], report['role'], report['targets']) != (args.fold, role, 47):
            raise ValueError('Wrong cache contract')
        if sha256(path) != report['private_sha256']:
            raise ValueError('Changed feature cache')
        connections[role] = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
        paths.extend((rp, path))
    fp, sp, provenance = root/'partitions_20260922.pt', root/'scaling_20260922.pt', root/'provenance.parquet'
    roles = torch.load(fp, weights_only=False)['folds'][args.fold]
    scaling = torch.load(sp, weights_only=False)
    variables = scaling['variables']
    links = pd.read_parquet(provenance)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    base_path = root/f'coverage_selection_20260926/fold{args.fold}.parquet'
    base = pd.read_parquet(base_path).set_index(['patient', 'target'])
    source_path = Path('artifacts/native_reconstruction_full_continuation_selection_20260930.json')
    source = json.loads(source_path.read_text())
    verify_bound_files(source['bindings'], checked)
    paths.extend((fp, sp, provenance, base_path, source_path))
    bindings = {str(p): sha256(p) for p in paths}
    contract = dict(fold=args.fold, parameters=PARAMETERS, bindings=bindings)
    contract_path = dest/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Changed probe resume contract')
    if not contract_path.exists():
        contract_path.write_text(json.dumps(contract, indent=2)+'\n')
    columns = [int(row[0]) for row in connections['fit'].execute('SELECT DISTINCT target FROM features ORDER BY target')]
    query_columns = [int(row[0]) for row in connections['selection'].execute('SELECT DISTINCT target FROM features ORDER BY target')]
    if columns != query_columns or len(columns) != 47:
        raise ValueError('Incomplete paired target family')
    results = {}
    try:
        for c in columns:
            target = variables[c]
            metadata_path = dest/f'{target}.json'
            prediction_path = dest/f'{target}.parquet'
            model_path = dest/f'{target}.joblib'
            if metadata_path.exists():
                done = json.loads(metadata_path.read_text())
                if (done['fold'], done['target'], done['contract_sha256']) != (args.fold, target, sha256(contract_path)):
                    raise ValueError('Wrong saved target metadata')
                if sha256(prediction_path) != done['prediction_sha256'] or sha256(model_path) != done['model_sha256']:
                    raise ValueError('Changed saved target artifact')
                results[target] = done
                continue
            fit_ids, fitting = load_target(connections['fit'], c)
            query_ids, query = load_target(connections['selection'], c)
            if (set(fit_ids) & set(query_ids) or any(roles[p] != 'fit' for p in fit_ids)
                    or any(roles[p] != 'selection' for p in query_ids)):
                raise ValueError('Patient role leakage')
            weights = bounded_weights([components[p] for p in fit_ids])
            models, predictions = fit_probes(fitting['body'], fitting['raw'], fitting['truth'], weights,
                                            query['body'], query['raw'])
            predictions['native'] = query['prediction']
            scaler = scaling['scalers'][args.fold]
            original = base.loc[[(p, target) for p in query_ids]]
            expected_truth = (original.truth.to_numpy()-scaler['location'][c])/scaler['scale'][c]
            if not np.allclose(query['truth'], expected_truth, rtol=1e-10, atol=1e-10):
                raise ValueError('Changed normalized truth')
            predictions['population'] = (original.population.to_numpy()-scaler['location'][c])/scaler['scale'][c]
            frame = pd.DataFrame(dict(patient=query_ids, component=[components[p] for p in query_ids],
                                      target=target, fold=args.fold, truth=query['truth'], **predictions))
            errors = frame[list(predictions)].subtract(frame.truth, axis=0).abs()
            means = dict(patient=errors.mean().to_dict(), component=errors.groupby(frame.component).mean().mean().to_dict())
            for level in ('patient', 'component'):
                if not np.isclose(means[level]['native'], source['folds'][str(args.fold)][target]['normalized_mae'][level]['real10'], atol=1e-10, rtol=1e-10):
                    raise ValueError('Unchanged model reproduction failed')
            tmp = prediction_path.with_suffix('.tmp')
            frame.to_parquet(tmp, index=False)
            tmp.replace(prediction_path)
            tmp = model_path.with_suffix('.tmp')
            joblib.dump(dict(models=models, target=target, parameters=PARAMETERS), tmp)
            tmp.replace(model_path)
            done = dict(fold=args.fold, target=target, fit_patients=len(fit_ids), selection_patients=len(query_ids),
                        contract_sha256=sha256(contract_path), prediction_sha256=sha256(prediction_path),
                        model_sha256=sha256(model_path), normalized_mae=means)
            metadata_path.write_text(json.dumps(done, indent=2)+'\n')
            results[target] = done
            print('Completed fixed probe', args.fold, target, len(results), 'of', len(columns), flush=True)
    finally:
        for db in connections.values():
            db.close()
    final_path.write_text(json.dumps(dict(fold=args.fold, targets=results, parameters=PARAMETERS,
        bindings=bindings, runtime_modified=False, promotion_allowed=False,
        limitation='Single fold fixed source probe stage; no partial-fold efficacy or promotion decision. Frozen encoder, training-only heads, no selection tuning. All five folds and paired component assessment still required. No raw/latent exact-capacity-equivalence claim.'), indent=2)+'\n')
    print('All47 fixed source probes complete', args.fold, flush=True)


if __name__ == '__main__':
    main()
