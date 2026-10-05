"""E25 full-support readers; refuse partial extraction and reuse verified raw HGB."""
import argparse
from contextlib import closing
import hashlib
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
from scripts.cache_full_organ_compartments import DEST, PROTOCOL, ROOT, unpack_organs, validate_cache
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from whole_body_jepa.multisource_intervals import sha256

READERS = DEST/'readers'
ARTIFACTS = Path('artifacts')
KINDS = ('body', 'organs', 'shuffled', 'expanded')
ROLES = ('fit', 'selection')


def require_complete_grid(reports):
    expected = {(f, r) for f in range(5) for r in ROLES}
    if set(reports) != expected:
        raise ValueError('All ten completed extraction reports required')
    first = reports[0, 'fit']['contract']
    variables = first['variables']
    if len(variables) != 53 or len(set(variables)) != 53:
        raise ValueError('Complete canonical inventory required')
    shape = first['organ_shape']
    if len(shape) != 2 or min(shape) < 1 or len(first['organ_names']) != shape[0]:
        raise ValueError('Invalid organ schema')
    for (fold, role), report in reports.items():
        contract = report['contract']
        if (contract['fold'], contract['role'], contract['source_fold']) != (fold, role, 0 if fold == 2 else fold):
            raise ValueError('Wrong extraction role or source fold')
        if any(contract[k] != first[k] for k in ('variables', 'organ_shape', 'organ_names')):
            raise ValueError('Inconsistent full-grid schema')
        if (report['complete'] is not True or report['expected'] <= 0
                or report['stored'] != report['expected'] or report['remaining'] != 0):
            raise ValueError('Partial extraction: full-support training prohibited')
        support = report['support']
        if (len(support) != 47 or not set(support) <= set(variables)
                or sum(s['expected'] for s in support.values()) != report['expected']
                or any(s['stored'] != s['expected'] or s['expected'] < 2 for s in support.values())):
            raise ValueError('Incomplete target support')
    if any(set(r['support']) != set(reports[0, 'fit']['support']) for r in reports.values()):
        raise ValueError('Different supported target family')
    return variables, tuple(shape)


def verify_full_grid():
    reports, bindings, checked = {}, {}, {}
    for fold in range(5):
        for role in ROLES:
            path = ARTIFACTS/f'native_reconstruction_full_organs_fold{fold}_{role}_20261003.json'
            if not path.exists():
                raise ValueError('All ten completed extraction reports required')
            reports[fold, role] = json.loads(path.read_text())
            bindings[str(path)] = sha256(path)
    variables, shape = require_complete_grid(reports)
    roles_all = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds']
    for (fold, role), report in reports.items():
        contract = report['contract']
        verify_bound_files(contract['bindings'], checked)
        path = DEST/f'fold{fold}_{role}.sqlite'
        verify_bound_files({str(path): report['private_sha256']}, checked)
        reference = ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
        rp = ARTIFACTS/f'native_reconstruction_nonlinear_cache_fold{fold}_{role}_20261001.json'
        parent = json.loads(rp.read_text())
        if (parent['fold'], parent['role'], parent['examples'], parent['targets']) != (fold, role, report['expected'], 47):
            raise ValueError('Extraction differs from original full support')
        verify_bound_files(parent['bindings'], checked)
        verify_bound_files({str(reference): parent['private_sha256']}, checked)
        if contract['bindings'].get(str(reference)) != parent['private_sha256'] or contract['bindings'].get(str(rp)) != sha256(rp):
            raise ValueError('Unbound original reference')
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
            saved = json.loads(db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0])
            if saved != contract:
                raise ValueError('Changed extraction metadata')
            db.execute('ATTACH DATABASE ? AS reference', (reference.resolve().as_uri()+'?mode=ro',))
            status = validate_cache(db, shape)
            if status != {k: report[k] for k in ('expected', 'stored', 'remaining', 'complete')}:
                raise ValueError('Changed extraction coverage')
            if any(roles_all[fold].get(p) != role for p, in db.execute('SELECT DISTINCT patient FROM compartments')):
                raise ValueError('Extraction patient role leakage')
            counts = {variables[c]: n for c, n in db.execute('SELECT target,COUNT(*) FROM compartments GROUP BY target')}
            if counts != {v: s['stored'] for v, s in report['support'].items()}:
                raise ValueError('Changed target coverage')
        bindings.update(contract['bindings'])
        bindings.update(parent['bindings'])
        bindings[str(path)] = report['private_sha256']
    return reports, variables, shape, bindings


def paired_data(reference, organs, target, shape, roles, role):
    ids, data = load_target(reference, target)
    if any(roles.get(p) != role for p in ids):
        raise ValueError('Wrong-role patient')
    rows = list(organs.execute('SELECT patient,target,reference_digest,payload,digest FROM compartments '
                              'WHERE target=? ORDER BY patient', (target,)))
    if [r[0] for r in rows] != ids:
        raise ValueError('Organ/reference key mismatch')
    digests = dict(reference.execute('SELECT patient,digest FROM features WHERE target=?', (target,)))
    if any(r[2] != digests[r[0]] for r in rows):
        raise ValueError('Changed organ/reference binding')
    data['organs'] = np.stack([unpack_organs(*row, shape) for row in rows]).reshape(len(ids), -1)
    return ids, data


def features(data, ids, kind, projection):
    if kind not in KINDS or len(ids) < 2 or len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError('Distinct sorted same-target patients required')
    body, organs, raw = [np.asarray(data[k]) for k in ('body', 'organs', 'raw')]
    if (body.shape != (len(ids), 64) or raw.shape != (len(ids), 265)
            or organs.ndim != 2 or organs.shape[0] != len(ids)
            or projection.shape != (64, organs.shape[1])
            or not all(np.isfinite(a).all() for a in (body, organs, raw, projection))):
        raise ValueError('Invalid paired readout features')
    process = raw[:, 106:]
    extra = None
    if kind == 'organs':
        extra = organs
    elif kind == 'shuffled':
        extra = np.roll(organs, 1, axis=0)
    elif kind == 'expanded':
        extra = body@projection
    return np.concatenate((body, process) if extra is None else (body, extra, process), axis=1)


def require_model(model, width):
    if (model.n_features_in_ != width or model.n_iter_ != 100
            or any(model.get_params()[k] != v for k, v in PARAMETERS.items())):
        raise ValueError('Changed reader capacity or parameters')


def verify_raw(bundle, saved, meta, fold, target, fit_ids, query_ids, fit, query, roles, components):
    if (set(fit_ids) & set(query_ids) or len(fit_ids) != len(set(fit_ids))
            or len(query_ids) != len(set(query_ids))
            or any(roles.get(p) != 'fit' for p in fit_ids)
            or any(roles.get(p) != 'selection' for p in query_ids)):
        raise ValueError('Paired patient role leakage')
    if (meta['fold'], meta['target'], meta['fit_patients'], meta['selection_patients']) != (
            fold, target, len(fit_ids), len(query_ids)):
        raise ValueError('Changed raw fit support')
    if (bundle['target'] != target or bundle['parameters'] != PARAMETERS
            or saved.patient.duplicated().any() or set(saved.patient) != set(query_ids)
            or not (saved.target == target).all() or not (saved.fold == fold).all()):
        raise ValueError('Changed raw model or prediction grid')
    original = saved.set_index('patient').loc[query_ids]
    np.testing.assert_array_equal(original.component, [components[p] for p in query_ids])
    np.testing.assert_array_equal(original.truth, query['truth'])
    np.testing.assert_array_equal(original.native, query['prediction'])
    model = bundle['models']['raw_probe']
    require_model(model, 265)
    prediction = model.predict(query['raw'])
    np.testing.assert_allclose(prediction, original.raw_probe, rtol=1e-12, atol=1e-12)
    weights = bounded_weights([components[p] for p in fit_ids])
    if not np.isfinite(fit['raw']).all() or not np.isfinite(fit['truth']).all():
        raise ValueError('Invalid raw fitting data')
    return model, prediction, weights


def fit_readers(fit, query, fit_ids, query_ids, weights, projection):
    models, predictions, diagnostics = {}, {}, {}
    y = np.asarray(fit['truth'])
    if y.shape != (len(fit_ids),) or weights.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid fitting labels or weights')
    if set(fit_ids) & set(query_ids):
        raise ValueError('Fitting and selection patients overlap')
    with threadpool_limits(limits=4):
        for kind in KINDS:
            x, q = features(fit, fit_ids, kind, projection), features(query, query_ids, kind, projection)
            model = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=weights)
            require_model(model, x.shape[1])
            pred, train = model.predict(q), model.predict(x)
            if not np.isfinite(pred).all() or not np.isfinite(train).all():
                raise ValueError('Nonfinite reader prediction')
            models[kind], predictions[kind] = model, pred
            diagnostics[kind] = dict(fit_mae=float(np.mean(abs(train-y))),
                weighted_fit_mae=float(np.average(abs(train-y), weights=weights)),
                mean_leaves=float(np.mean([t.get_n_leaf_nodes() for group in model._predictors for t in group])))
    return models, predictions, diagnostics


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run_fold(fold):
    reports, variables, shape, bindings = verify_full_grid()
    checked = {}
    paths = [Path(__file__), PROTOCOL, Path('scripts/train_nonlinear_probe.py'),
             Path('scripts/bounded_site_sampling.py'), Path('scripts/cache_full_organ_compartments.py'),
             Path('scripts/score_monitored_continuation_outer.py')]
    fp, provenance = ROOT/'partitions_20260922.pt', ROOT/'provenance.parquet'
    paths.extend((fp, provenance))
    roles = torch.load(fp, weights_only=False)['folds'][fold]
    links = pd.read_parquet(provenance)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any() or links.isna().any().any():
        raise ValueError('Invalid patient/component mapping')
    components = dict(links.itertuples(index=False, name=None))
    raw_dir = ROOT/f'nonlinear_probe_20261001/fold{fold}_probes'
    rp = ARTIFACTS/f'native_reconstruction_nonlinear_training_fold{fold}_20261001.json'
    raw_report, raw_contract = json.loads(rp.read_text()), json.loads((raw_dir/'contract.json').read_text())
    if (raw_report['fold'] != fold or raw_contract['fold'] != fold
            or raw_report['parameters'] != PARAMETERS or raw_contract['parameters'] != PARAMETERS
            or raw_report['bindings'] != raw_contract['bindings']
            or set(raw_report['targets']) != set(reports[fold, 'fit']['support'])):
        raise ValueError('Changed saved raw training contract')
    verify_bound_files(raw_contract['bindings'], checked)
    # Same bound reference rows, provenance and bounded_weights implementation
    # establish identical fitting support/weights, not merely matching predictions.
    for path in (provenance, fp, Path('scripts/bounded_site_sampling.py')):
        if raw_contract['bindings'].get(str(path)) != sha256(path):
            raise ValueError('Raw weighting dependencies not identical')
    for role in ROLES:
        ref = ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
        if raw_contract['bindings'].get(str(ref)) != reports[fold, role]['contract']['bindings'][str(ref)]:
            raise ValueError('Raw and organ fitting inputs differ')
    paths.extend((rp, raw_dir/'contract.json'))
    bindings.update(raw_contract['bindings'])
    bindings.update({str(p): sha256(p) for p in paths})
    projection = np.random.default_rng(20261003+fold).normal(size=(64, int(np.prod(shape))))/8
    contract = dict(fold=fold, variables=variables, organ_shape=list(shape), parameters=PARAMETERS,
                    projection_seed=20261003+fold, bindings=bindings)
    dest = READERS/f'fold{fold}'
    output = ARTIFACTS/f'native_reconstruction_full_organ_readers_fold{fold}_20261003.json'
    if output.exists():
        raise ValueError('Completed full-support readers already exist')
    dest.mkdir(parents=True, exist_ok=True)
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed reader resume contract')
    if not cp.exists():
        write_json(cp, contract)
    references, organs, results = {}, {}, {}
    try:
        for role in ROLES:
            ref = ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
            references[role] = sqlite3.connect(ref.resolve().as_uri()+'?mode=ro', uri=True)
            organs[role] = sqlite3.connect((DEST/f'fold{fold}_{role}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
        for c, target in enumerate(variables):
            if target not in reports[fold, 'fit']['support']:
                results[target] = dict(status='unsupported', fit_patients=0, selection_patients=0)
                continue
            fids, f = paired_data(references['fit'], organs['fit'], c, shape, roles, 'fit')
            qids, q = paired_data(references['selection'], organs['selection'], c, shape, roles, 'selection')
            old = raw_report['targets'][target]
            if json.loads((raw_dir/f'{target}.json').read_text()) != old:
                raise ValueError('Changed raw target metadata')
            verify_bound_files({str(raw_dir/'contract.json'): old['contract_sha256'],
                str(raw_dir/f'{target}.joblib'): old['model_sha256'],
                str(raw_dir/f'{target}.parquet'): old['prediction_sha256']}, checked)
            raw, raw_prediction, weights = verify_raw(joblib.load(raw_dir/f'{target}.joblib'),
                pd.read_parquet(raw_dir/f'{target}.parquet'), old, fold, target, fids, qids, f, q, roles, components)
            mp, pp, meta = dest/f'{target}.joblib', dest/f'{target}.parquet', dest/f'{target}.json'
            if meta.exists():
                done = json.loads(meta.read_text())
                if (done['fold'], done['target'], done['contract_sha256']) != (fold, target, sha256(cp)):
                    raise ValueError('Changed target resume metadata')
                verify_bound_files({str(mp): done['model_sha256'], str(pp): done['prediction_sha256']}, checked)
                results[target] = done
                continue
            if mp.exists() or pp.exists():
                raise ValueError('Uncommitted reader artifacts require inspection')
            models, predictions, diagnostics = fit_readers(f, q, fids, qids, weights, projection)
            with threadpool_limits(limits=4):
                raw_fit = raw.predict(f['raw'])
            diagnostics['raw'] = dict(fit_mae=float(np.mean(abs(raw_fit-f['truth']))),
                weighted_fit_mae=float(np.average(abs(raw_fit-f['truth']), weights=weights)),
                mean_leaves=float(np.mean([t.get_n_leaf_nodes() for group in raw._predictors for t in group])))
            diagnostics['native'] = dict(fit_mae=float(np.mean(abs(f['prediction']-f['truth']))))
            frame = pd.DataFrame(dict(patient=qids, component=[components[p] for p in qids],
                variable=target, fold=fold, truth=q['truth'], native=q['prediction'], raw=raw_prediction, **predictions))
            tmp = mp.with_suffix('.tmp')
            joblib.dump(dict(models=models, projection=projection, target=target, parameters=PARAMETERS), tmp)
            tmp.replace(mp)
            tmp = pp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(pp)
            done = dict(status='fitted', fold=fold, target=target, fit_patients=len(fids), selection_patients=len(qids),
                contract_sha256=sha256(cp), model_sha256=sha256(mp), prediction_sha256=sha256(pp),
                weights_sha256=hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest(),
                raw_model_sha256=old['model_sha256'], raw_predictions_reproduced=True, fit_diagnostics=diagnostics)
            write_json(meta, done); results[target] = done
            print('Completed full-support target', fold, target, flush=True)
    finally:
        for db in (*references.values(), *organs.values()):
            db.close()
    write_json(output, dict(contract=contract, targets=results, complete=True,
        runtime_modified=False, promotion_allowed=False,
        limitation='Full-support fitting only; all five folds, fixed paired assessment and independent verification still required.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    run_fold(args.fold)


if __name__ == '__main__':
    main()
