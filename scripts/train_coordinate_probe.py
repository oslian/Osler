"""Lossless coordinates isolate finite-readout geometry, not information removal."""

import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ortho_group
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import verify_query
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from scripts.train_observation_skip import ROOT, PRIVATE, check_masked_target
from whole_body_jepa.multisource_intervals import sha256

DEST = PRIVATE/'coordinates_20261003'
ARMS = ('body_pca', 'body_rotation', 'raw_pca', 'raw_rotation')
SEED = 20261003


def coordinates(x, weights, seed):
    x, weights = np.asarray(x, dtype=float), np.asarray(weights, dtype=float)
    if (x.ndim != 2 or not len(x) or x.shape[1] < 2 or weights.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(weights).all() or not (weights > 0).all()):
        raise ValueError('Invalid fitting coordinates')
    center = np.average(x, axis=0, weights=weights)
    z = x-center
    covariance = (z.T * (weights/weights.sum())) @ z
    _, vectors = eigh(covariance)
    bases = dict(pca=vectors[:, ::-1], rotation=ortho_group.rvs(x.shape[1], random_state=seed))
    for basis in bases.values():
        if not np.allclose(basis.T@basis, np.eye(x.shape[1]), atol=1e-10, rtol=0):
            raise ValueError('Nonorthogonal coordinates')
    return {name: dict(center=center, basis=basis) for name,basis in bases.items()}


def transform(x, mapping):
    x = np.asarray(x, dtype=float)
    center, basis = mapping['center'], mapping['basis']
    if (x.ndim != 2 or x.shape[1] != len(center) or basis.shape != (len(center), len(center))
            or not len(x) or not np.isfinite(x).all() or not np.isfinite(basis).all()
            or not np.isfinite(center).all()):
        raise ValueError('Invalid transform input')
    z = (x-center)@basis
    restored = z@basis.T+center
    relative = float(np.max(abs(restored-x))/max(1., float(np.max(abs(x)))))
    if not np.isfinite(z).all() or relative > 1e-10:
        raise ValueError('Lossy transformation')
    return z, relative


def fit_heads(fitting, query, weights, seed):
    result, prediction, checks = {}, {}, {}
    with threadpool_limits(limits=4):
        for source in ('body', 'raw'):
            for method, mapping in coordinates(fitting[source], weights, seed).items():
                name = source+'_'+method
                x, fit_error = transform(fitting[source], mapping)
                q, query_error = transform(query[source], mapping)
                model = HistGradientBoostingRegressor(**PARAMETERS).fit(x, fitting['truth'], sample_weight=weights)
                if model.n_iter_ != 100:
                    raise ValueError('Incomplete fit')
                prediction[name] = model.predict(q)
                if not np.isfinite(prediction[name]).all():
                    raise ValueError('Nonfinite predictions')
                result[name] = dict(model=model, mapping=mapping)
                checks[name] = dict(fit_inverse_error=fit_error, query_inverse_error=query_error,
                    width=x.shape[1], fit_mae=float(np.mean(abs(model.predict(x)-fitting['truth']))))
    return result, prediction, checks


def run(fold, checked):
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    proof_path = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    proof = json.loads(proof_path.read_text())
    verify_bound_files(proof['bindings'], checked)
    if proof['fold'] != fold or proof['parameters'] != PARAMETERS or len(proof['targets']) != 47:
        raise ValueError('Wrong source family')
    paths = [Path(__file__), Path('scripts/score_coordinate_probe.py'),
        Path('RECONSTRUCTION_COORDINATE_PROTOCOL_20261003.md'), proof_path,
        *[Path(p) for p in proof['bindings']]]
    contract = dict(fold=fold, parameters=PARAMETERS, seed=SEED,
        bindings={str(p): sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=2)+'\n')
    roles = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds'][fold]
    variables = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)['variables']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous patient component')
    components = dict(links.itertuples(index=False, name=None))
    dbs = {r: sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
           for r in ('fit', 'selection')}
    results = {}
    try:
        for target in sorted(proof['targets']):
            rp, mp, qp = [dest/f'{target}.{s}' for s in ('json', 'joblib', 'parquet')]
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256'] != sha256(cp) or done['target'] != target or done['fold'] != fold
                        or done['model_sha256'] != sha256(mp) or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed completed target')
                results[target] = done
                continue
            c = variables.index(target)
            fit_ids, fitting = load_target(dbs['fit'], c)
            ids, query = load_target(dbs['selection'], c)
            groups = [components[p] for p in fit_ids]
            if any(roles[p] != 'fit' for p in fit_ids):
                raise ValueError('Incorrect fitting role')
            old = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            paired = verify_query(ids, query, old, roles, 'selection', fit_ids, groups)
            if not np.array_equal(paired.component, [components[p] for p in ids]):
                raise ValueError('Incorrect query component')
            for data in (fitting, query):
                check_masked_target(data['raw'], c)
            models, prediction, checks = fit_heads(fitting, query, bounded_weights(groups), SEED+c)
            frame = paired.reset_index().assign(**prediction)
            tmp = mp.with_suffix('.tmp'); joblib.dump(models, tmp); tmp.replace(mp)
            tmp = qp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(qp)
            done = dict(target=target, fold=fold, fit_patients=len(fit_ids), selection_patients=len(ids),
                checks=checks, contract_sha256=sha256(cp), model_sha256=sha256(mp), prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done, indent=2, allow_nan=False)+'\n')
            results[target] = done
            print('Coordinates', fold, target, len(results), 'of47', flush=True)
    finally:
        for db in dbs.values():
            db.close()
    out = Path(f'artifacts/native_reconstruction_coordinates_fold{fold}_20261003.json')
    report = dict(contract, targets=results, runtime_modified=False, promotion_allowed=False)
    if out.exists() and json.loads(out.read_text()) != report:
        raise ValueError('Changed complete report')
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    checked = {}
    for fold in range(5):
        run(fold, checked)
