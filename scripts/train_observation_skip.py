"""Source-only, fixed nonlinear observation skip on the same frozen body state."""

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
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices, verify_query
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from whole_body_jepa.multisource_intervals import sha256


ROOT = Path('data/research_recovery/additional_development_domains_20260921')
PRIVATE = ROOT / 'nonlinear_probe_20261001'
NEW_ARMS = ('observation_skip', 'process_skip', 'shuffled_skip')
SEED = 20261003


def inputs(body, raw, arm, donor=None):
    body, raw = np.asarray(body, dtype=float), np.asarray(raw, dtype=float)
    if (body.ndim != 2 or body.shape[1] != 64 or not len(body)
            or raw.shape != (len(body), 265) or not np.isfinite(body).all()
            or not np.isfinite(raw).all() or arm not in NEW_ARMS):
        raise ValueError('Invalid observation-skip inputs')
    raw = raw.copy()
    if arm == 'process_skip':
        raw[:, :106] = 0
    if donor is not None:
        donor = np.asarray(donor)
        if (arm != 'shuffled_skip' or donor.shape != (len(body),)
                or not np.issubdtype(donor.dtype, np.integer)
                or not np.array_equal(np.sort(donor), np.arange(len(body)))):
            raise ValueError('Invalid paired-branch permutation')
        raw = raw[donor]
    return np.concatenate((body, raw), axis=1)


def check_masked_target(raw, c):
    raw = np.asarray(raw)
    if not np.array_equal(raw[:, [c, 53+c, 106+c, 212+c]],
                          np.zeros((len(raw), 4))):
        raise ValueError('Target leaked into observation branch')


def fit_heads(body, raw, truth, people, components, seed):
    y = np.asarray(truth, dtype=float)
    if y.shape != (len(people),) or not np.isfinite(y).all():
        raise ValueError('Invalid fitting truth')
    donor = donor_indices(people, components, seed)
    weights = bounded_weights(components)
    models = {}
    with threadpool_limits(limits=4):
        for arm in NEW_ARMS:
            x = inputs(body, raw, arm, donor if arm == 'shuffled_skip' else None)
            if len(x) != len(y):
                raise ValueError('Unpaired fitting rows')
            models[arm] = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=weights)
            if models[arm].n_iter_ != 100:
                raise ValueError('Incomplete fixed fit')
    return models, int(np.count_nonzero(donor == np.arange(len(donor))))


def run(fold, checked):
    out = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    dest = PRIVATE / f'observation_skip_20261003/fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('RECONSTRUCTION_OBSERVATION_SKIP_PROTOCOL_20261003.md'),
        Path('scripts/train_nonlinear_probe.py'), Path('scripts/train_nonlinear_placebo.py'),
        Path('scripts/cache_nonlinear_probe.py'), Path('scripts/bounded_site_sampling.py'),
        Path('scripts/score_monitored_continuation_outer.py')]
    proofs = {}
    for role in ('fit', 'selection', 'training'):
        part = f'training_fold{fold}' if role == 'training' else f'cache_fold{fold}_{role}'
        p = Path(f'artifacts/native_reconstruction_nonlinear_{part}_20261001.json')
        proofs[role] = json.loads(p.read_text())
        verify_bound_files(proofs[role]['bindings'], checked)
        if proofs[role]['fold'] != fold:
            raise ValueError('Wrong original fold')
        paths.append(p)
    fp, sp, pp = ROOT/'partitions_20260922.pt', ROOT/'scaling_20260922.pt', ROOT/'provenance.parquet'
    roles = torch.load(fp, weights_only=False)['folds'][fold]
    variables = torch.load(sp, weights_only=False)['variables']
    links = pd.read_parquet(pp)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    paths.extend((fp, sp, pp))
    databases = {}
    for role in ('fit', 'selection'):
        p = PRIVATE/f'fold{fold}_{role}.sqlite'
        if proofs[role]['role'] != role or sha256(p) != proofs[role]['private_sha256']:
            raise ValueError('Changed feature cache')
        databases[role] = sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True)
        paths.append(p)
    targets = sorted(proofs['training']['targets'])
    if len(targets) != 47 or proofs['training']['parameters'] != PARAMETERS:
        raise ValueError('Changed family or reference parameters')
    for target, meta in proofs['training']['targets'].items():
        for suffix, key in (('parquet', 'prediction_sha256'), ('joblib', 'model_sha256')):
            p = PRIVATE/f'fold{fold}_probes/{target}.{suffix}'
            if sha256(p) != meta[key]:
                raise ValueError('Changed original predictions/model')
            paths.append(p)
    contract = dict(fold=fold, parameters=PARAMETERS, seed=SEED,
                    bindings={str(p): sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=2)+'\n')
    results = {}
    try:
        for target in targets:
            rp, mp, qp = dest/f'{target}.json', dest/f'{target}.joblib', dest/f'{target}.parquet'
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256'] != sha256(cp) or done['target'] != target
                        or done['fold'] != fold or done['model_sha256'] != sha256(mp)
                        or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed completed target')
                results[target] = done
                continue
            c = variables.index(target)
            fit_ids, fitting = load_target(databases['fit'], c)
            ids, query = load_target(databases['selection'], c)
            prior = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            groups = [components[p] for p in fit_ids]
            if any(roles[p] != 'fit' for p in fit_ids):
                raise ValueError('Invalid fit role')
            expected = verify_query(ids, query, prior, roles, 'selection', fit_ids, groups)
            if not np.array_equal(expected.component, [components[p] for p in ids]):
                raise ValueError('Changed query component')
            check_masked_target(fitting['raw'], c)
            check_masked_target(query['raw'], c)
            original = joblib.load(PRIVATE/f'fold{fold}_probes/{target}.joblib')
            with threadpool_limits(limits=4):
                parity = original['models']['body_probe'].predict(query['body'])
            if not np.allclose(parity, expected.body_probe, atol=1e-12, rtol=1e-12):
                raise ValueError('Original prediction reproduction failed')
            models, singletons = fit_heads(fitting['body'], fitting['raw'], fitting['truth'],
                                           fit_ids, groups, SEED+c)
            frame = expected.reset_index()
            with threadpool_limits(limits=4):
                for arm in NEW_ARMS:
                    frame[arm] = models[arm].predict(inputs(query['body'], query['raw'], arm))
            if not np.isfinite(frame[list(NEW_ARMS)]).all().all():
                raise ValueError('Nonfinite candidate predictions')
            tmp = mp.with_suffix('.tmp'); joblib.dump(models, tmp); tmp.replace(mp)
            tmp = qp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(qp)
            error = frame[list(NEW_ARMS)].subtract(frame.truth, axis=0).abs()
            done = dict(target=target, fold=fold, fit_patients=len(fit_ids), selection_patients=len(ids),
                unchanged_singletons=singletons, contract_sha256=sha256(cp),
                prediction_sha256=sha256(qp), model_sha256=sha256(mp),
                normalized_mae=dict(patient=error.mean().to_dict(),
                    component=error.groupby(frame.component).mean().mean().to_dict()))
            rp.write_text(json.dumps(done, indent=2, allow_nan=False)+'\n')
            results[target] = done
            print('Observation skip complete', fold, target, len(results), 'of47', flush=True)
    finally:
        for db in databases.values():
            db.close()
    report = dict(contract, targets=results, runtime_modified=False, promotion_allowed=False,
        limitation='Source development only; same frozen body with explicit masked observation branch. Not full encoder null, future forecast or promotion. All47/fivefold assessment required.')
    if out.exists() and json.loads(out.read_text()) != report:
        raise ValueError('Changed complete report')
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('All47 observation skips complete', fold, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    checked = {}
    for f in (range(5) if args.fold is None else [args.fold]):
        run(f, checked)
