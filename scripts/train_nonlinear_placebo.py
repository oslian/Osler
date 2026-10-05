"""Same-dimensional readout null using within-fit-component state reassignment."""

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
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from whole_body_jepa.multisource_intervals import sha256


SEED = 20261003


def donor_indices(people, components, seed):
    people, components = np.asarray(people), np.asarray(components)
    if (people.ndim != 1 or components.shape != people.shape or not len(people)
            or len(set(people)) != len(people)):
        raise ValueError('Invalid fit identity')
    rng = np.random.default_rng(seed)
    donor = np.arange(len(people))
    for group in sorted(set(components)):
        ix = np.flatnonzero(components == group)
        ix = ix[np.argsort(people[ix])]
        order = rng.permutation(ix)
        if len(order) > 1:
            donor[order] = np.roll(order, 1)
    if (len(set(donor)) != len(donor) or not np.array_equal(components, components[donor])):
        raise ValueError('Invalid within-component permutation')
    return donor


def fit_placebo(body, truth, people, components, seed):
    body, truth = np.asarray(body), np.asarray(truth)
    if (body.shape != (len(people), 64) or truth.shape != (len(people),)
            or not np.isfinite(body).all() or not np.isfinite(truth).all()):
        raise ValueError('Invalid fit data')
    donor = donor_indices(people, components, seed)
    with threadpool_limits(limits=4):
        model = HistGradientBoostingRegressor(**PARAMETERS).fit(
            body[donor], truth, sample_weight=bounded_weights(components))
    if model.n_iter_ != PARAMETERS['max_iter']:
        raise ValueError('Incomplete fit')
    return model, int(np.count_nonzero(donor == np.arange(len(donor))))


def verify_query(ids, values, prior, roles, role, fit_ids, fit_components):
    if (len(set(ids)) != len(ids) or set(ids) & set(fit_ids)
            or any(roles[p] != role for p in ids) or prior.patient.duplicated().any()
            or set(ids) != set(prior.patient)):
        raise ValueError('Changed query identity/role')
    expected = prior.set_index('patient').loc[ids]
    if set(expected.component) & set(fit_components):
        raise ValueError('Fit/query component overlap')
    if not np.allclose(values['truth'], expected.truth, rtol=1e-10, atol=1e-10):
        raise ValueError('Changed query truth')
    if not np.allclose(values['prediction'], expected.native, rtol=1e-10, atol=1e-10):
        raise ValueError('Changed native readout')
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    fold = parser.parse_args().fold
    root = Path('data/research_recovery/additional_development_domains_20260921')
    original = root / 'nonlinear_probe_20261001'
    dest = original / f'placebo_20261003/fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    out = Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
    if out.exists():
        raise ValueError('Existing complete placebo')
    paths = [Path(__file__), Path('RECONSTRUCTION_NONLINEAR_PLACEBO_PROTOCOL_20261003.md'),
        Path('scripts/train_nonlinear_probe.py'), Path('scripts/cache_nonlinear_probe.py'),
        Path('scripts/bounded_site_sampling.py'), Path('scripts/score_monitored_continuation_outer.py')]
    proofs, checked = {}, {}
    names = dict(fit=f'cache_fold{fold}_fit_20261001',
                 selection=f'cache_fold{fold}_selection_20261001',
                 test=f'outer_fold{fold}_20261002', training=f'training_fold{fold}_20261001')
    for role, name in names.items():
        p = Path(f'artifacts/native_reconstruction_nonlinear_{name}.json')
        proofs[role] = json.loads(p.read_text())
        if proofs[role]['fold'] != fold:
            raise ValueError('Wrong proof fold')
        verify_bound_files(proofs[role]['bindings'], checked)
        paths.append(p)
    fp, sp, pp = root/'partitions_20260922.pt', root/'scaling_20260922.pt', root/'provenance.parquet'
    roles = torch.load(fp, weights_only=False)['folds'][fold]
    variables = torch.load(sp, weights_only=False)['variables']
    links = pd.read_parquet(pp)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    paths.extend((fp, sp, pp))
    databases = {}
    for role in ('fit', 'selection', 'test'):
        p = original/(f'outer_20261002/fold{fold}_features.sqlite' if role == 'test' else f'fold{fold}_{role}.sqlite')
        key = 'feature_sha256' if role == 'test' else 'private_sha256'
        if sha256(p) != proofs[role][key] or proofs[role]['role'] != role:
            raise ValueError('Changed feature cache')
        databases[role] = sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True)
        paths.append(p)
    outer_path = original/f'outer_20261002/fold{fold}.parquet'
    if sha256(outer_path) != proofs['test']['prediction_sha256']:
        raise ValueError('Changed outer predictions')
    outer = pd.read_parquet(outer_path)
    paths.append(outer_path)
    targets = sorted(proofs['training']['targets'])
    if len(targets) != 47:
        raise ValueError('Changed target family')
    for target, meta in proofs['training']['targets'].items():
        for suffix, key in (('joblib', 'model_sha256'), ('parquet', 'prediction_sha256')):
            p = original/f'fold{fold}_probes/{target}.{suffix}'
            if sha256(p) != meta[key]:
                raise ValueError('Changed original head or predictions')
            paths.append(p)
    bindings = {str(p): sha256(p) for p in paths}
    contract = dict(fold=fold, parameters=PARAMETERS, seed=SEED, bindings=bindings)
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
                        or done['model_sha256'] != sha256(mp) or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed completed target')
                results[target] = done
                continue
            c = variables.index(target)
            fit_ids, fitting = load_target(databases['fit'], c)
            if any(roles[p] != 'fit' for p in fit_ids):
                raise ValueError('Invalid fit role')
            fit_groups = [components[p] for p in fit_ids]
            model, unchanged = fit_placebo(fitting['body'], fitting['truth'], fit_ids, fit_groups, SEED+c)
            bundle = joblib.load(original/f'fold{fold}_probes/{target}.joblib')
            if bundle['parameters'] != PARAMETERS or bundle['target'] != target:
                raise ValueError('Wrong real model')
            real = bundle['models']['body_probe']
            if real.n_iter_ != 100 or any(real.get_params()[k] != v for k,v in PARAMETERS.items()):
                raise ValueError('Wrong real-model capacity')
            frames = []
            for role in ('selection', 'test'):
                prior = (pd.read_parquet(original/f'fold{fold}_probes/{target}.parquet')
                         if role == 'selection' else outer.loc[outer.target == target])
                if prior.empty:
                    if role != 'test' or target not in proofs['test']['unsupported_targets']:
                        raise ValueError('Unexpected empty query')
                    continue
                ids, values = load_target(databases[role], c)
                expected = verify_query(ids, values, prior, roles, role, fit_ids, fit_groups)
                if not np.array_equal(expected.component, [components[p] for p in ids]):
                    raise ValueError('Changed query provenance')
                with threadpool_limits(limits=4):
                    exact = real.predict(values['body'])
                    prediction = model.predict(values['body'])
                if (not np.allclose(exact, expected.body_probe, atol=1e-12, rtol=1e-12)
                        or not np.isfinite(prediction).all()):
                    raise ValueError('Real model parity or finite control failure')
                frame = expected.reset_index()
                frame['role'], frame['trained_shuffle'] = role, prediction
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True)
            tmp = qp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(qp)
            tmp = mp.with_suffix('.tmp'); joblib.dump(model, tmp); tmp.replace(mp)
            done = dict(target=target, fold=fold, fit_patients=len(fit_ids), unchanged_singletons=unchanged,
                query_counts=frame.groupby('role').size().to_dict(), contract_sha256=sha256(cp),
                model_sha256=sha256(mp), prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done, indent=2)+'\n')
            results[target] = done
            print('Completed matched placebo', fold, target, len(results), 'of47', flush=True)
    finally:
        for db in databases.values():
            db.close()
    out.write_text(json.dumps(dict(contract, targets=results, runtime_modified=False, promotion_allowed=False,
        limitation='Same-dimensional body readout null, fixed within-fit-component cyclic shuffle. Encoder stays real-trained; single placebo seed, reused development; not full encoder null, forecast or runtime promotion.'), indent=2)+'\n')
    print('Complete all47 placebo fold', fold, flush=True)


if __name__ == '__main__':
    main()
