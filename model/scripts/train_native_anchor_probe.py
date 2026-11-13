"""Fixed truth-supervised nonlinear correction of the SAME frozen body state."""

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
from scripts.train_observation_skip import ROOT, PRIVATE, check_masked_target
from whole_body_jepa.multisource_intervals import sha256


NEW_ARMS = ('anchored_residual', 'absolute_anchor', 'anchor_only', 'shuffled_anchor')
SEED = 20261003
DEST = PRIVATE / 'native_anchor_20261003'


def inputs(body, native, arm, donor=None):
    body, native = np.asarray(body, dtype=float), np.asarray(native, dtype=float)
    if (native.ndim != 1 or not len(native) or body.shape != (len(native), 64)
            or not np.isfinite(body).all() or not np.isfinite(native).all() or arm not in NEW_ARMS):
        raise ValueError('Invalid anchored input')
    body = body.copy()
    if arm == 'anchor_only':
        body[:] = 0
    if donor is not None:
        donor = np.asarray(donor)
        if (arm != 'shuffled_anchor' or donor.shape != (len(native),)
                or not np.issubdtype(donor.dtype, np.integer)
                or not np.array_equal(np.sort(donor), np.arange(len(native)))):
            raise ValueError('Invalid body-only permutation')
        body = body[donor]
    return np.column_stack((body, native))


def labels(truth, native, arm):
    truth, native = np.asarray(truth, dtype=float), np.asarray(native, dtype=float)
    if (arm not in NEW_ARMS or truth.ndim != 1 or not len(truth) or native.shape != truth.shape
            or not np.isfinite(truth).all() or not np.isfinite(native).all()):
        raise ValueError('Invalid real-label target')
    return truth if arm == 'absolute_anchor' else truth-native


def predict(model, body, native, arm):
    value = model.predict(inputs(body, native, arm))
    if arm != 'absolute_anchor':
        value = value + native
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite anchored prediction')
    return value


def fit_heads(body, native, truth, people, components, seed):
    donor = donor_indices(people, components, seed)
    weights = bounded_weights(components)
    models = {}
    with threadpool_limits(limits=4):
        for arm in NEW_ARMS:
            x = inputs(body, native, arm, donor if arm == 'shuffled_anchor' else None)
            y = labels(truth, native, arm)
            if len(x) != len(people) or len(y) != len(people):
                raise ValueError('Unpaired fitting inputs')
            model = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=weights)
            if model.n_iter_ != PARAMETERS['max_iter']:
                raise ValueError('Incomplete fixed fit')
            models[arm] = model
    return models, int(np.count_nonzero(donor == np.arange(len(donor))))


def fit_fold(fold, checked):
    dest = DEST/f'fold{fold}'
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('RECONSTRUCTION_NATIVE_ANCHOR_PROTOCOL_20261003.md'),
        Path('scripts/score_native_anchor_probe.py'), Path('scripts/train_nonlinear_probe.py'),
        Path('scripts/train_nonlinear_placebo.py'), Path('scripts/train_observation_skip.py'),
        Path('scripts/bounded_site_sampling.py'), Path('scripts/score_monitored_continuation_outer.py')]
    proof_path = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    proof = json.loads(proof_path.read_text())
    verify_bound_files(proof['bindings'], checked)
    if proof['fold'] != fold or proof['parameters'] != PARAMETERS or len(proof['targets']) != 47:
        raise ValueError('Wrong source contract')
    paths.append(proof_path)
    paths.extend(Path(p) for p in proof['bindings'])
    roles = torch.load(ROOT/'partitions_20260922.pt', weights_only=False)['folds'][fold]
    variables = torch.load(ROOT/'scaling_20260922.pt', weights_only=False)['variables']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous provenance')
    components = dict(links.itertuples(index=False, name=None))
    contract = dict(fold=fold, parameters=PARAMETERS, seed=SEED,
                    bindings={str(p): sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    if not cp.exists():
        cp.write_text(json.dumps(contract, indent=2)+'\n')
    dbs = {r: sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
           for r in ('fit', 'selection')}
    results = {}
    try:
        for target in sorted(proof['targets']):
            rp, mp, qp = dest/f'{target}.json', dest/f'{target}.joblib', dest/f'{target}.parquet'
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256'] != sha256(cp) or done['target'] != target or done['fold'] != fold
                        or done['model_sha256'] != sha256(mp) or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed completed fit')
                results[target] = done
                continue
            c = variables.index(target)
            fit_ids, fitting = load_target(dbs['fit'], c)
            ids, query = load_target(dbs['selection'], c)
            groups = [components[p] for p in fit_ids]
            if any(roles[p] != 'fit' for p in fit_ids):
                raise ValueError('Wrong fitting role')
            prior = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            expected = verify_query(ids, query, prior, roles, 'selection', fit_ids, groups)
            if not np.array_equal(expected.component, [components[p] for p in ids]):
                raise ValueError('Changed query provenance')
            for data in (fitting, query):
                check_masked_target(data['raw'], c)
            op = PRIVATE/f'observation_skip_20261003/fold{fold}/{target}.parquet'
            if sha256(op) != proof['targets'][target]['prediction_sha256']:
                raise ValueError('Changed observation-skip comparator')
            oldskip = pd.read_parquet(op)
            if oldskip.patient.duplicated().any() or set(oldskip.patient) != set(ids):
                raise ValueError('Unpaired observation comparator')
            oldskip = oldskip.set_index('patient').loc[ids]
            for col in ('truth', 'native', 'body_probe', 'raw_probe', 'population', 'component'):
                if not np.array_equal(expected[col], oldskip[col]):
                    raise ValueError('Changed reference row')
            models, singletons = fit_heads(fitting['body'], fitting['prediction'], fitting['truth'],
                                           fit_ids, groups, SEED+c)
            frame = expected.reset_index()
            frame['observation_skip'] = oldskip.observation_skip.to_numpy()
            fit_predictions = dict(native=fitting['prediction'])
            with threadpool_limits(limits=4):
                for arm in NEW_ARMS:
                    frame[arm] = predict(models[arm], query['body'], query['prediction'], arm)
                    fit_predictions[arm] = predict(models[arm], fitting['body'], fitting['prediction'], arm)
            fit_errors = pd.DataFrame({a: abs(p-fitting['truth']) for a, p in fit_predictions.items()})
            fit_metrics = dict(patient=fit_errors.mean().to_dict(),
                component=fit_errors.groupby(np.asarray(groups)).mean().mean().to_dict(),
                bounded={a: float(np.average(v, weights=bounded_weights(groups))) for a,v in fit_errors.items()})
            tmp = mp.with_suffix('.tmp'); joblib.dump(models, tmp); tmp.replace(mp)
            tmp = qp.with_suffix('.tmp'); frame.to_parquet(tmp, index=False); tmp.replace(qp)
            done = dict(target=target, fold=fold, fit_patients=len(fit_ids), selection_patients=len(ids),
                unchanged_singletons=singletons, fit_truth_mae=fit_metrics, contract_sha256=sha256(cp),
                model_sha256=sha256(mp), prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done, indent=2, allow_nan=False)+'\n')
            results[target] = done
            print('Native-anchor fit', fold, target, len(results), 'of47', flush=True)
    finally:
        for db in dbs.values():
            db.close()
    out = Path(f'artifacts/native_reconstruction_native_anchor_fold{fold}_20261003.json')
    report = dict(contract, targets=results, runtime_modified=False, promotion_allowed=False,
        limitation='Real-label source fitting; frozen encoder saw fit patients, native fit outputs not OOF. Shuffled-state null retains real native estimate. Fit null metrics use own state. No outer/runtime/interval change.')
    if out.exists() and json.loads(out.read_text()) != report:
        raise ValueError('Changed completed report')
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return out


if __name__ == '__main__':
    checked = {}
    for fold in range(5):
        fit_fold(fold, checked)
