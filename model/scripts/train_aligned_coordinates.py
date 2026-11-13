"""Observation-aligned axes of frozen body state; no extra query information."""
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices, verify_query
from scripts.train_nonlinear_probe import PARAMETERS, load_target
from scripts.train_observation_skip import ROOT, PRIVATE, check_masked_target
from whole_body_jepa.multisource_intervals import sha256

DEST = PRIVATE/'aligned_coordinates_20261003'
ARMS = ('aligned_body', 'random_axes', 'shuffled_axes')
SEED = 20261003
ALPHA = 100.


def fit_axes(body, raw, weights, people, groups, seed):
    b, r, w = [np.asarray(x, dtype=float) for x in (body, raw, weights)]
    if (b.ndim != 2 or r.ndim != 2 or not len(b) or len(r) != len(b)
            or w.shape != (len(b),) or len(people) != len(b) or len(groups) != len(b)
            or not all(np.isfinite(x).all() for x in (b,r,w)) or not (w > 0).all()):
        raise ValueError('Invalid fit axes')
    bs = StandardScaler().fit(b, sample_weight=w)
    rs = StandardScaler().fit(r, sample_weight=w)
    x, y = bs.transform(b), rs.transform(r)
    constant = np.ptp(r, axis=0) == 0
    y[:, constant] = 0
    donors = donor_indices(people, groups, seed)
    real = Ridge(alpha=ALPHA, solver='cholesky').fit(x,y,sample_weight=w)
    null = Ridge(alpha=ALPHA, solver='cholesky').fit(x,y[donors],sample_weight=w)
    basis = np.random.default_rng(seed).normal(size=(b.shape[1],r.shape[1]))
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return dict(body_scaler=bs, raw_scaler=rs, constant=constant,
        aligned_body=real, shuffled_axes=null, random_axes=basis,
        unchanged_singletons=int((donors == np.arange(len(b))).sum()))


def transform(body, mapping, arm):
    b = np.asarray(body, dtype=float)
    if b.ndim != 2 or not len(b) or not np.isfinite(b).all() or arm not in ARMS:
        raise ValueError('Invalid body query')
    x = mapping['body_scaler'].transform(b)
    if arm == 'random_axes':
        axes = x@mapping[arm]
    else:
        axes = mapping[arm].predict(x)
        axes[:, mapping['constant']] = 0
    result = np.concatenate((b,axes),axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite decoded axes')
    return result


def retention(raw, decoded, mapping):
    """Observed numeric entries only; missing zeros are not observed truth."""
    raw = np.asarray(raw, dtype=float)
    actual = mapping['raw_scaler'].transform(raw)
    decoded = np.asarray(decoded, dtype=float)
    if raw.shape[1] != 265 or decoded.shape != actual.shape:
        raise ValueError('Wrong observation summary width')
    output = {}
    for name, left, right in (('latest',0,53),('mean',53,106),('process',106,265)):
        ratios, supported = [], 0
        for c in range(left,right):
            valid = raw[:,106+c%53] > 0 if c < 106 else np.ones(len(raw),dtype=bool)
            if mapping['constant'][c] or valid.sum() < 30:
                continue
            denominator = float(np.mean(actual[valid,c]**2))
            if denominator <= 1e-12:
                continue
            ratios.append(float(np.mean((decoded[valid,c]-actual[valid,c])**2)/denominator))
            supported += int(valid.sum())
        output[name] = dict(features=len(ratios), entries=supported,
            median_mse_ratio=None if not ratios else float(np.median(ratios)),
            better_than_fit_mean=sum(x < 1 for x in ratios))
    return output


def fit_heads(fitting, query, weights, people, groups, seed):
    with threadpool_limits(limits=4):
        mapping = fit_axes(fitting['body'],fitting['raw'],weights,people,groups,seed)
        models, predictions, checks = {}, {}, {}
        for arm in ARMS:
            x, q = [transform(d['body'],mapping,arm) for d in (fitting,query)]
            model = HistGradientBoostingRegressor(**PARAMETERS).fit(x,fitting['truth'],sample_weight=weights)
            predictions[arm] = model.predict(q)
            if model.n_iter_ != 100 or not np.isfinite(predictions[arm]).all():
                raise ValueError('Incomplete/nonfinite HGB')
            checks[arm] = dict(width=x.shape[1],fit_mae=float(np.mean(abs(model.predict(x)-fitting['truth']))))
            models[arm] = model
        diagnostic = {role:retention(d['raw'],transform(d['body'],mapping,'aligned_body')[:,64:],mapping)
                      for role,d in (('fit',fitting),('selection',query))}
    return dict(mapping=mapping,models=models), predictions, checks, diagnostic


def run(fold,checked):
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    proof_path = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    proof = json.loads(proof_path.read_text()); verify_bound_files(proof['bindings'],checked)
    if proof['fold'] != fold or proof['parameters'] != PARAMETERS or len(proof['targets']) != 47:
        raise ValueError('Wrong source family')
    paths = [Path(__file__),Path('scripts/score_aligned_coordinates.py'),
        Path('scripts/train_nonlinear_placebo.py'),Path('RECONSTRUCTION_ALIGNED_COORDINATE_PROTOCOL_20261003.md'),
        proof_path,*[Path(p) for p in proof['bindings']]]
    contract = dict(fold=fold,parameters=PARAMETERS,seed=SEED,alpha=ALPHA,
        bindings={str(p):sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('Changed resume contract')
    if not cp.exists(): cp.write_text(json.dumps(contract,indent=2)+'\n')
    roles = torch.load(ROOT/'partitions_20260922.pt',weights_only=False)['folds'][fold]
    variables = torch.load(ROOT/'scaling_20260922.pt',weights_only=False)['variables']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False,name=None))
    dbs = {r:sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
           for r in ('fit','selection')}
    results = {}
    try:
        for target in sorted(proof['targets']):
            rp,mp,qp = [dest/f'{target}.{s}' for s in ('json','joblib','parquet')]
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256'] != sha256(cp) or done['target'] != target or done['fold'] != fold
                        or done['model_sha256'] != sha256(mp) or done['prediction_sha256'] != sha256(qp)):
                    raise ValueError('Changed completed target')
                results[target] = done; continue
            c = variables.index(target)
            fit_ids,fitting = load_target(dbs['fit'],c)
            ids,query = load_target(dbs['selection'],c)
            groups = [components[p] for p in fit_ids]
            if any(roles[p] != 'fit' for p in fit_ids): raise ValueError('Wrong fitting role')
            old = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            paired = verify_query(ids,query,old,roles,'selection',fit_ids,groups)
            if not np.array_equal(paired.component,[components[p] for p in ids]):
                raise ValueError('Changed query component')
            for d in (fitting,query): check_masked_target(d['raw'],c)
            models,pred,checks,diagnostic = fit_heads(fitting,query,bounded_weights(groups),fit_ids,groups,SEED+c)
            frame = paired.reset_index().assign(**pred)
            tmp=mp.with_suffix('.tmp');joblib.dump(models,tmp);tmp.replace(mp)
            tmp=qp.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(qp)
            done = dict(target=target,fold=fold,fit_patients=len(fit_ids),selection_patients=len(ids),
                checks=checks,retention=diagnostic,unchanged_singletons=models['mapping']['unchanged_singletons'],
                contract_sha256=sha256(cp),model_sha256=sha256(mp),prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done,indent=2,allow_nan=False)+'\n')
            results[target]=done
            print('Aligned coordinates',fold,target,len(results),'of47',flush=True)
    finally:
        for db in dbs.values(): db.close()
    out=Path(f'artifacts/native_reconstruction_aligned_coordinates_fold{fold}_20261003.json')
    report=dict(contract,targets=results,runtime_modified=False,promotion_allowed=False)
    if out.exists() and json.loads(out.read_text()) != report: raise ValueError('Changed completed report')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    checked={}
    for fold in range(5): run(fold,checked)
