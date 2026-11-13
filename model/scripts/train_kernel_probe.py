"""Coordinate-independent RBF features with real-label absolute-error readouts."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.diagnose_nonlinear_fit_gap import weighted_median
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices, verify_query
from scripts.train_nonlinear_probe import load_target
from scripts.train_observation_skip import ROOT, PRIVATE, check_masked_target
from whole_body_jepa.multisource_intervals import sha256

DEST = PRIVATE/'kernel_probe_20261003'
ARMS = ('kernel_body','kernel_raw','kernel_process','kernel_shuffle')
SETTINGS = dict(landmarks=128,pairs=1024,C_numerator=100.,tol=1e-6,max_iter=50000,seed=19)
SEED = 20261003


def fit_map(x, weights, seed):
    x, weights = np.asarray(x,dtype=float), np.asarray(weights,dtype=float)
    if (x.ndim!=2 or len(x)<2 or x.shape[1]<2 or weights.shape!=(len(x),)
            or not np.isfinite(x).all() or not np.isfinite(weights).all() or not (weights>0).all()):
        raise ValueError('Invalid kernel fitting arrays')
    center = np.average(x,axis=0,weights=weights)
    z = x-center
    rng = np.random.default_rng(seed)
    pairs = rng.choice(len(x),size=(SETTINGS['pairs'],2),p=weights/weights.sum())
    distances = np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
    positive = distances[distances>0]
    gamma = 1./float(np.median(positive)) if len(positive) else 1.
    kernel = Nystroem(kernel='rbf',gamma=gamma,n_components=min(SETTINGS['landmarks'],len(x)),random_state=SETTINGS['seed'])
    features = kernel.fit_transform(z)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite kernel features')
    return dict(center=center,kernel=kernel), features


def map_query(x, mapping):
    x = np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(mapping['center']) or not np.isfinite(x).all():
        raise ValueError('Invalid kernel query')
    result = mapping['kernel'].transform(x-mapping['center'])
    if not np.isfinite(result).all(): raise ValueError('Nonfinite kernel query')
    return result


def rotation_check(x, features, mapping, model, offset, seed):
    basis = ortho_group.rvs(x.shape[1],random_state=seed)
    kernel = deepcopy(mapping['kernel'])
    kernel.components_ = kernel.components_@basis
    rotated = kernel.transform((x-mapping['center'])@basis)
    feature_error = float(np.max(abs(rotated-features)))
    original = model.predict(features)+offset
    output_error = float(np.max(abs(model.predict(rotated)+offset-original))/max(1.,float(np.max(abs(original)))))
    if feature_error>1e-6 or output_error>1e-6:
        raise ValueError('Rotation invariance failed')
    return dict(feature_error=feature_error,output_error=output_error)


def fit_regressor(x,y,weights):
    offset = weighted_median(y,weights)
    model = LinearSVR(loss='epsilon_insensitive',epsilon=0.,C=SETTINGS['C_numerator']/len(y),
        tol=SETTINGS['tol'],max_iter=SETTINGS['max_iter'],dual=True,
        fit_intercept=True,random_state=SETTINGS['seed'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(x,y-offset,sample_weight=weights)
    nonconvergence = sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    other = [str(w.message) for w in caught if not issubclass(w.category,ConvergenceWarning)]
    if other: raise ValueError('Unexpected fitting warnings: '+str(other))
    return model, offset, dict(iterations=int(model.n_iter_),convergence_warnings=nonconvergence,
        converged=bool(not nonconvergence and model.n_iter_<SETTINGS['max_iter']))


def fit_heads(fitting,query,people,components,seed):
    weights = bounded_weights(components)
    donor = donor_indices(people,components,seed)
    models,predictions,checks = {},{},{}
    with threadpool_limits(limits=4):
        for source in ('body','raw','process'):
            x = np.array(fitting['body' if source=='body' else 'raw'],dtype=float,copy=True)
            q = np.array(query['body' if source=='body' else 'raw'],dtype=float,copy=True)
            if source=='process': x[:,:106]=0; q[:,:106]=0
            mapping,features = fit_map(x,weights,seed)
            qfeatures = map_query(q,mapping)
            names = ('kernel_body','kernel_shuffle') if source=='body' else ('kernel_'+source,)
            for arm in names:
                train_x = features[donor] if arm=='kernel_shuffle' else features
                model,offset,diag = fit_regressor(train_x,fitting['truth'],weights)
                predictions[arm] = model.predict(qfeatures)+offset
                if not np.isfinite(predictions[arm]).all(): raise ValueError('Nonfinite prediction')
                checks[arm] = dict(diag,fit_mae=float(np.mean(abs(model.predict(train_x)+offset-fitting['truth']))),
                    feature_width=features.shape[1],gamma=float(mapping['kernel'].gamma))
                if arm!='kernel_shuffle':
                    checks[arm]['fit_rotation'] = rotation_check(x,features,mapping,model,offset,seed)
                    checks[arm]['query_rotation'] = rotation_check(q,qfeatures,mapping,model,offset,seed)
                models[arm] = dict(mapping=mapping,model=model,offset=offset)
    return models,predictions,checks,int(np.count_nonzero(donor==np.arange(len(donor))))


def run(fold,checked):
    dest = DEST/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    pp = Path(f'artifacts/native_reconstruction_observation_skip_fold{fold}_20261003.json')
    proof = json.loads(pp.read_text()); verify_bound_files(proof['bindings'],checked)
    if proof['fold']!=fold or len(proof['targets'])!=47: raise ValueError('Wrong source family')
    paths = [Path(__file__),Path('scripts/score_kernel_probe.py'),Path('scripts/diagnose_nonlinear_fit_gap.py'),
        Path('RECONSTRUCTION_KERNEL_PROTOCOL_20261003.md'),pp,*[Path(p) for p in proof['bindings']]]
    contract = dict(fold=fold,settings=SETTINGS,seed=SEED,bindings={str(p):sha256(p) for p in paths})
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract: raise ValueError('Changed resume contract')
    if not cp.exists(): cp.write_text(json.dumps(contract,indent=2)+'\n')
    roles = torch.load(ROOT/'partitions_20260922.pt',weights_only=False)['folds'][fold]
    variables = torch.load(ROOT/'scaling_20260922.pt',weights_only=False)['variables']
    links = pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous components')
    components = dict(links.itertuples(index=False,name=None))
    dbs = {r:sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) for r in ('fit','selection')}
    results = {}
    try:
        for target in sorted(proof['targets']):
            rp,mp,qp = [dest/f'{target}.{s}' for s in ('json','joblib','parquet')]
            if rp.exists():
                done = json.loads(rp.read_text())
                if (done['contract_sha256']!=sha256(cp) or done['target']!=target or done['fold']!=fold
                        or done['model_sha256']!=sha256(mp) or done['prediction_sha256']!=sha256(qp)):
                    raise ValueError('Changed complete target')
                results[target]=done; continue
            c = variables.index(target)
            fit_ids,fitting = load_target(dbs['fit'],c); ids,query = load_target(dbs['selection'],c)
            groups = [components[p] for p in fit_ids]
            if any(roles[p]!='fit' for p in fit_ids): raise ValueError('Incorrect fitting role')
            prior = pd.read_parquet(PRIVATE/f'fold{fold}_probes/{target}.parquet')
            paired = verify_query(ids,query,prior,roles,'selection',fit_ids,groups)
            if not np.array_equal(paired.component,[components[p] for p in ids]): raise ValueError('Changed query provenance')
            for data in (fitting,query): check_masked_target(data['raw'],c)
            models,predictions,checks,singletons = fit_heads(fitting,query,fit_ids,groups,SEED+c)
            frame = paired.reset_index().assign(**predictions)
            tmp=mp.with_suffix('.tmp'); joblib.dump(models,tmp); tmp.replace(mp)
            tmp=qp.with_suffix('.tmp'); frame.to_parquet(tmp,index=False); tmp.replace(qp)
            done=dict(target=target,fold=fold,fit_patients=len(fit_ids),selection_patients=len(ids),checks=checks,
                unchanged_singletons=singletons,contract_sha256=sha256(cp),model_sha256=sha256(mp),prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done,indent=2,allow_nan=False)+'\n'); results[target]=done
            print('Kernel',fold,target,len(results),'of47',flush=True)
    finally:
        for db in dbs.values(): db.close()
    out=Path(f'artifacts/native_reconstruction_kernel_fold{fold}_20261003.json')
    report=dict(contract,targets=results,runtime_modified=False,promotion_allowed=False)
    if out.exists() and json.loads(out.read_text())!=report: raise ValueError('Changed final report')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    checked={}
    for fold in range(5): run(fold,checked)
