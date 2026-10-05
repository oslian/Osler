"""Fixed C-times-ten intervention with exactly the same saved feature maps."""
import json
from pathlib import Path
import sqlite3
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_kernel_probe import map_query,SEED
from scripts.train_scaled_kernel import DEST as OLD_DEST,ROOT,PRIVATE
from scripts.train_nonlinear_placebo import donor_indices,verify_query
from scripts.train_nonlinear_probe import load_target
from whole_body_jepa.multisource_intervals import sha256

DEST=PRIVATE/'relaxed_kernel_20261003'
MULTIPLIER=10.
ARMS=('relaxed_body','relaxed_raw','relaxed_process','relaxed_shuffle')


def refit(old, features, truth, weights):
    model=clone(old['model']).set_params(C=old['model'].C*MULTIPLIER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(features,truth-old['offset'],sample_weight=weights)
    count=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    if any(not issubclass(w.category,ConvergenceWarning) for w in caught):
        raise ValueError('Unexpected fit warning')
    saved=dict(old,model=model)
    prediction=model.predict(features)+old['offset']
    if not np.isfinite(prediction).all(): raise ValueError('Nonfinite fit')
    return saved,dict(iterations=int(model.n_iter_),convergence_warnings=count,
        converged=bool(not count and model.n_iter_<model.max_iter),C=float(model.C),
        fit_mae=float(np.mean(abs(prediction-truth))),fit_weighted_mae=float(np.average(abs(prediction-truth),weights=weights)))


def fit_heads(old,fitting,query,people,groups,seed,prior):
    weights=bounded_weights(groups); donor=donor_indices(people,groups,seed)
    models,predictions,checks={},{},{}
    with threadpool_limits(limits=4):
        for source in ('body','raw','process'):
            key='body' if source=='body' else 'raw'; parent=old['scaled_'+source]
            x,q=[np.array(d[key],float,copy=True) for d in (fitting,query)]
            if source=='process': x[:,:106]=0; q[:,:106]=0
            z=map_query(parent['scaler'].transform(x),parent['mapping'])
            qz=map_query(parent['scaler'].transform(q),parent['mapping'])
            names=('body','shuffle') if source=='body' else (source,)
            for name in names:
                a='relaxed_'+name; olda='scaled_'+name; saved=old[olda]
                train=z[donor] if name=='shuffle' else z
                expected=saved['model'].predict(qz)+saved['offset']
                if not np.allclose(expected,prior[olda],rtol=1e-10,atol=1e-10): raise ValueError('Old query not reproduced')
                models[a],checks[a]=refit(saved,train,fitting['truth'],weights)
                checks[a]['old_fit_mae']=float(np.mean(abs(saved['model'].predict(train)+saved['offset']-fitting['truth'])))
                predictions[a]=models[a]['model'].predict(qz)+saved['offset']
                if not np.isfinite(predictions[a]).all(): raise ValueError('Nonfinite query')
    return models,predictions,checks,int(np.count_nonzero(donor==np.arange(len(donor))))


def run(fold,checked):
    dest=DEST/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    pp=Path(f'artifacts/native_reconstruction_scaled_kernel_fold{fold}_20261003.json')
    proof=json.loads(pp.read_text()); verify_bound_files(proof['bindings'],checked)
    if proof['fold']!=fold or len(proof['targets'])!=47: raise ValueError('Wrong source family')
    paths=[Path(__file__),Path('scripts/score_relaxed_kernel.py'),Path('RECONSTRUCTION_KERNEL_RELAXATION_PROTOCOL_20261003.md'),pp,*map(Path,proof['bindings'])]
    contract=dict(fold=fold,C_multiplier=MULTIPLIER,seed=SEED,bindings={str(p):sha256(p) for p in paths})
    cp=dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract: raise ValueError('Changed resume contract')
    if not cp.exists(): cp.write_text(json.dumps(contract,indent=2)+'\n')
    roles=torch.load(ROOT/'partitions_20260922.pt',weights_only=False)['folds'][fold]
    variables=torch.load(ROOT/'scaling_20260922.pt',weights_only=False)['variables']
    links=pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    components=dict(links.itertuples(index=False,name=None))
    dbs={r:sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) for r in ('fit','selection')}
    results={}
    try:
        for t in sorted(proof['targets']):
            rp,mp,qp=[dest/f'{t}.{s}' for s in ('json','joblib','parquet')]
            if rp.exists():
                done=json.loads(rp.read_text())
                if (done['contract_sha256']!=sha256(cp) or done['model_sha256']!=sha256(mp)
                        or done['prediction_sha256']!=sha256(qp) or done['target']!=t or done['fold']!=fold): raise ValueError('Changed saved target')
                results[t]=done; continue
            c=variables.index(t); fit_ids,fitting=load_target(dbs['fit'],c); ids,query=load_target(dbs['selection'],c)
            groups=[components[p] for p in fit_ids]
            if any(roles[p]!='fit' for p in fit_ids): raise ValueError('Wrong fit role')
            oldp,oldm=[OLD_DEST/f'fold{fold}/{t}.{s}' for s in ('parquet','joblib')]; meta=proof['targets'][t]
            if sha256(oldp)!=meta['prediction_sha256'] or sha256(oldm)!=meta['model_sha256']: raise ValueError('Changed parent artifacts')
            paired=verify_query(ids,query,pd.read_parquet(oldp),roles,'selection',fit_ids,groups)
            if not np.array_equal(paired.component,[components[p] for p in ids]): raise ValueError('Changed components')
            old=joblib.load(oldm)
            models,predictions,checks,singletons=fit_heads(old,fitting,query,fit_ids,groups,SEED+c,paired)
            for a,d in checks.items():
                if not np.isclose(d['old_fit_mae'],meta['checks'][a.replace('relaxed_','scaled_')]['fit_mae'],rtol=1e-10,atol=1e-12): raise ValueError('Old fit changed')
            tmp=mp.with_suffix('.tmp'); joblib.dump(models,tmp); tmp.replace(mp)
            tmp=qp.with_suffix('.tmp'); paired.reset_index().assign(**predictions).to_parquet(tmp,index=False); tmp.replace(qp)
            done=dict(target=t,fold=fold,fit_patients=len(fit_ids),selection_patients=len(ids),checks=checks,
                      unchanged_singletons=singletons,contract_sha256=sha256(cp),model_sha256=sha256(mp),prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done,indent=2,allow_nan=False)+'\n'); results[t]=done
            print('Relaxed kernel',fold,t,len(results),'of47',flush=True)
    finally:
        for db in dbs.values(): db.close()
    out=Path(f'artifacts/native_reconstruction_relaxed_kernel_fold{fold}_20261003.json')
    report=dict(contract,targets=results,runtime_modified=False,promotion_allowed=False)
    if out.exists() and json.loads(out.read_text())!=report: raise ValueError('Changed final report')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    checked={}
    for fold in range(5): run(fold,checked)
