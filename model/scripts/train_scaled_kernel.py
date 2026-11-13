"""Single fit-only metric intervention on frozen all-target reconstruction."""
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.diagnose_kernel_metric import decompose
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_kernel_probe import (fit_map, map_query, fit_regressor, rotation_check,
                                       SETTINGS, SEED, ROOT, PRIVATE, DEST as OLD_DEST)
from scripts.train_nonlinear_placebo import donor_indices, verify_query
from scripts.train_nonlinear_probe import load_target
from scripts.train_observation_skip import check_masked_target
from whole_body_jepa.multisource_intervals import sha256

DEST = PRIVATE/'scaled_kernel_20261003'
ARMS = ('scaled_body','scaled_raw','scaled_process','scaled_shuffle')


def standardize_fit(x, weights):
    x, weights = np.asarray(x,dtype=float), np.asarray(weights,dtype=float)
    if (x.ndim!=2 or len(x)<2 or weights.shape!=(len(x),) or not np.isfinite(x).all()
            or not np.isfinite(weights).all() or not (weights>0).all()):
        raise ValueError('Invalid scaling inputs')
    scaler = StandardScaler().fit(x,sample_weight=weights)
    return scaler, scaler.transform(x)


def fit_heads(fitting, query, people, components, seed):
    weights = bounded_weights(components)
    donor = donor_indices(people,components,seed)
    models, predictions, checks = {}, {}, {}
    with threadpool_limits(limits=4):
        for source in ('body','raw','process'):
            key = 'body' if source=='body' else 'raw'
            x, q = [np.array(d[key],dtype=float,copy=True) for d in (fitting,query)]
            if source=='process': x[:,:106]=0; q[:,:106]=0
            scaler, z = standardize_fit(x,weights)
            qz = scaler.transform(q)
            if not np.isfinite(qz).all(): raise ValueError('Nonfinite standardized query')
            mapping, features = fit_map(z,weights,seed)
            qfeatures = map_query(qz,mapping)
            names = ('scaled_body','scaled_shuffle') if source=='body' else ('scaled_'+source,)
            for arm in names:
                train_x = features[donor] if arm=='scaled_shuffle' else features
                model, offset, diag = fit_regressor(train_x,fitting['truth'],weights)
                predictions[arm] = model.predict(qfeatures)+offset
                if not np.isfinite(predictions[arm]).all(): raise ValueError('Nonfinite predictions')
                constant = scaler.var_==0
                checks[arm] = dict(diag,fit_mae=float(np.mean(abs(model.predict(train_x)+offset-fitting['truth']))),
                    feature_width=features.shape[1],gamma=float(mapping['kernel'].gamma),
                    constant_features=int(constant.sum()),
                    query_only_varying_features=int(np.count_nonzero(np.any(q[:,constant]!=scaler.mean_[constant],axis=0))),
                    max_abs_standardized_query=float(np.max(abs(qz))),
                    min_scale=float(scaler.scale_.min()),max_scale=float(scaler.scale_.max()))
                if arm!='scaled_shuffle':
                    checks[arm]['fit_rotation']=rotation_check(z,features,mapping,model,offset,seed)
                    checks[arm]['query_rotation']=rotation_check(qz,qfeatures,mapping,model,offset,seed)
                if arm=='scaled_raw':
                    checks[arm]['fit_metric']=decompose(z,weights,seed,mapping['kernel'].gamma)
                models[arm]=dict(scaler=scaler,mapping=mapping,model=model,offset=offset)
    return models,predictions,checks,int(np.count_nonzero(donor==np.arange(len(donor))))


def run(fold,checked):
    dest=DEST/f'fold{fold}'; dest.mkdir(parents=True,exist_ok=True)
    pp=Path(f'artifacts/native_reconstruction_kernel_fold{fold}_20261003.json')
    proof=json.loads(pp.read_text()); verify_bound_files(proof['bindings'],checked)
    if proof['fold']!=fold or len(proof['targets'])!=47: raise ValueError('Wrong target family')
    paths=[Path(__file__),Path('scripts/score_scaled_kernel.py'),Path('scripts/diagnose_kernel_metric.py'),
           Path('RECONSTRUCTION_SCALED_KERNEL_PROTOCOL_20261003.md'),pp,*map(Path,proof['bindings'])]
    contract=dict(fold=fold,settings=SETTINGS,seed=SEED,scaling='fit_weighted_StandardScaler',
                  bindings={str(p):sha256(p) for p in paths})
    cp=dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract: raise ValueError('Changed resume contract')
    if not cp.exists(): cp.write_text(json.dumps(contract,indent=2)+'\n')
    roles=torch.load(ROOT/'partitions_20260922.pt',weights_only=False)['folds'][fold]
    variables=torch.load(ROOT/'scaling_20260922.pt',weights_only=False)['variables']
    links=pd.read_parquet(ROOT/'provenance.parquet')[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous provenance')
    components=dict(links.itertuples(index=False,name=None))
    dbs={r:sqlite3.connect((PRIVATE/f'fold{fold}_{r}.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
         for r in ('fit','selection')}
    results={}
    try:
        for target in sorted(proof['targets']):
            rp,mp,qp=[dest/f'{target}.{s}' for s in ('json','joblib','parquet')]
            if rp.exists():
                done=json.loads(rp.read_text())
                if (done['contract_sha256']!=sha256(cp) or done['model_sha256']!=sha256(mp)
                        or done['prediction_sha256']!=sha256(qp) or done['target']!=target or done['fold']!=fold):
                    raise ValueError('Changed saved target')
                results[target]=done; continue
            c=variables.index(target)
            fit_ids,fitting=load_target(dbs['fit'],c); ids,query=load_target(dbs['selection'],c)
            groups=[components[p] for p in fit_ids]
            if any(roles[p]!='fit' for p in fit_ids): raise ValueError('Incorrect fit role')
            prior_path=OLD_DEST/f'fold{fold}/{target}.parquet'
            if sha256(prior_path)!=proof['targets'][target]['prediction_sha256']: raise ValueError('Changed old predictions')
            paired=verify_query(ids,query,pd.read_parquet(prior_path),roles,'selection',fit_ids,groups)
            if not np.array_equal(paired.component,[components[p] for p in ids]): raise ValueError('Query provenance changed')
            for d in (fitting,query): check_masked_target(d['raw'],c)
            models,predictions,checks,singletons=fit_heads(fitting,query,fit_ids,groups,SEED+c)
            frame=paired.reset_index().assign(**predictions)
            tmp=mp.with_suffix('.tmp'); joblib.dump(models,tmp); tmp.replace(mp)
            tmp=qp.with_suffix('.tmp'); frame.to_parquet(tmp,index=False); tmp.replace(qp)
            done=dict(target=target,fold=fold,fit_patients=len(fit_ids),selection_patients=len(ids),
                      checks=checks,unchanged_singletons=singletons,contract_sha256=sha256(cp),
                      model_sha256=sha256(mp),prediction_sha256=sha256(qp))
            rp.write_text(json.dumps(done,indent=2,allow_nan=False)+'\n'); results[target]=done
            print('Scaled kernel',fold,target,len(results),'of47',flush=True)
    finally:
        for db in dbs.values(): db.close()
    out=Path(f'artifacts/native_reconstruction_scaled_kernel_fold{fold}_20261003.json')
    report=dict(contract,targets=results,runtime_modified=False,promotion_allowed=False)
    if out.exists() and json.loads(out.read_text())!=report: raise ValueError('Changed final report')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    checked={}
    for fold in range(5): run(fold,checked)
