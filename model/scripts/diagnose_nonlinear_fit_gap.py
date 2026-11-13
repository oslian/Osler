"""Evaluate fixed nonlinear heads on their own complete fit inputs."""

import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from scripts.bounded_site_sampling import bounded_weights
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_placebo import donor_indices, SEED
from scripts.train_nonlinear_probe import load_target, PARAMETERS
from whole_body_jepa.multisource_intervals import sha256


ARMS = ('native','body_probe','raw_probe','process_summary','population','trained_shuffle')


def original_population(raw_centers, location, scale, cached_truth):
    raw_centers = np.asarray(raw_centers,float)
    if (not np.isfinite(raw_centers).all() or not np.isfinite([location,scale]).all()
            or scale<=0 or np.asarray(cached_truth).shape!=raw_centers.shape):
        raise ValueError('Invalid baseline centers')
    # Native labels passed through float32 before the affine transform; the
    # baseline median was computed directly from the original float64 centers.
    expected = (raw_centers.astype(np.float32).astype(float)-location)/scale
    if not np.allclose(expected,cached_truth,atol=1e-10,rtol=1e-10):
        raise ValueError('Cached fitting labels do not reproduce float conversion')
    exact = (raw_centers-location)/scale
    return float(np.median(exact)),float(np.max(abs(exact-cached_truth)))


def weighted_median(values, weights):
    values, weights = np.asarray(values,float), np.asarray(weights,float)
    if (values.ndim!=1 or weights.shape!=values.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<=0).any()):
        raise ValueError('Invalid weighted sample')
    order = np.argsort(values,kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative,cumulative[-1]/2,side='left')]])


def summarize(truth, predictions, components):
    truth, components = np.asarray(truth,float), np.asarray(components)
    if (truth.ndim!=1 or not len(truth) or components.shape!=truth.shape
            or not np.isfinite(truth).all()):
        raise ValueError('Invalid paired support')
    if any(np.asarray(p).shape!=truth.shape or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Invalid predictions')
    weights = bounded_weights(components)
    errors = pd.DataFrame({a:abs(np.asarray(p)-truth) for a,p in predictions.items()})
    grouped = errors.groupby(components).mean()
    bands = {}
    distance = abs(truth)
    for name, keep in (('central',distance<=1),('intermediate',(distance>1)&(distance<=3)),('tail',distance>3)):
        bands[name] = dict(patients=int(keep.sum()),
            normalized_mae={a:float(errors.loc[keep,a].mean()) if keep.any() else None for a in predictions},
            body_minus_native_contribution=float(((errors.body_probe-errors.native)*keep).mean()))
    return dict(patients=len(truth),components=len(grouped),
        normalized_mae=dict(patient=errors.mean().to_dict(),component=grouped.mean().to_dict(),
            bounded={a:float(np.average(errors[a],weights=weights)) for a in predictions}),
        mean_signed_error={a:float(np.mean(np.asarray(p)-truth)) for a,p in predictions.items()},
        bands=bands,per_component=[dict(component=str(c),patients=int(sum(components==c)),normalized_mae=row.to_dict()) for c,row in grouped.iterrows()])


def classify(fit, selection, reference, level):
    delta_fit = fit['normalized_mae'][level]['body_probe']-fit['normalized_mae'][level][reference]
    delta_query = selection['normalized_mae'][level]['body_probe']-selection['normalized_mae'][level][reference]
    return dict(fit_delta=delta_fit,selection_delta=delta_query,
        pattern=('benefit' if delta_fit<0 else 'nonbenefit')+'_to_'+('benefit' if delta_query<0 else 'nonbenefit'))


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    features = root/'nonlinear_probe_20261001'
    dest = features/'fit_gap_20261003_v2'
    out = Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    if out.exists(): raise ValueError('Completed fit diagnostic exists')
    dest.mkdir(exist_ok=True)
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    source = json.loads(source_path.read_text())
    checked = {}
    verify_bound_files(source['bindings'],checked)
    plan_path,scale_path,prov_path = root/'partitions_20260922.pt',root/'scaling_20260922.pt',root/'provenance.parquet'
    plan = torch.load(plan_path,weights_only=False)['folds']
    scaling = torch.load(scale_path,weights_only=False)
    variables = scaling['variables']
    person_index = {p:i for i,p in enumerate(scaling['people'])}
    links = pd.read_parquet(prov_path)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous patient provenance')
    mapping = dict(links.itertuples(index=False,name=None))
    paths = [Path(__file__),Path('RECONSTRUCTION_NONLINEAR_FIT_GAP_PROTOCOL_20261003.md'),source_path,
        plan_path,scale_path,prov_path,Path('scripts/train_nonlinear_probe.py'),Path('scripts/train_nonlinear_placebo.py'),
        Path('scripts/cache_nonlinear_probe.py'),Path('scripts/bounded_site_sampling.py')]
    proofs = {}
    for fold in range(5):
        rp = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_fit_20261001.json')
        pp = Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
        rp_report, pp_report = json.loads(rp.read_text()),json.loads(pp.read_text())
        if rp_report['role']!='fit' or rp_report['fold']!=fold or pp_report['fold']!=fold:
            raise ValueError('Wrong fold proof')
        verify_bound_files(rp_report['bindings'],checked)
        verify_bound_files(pp_report['bindings'],checked)
        cache = features/f'fold{fold}_fit.sqlite'
        if sha256(cache)!=rp_report['private_sha256']: raise ValueError('Changed fit cache')
        proofs[fold] = (rp_report,pp_report)
        paths.extend((rp,pp,cache))
    bindings = {str(p):sha256(p) for p in paths}
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=bindings: raise ValueError('Changed diagnostic contract')
    if not cp.exists(): cp.write_text(json.dumps(bindings,indent=2)+'\n')
    targets = sorted(source['targets'])
    if len(targets)!=47: raise ValueError('Changed target family')
    folds = {}
    for fold in range(5):
        results = {}
        db = sqlite3.connect((features/f'fold{fold}_fit.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for target in targets:
                saved_path = dest/f'fold{fold}_{target}.json'
                if saved_path.exists():
                    result = json.loads(saved_path.read_text())
                    if result['contract_sha256']!=sha256(cp): raise ValueError('Changed partial result')
                    results[target] = result
                    continue
                c = variables.index(target)
                ids, data = load_target(db,c)
                if any(plan[fold][p]!='fit' for p in ids): raise ValueError('Nonfit input')
                groups = np.array([mapping[p] for p in ids])
                model_path = features/f'fold{fold}_probes/{target}.joblib'
                if sha256(model_path)!=source['bindings'][str(model_path)]: raise ValueError('Changed fitted heads')
                bundle = joblib.load(model_path)
                if bundle['parameters']!=PARAMETERS or bundle['target']!=target: raise ValueError('Wrong head settings')
                null_path = features/f'placebo_20261003/fold{fold}/{target}.joblib'
                query_path = null_path.with_suffix('.parquet')
                null_meta = proofs[fold][1]['targets'][target]
                if sha256(null_path)!=null_meta['model_sha256'] or sha256(query_path)!=null_meta['prediction_sha256']:
                    raise ValueError('Changed matched control')
                query = pd.read_parquet(query_path,filters=[('role','==','selection')])
                if (query.patient.duplicated().any() or set(ids)&set(query.patient) or set(groups)&set(query.component)
                        or any(plan[fold][p]!='selection' for p in query.patient)
                        or not np.array_equal(query.component,query.patient.map(mapping))):
                    raise ValueError('Role/component overlap or changed query')
                if len(ids)!=null_meta['fit_patients']: raise ValueError('Incomplete fitting support')
                scaler = scaling['scalers'][fold]
                centers = scaling['centers'][[person_index[p] for p in ids],c]
                population, precision_gap = original_population(centers,scaler['location'][c],scaler['scale'][c],data['truth'])
                if not np.allclose(query.population,population,atol=1e-10,rtol=1e-10):
                    raise ValueError('Original population median differs')
                process = data['raw'].copy(); process[:,:106] = 0
                predictions = dict(native=data['prediction'],population=np.full(len(ids),population))
                null = joblib.load(null_path)
                models = [*bundle['models'].values(),null]
                if any(m.n_iter_!=100 or any(m.get_params()[k]!=v for k,v in PARAMETERS.items()) for m in models):
                    raise ValueError('Wrong model capacity')
                donor = donor_indices(ids,groups,SEED+c)
                with threadpool_limits(limits=4):
                    for arm,values in (('body_probe',data['body']),('raw_probe',data['raw']),('process_summary',process)):
                        predictions[arm] = bundle['models'][arm].predict(values)
                    predictions['trained_shuffle'] = null.predict(data['body'])
                    predictions['shuffle_training_assignment'] = null.predict(data['body'][donor])
                median = weighted_median(data['truth'],bounded_weights(groups))
                predictions['weighted_median'] = np.full(len(ids),median)
                fit = summarize(data['truth'],predictions,groups)
                selection = summarize(query.truth,{a:query[a].to_numpy() for a in ARMS},query.component)
                expected = source['folds'][str(fold)][target]
                for level in ('patient','component'):
                    for arm in ARMS[:-1]:
                        if not np.isclose(selection['normalized_mae'][level][arm],expected['normalized_mae'][level][arm],atol=1e-10,rtol=1e-10):
                            raise ValueError('Original source metrics changed')
                result = dict(fit=fit,selection=selection,fit_label_float_conversion_max_abs=precision_gap,
                    contrasts={level:{ref:classify(fit,selection,ref,level) for ref in ('native','population','process_summary','trained_shuffle')} for level in ('patient','component','bounded')},
                    contract_sha256=sha256(cp),head_sha256=sha256(model_path),null_sha256=sha256(null_path))
                saved_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                results[target] = result
                print('Fit diagnostic',fold,target,len(results),'of47',flush=True)
            if sum(r['fit']['patients'] for r in results.values())!=proofs[fold][0]['examples']:
                raise ValueError('Incomplete fitting grid')
            folds[str(fold)] = results
        finally:
            db.close()
    summary = {}
    for level in ('patient','component','bounded'):
        summary[level] = {}
        for reference in ('native','population','process_summary','trained_shuffle'):
            counts = {}
            for rows in folds.values():
                for r in rows.values():
                    name = r['contrasts'][level][reference]['pattern']
                    counts[name] = counts.get(name,0)+1
            summary[level][reference] = counts
    out.write_text(json.dumps(dict(folds=folds,summary=summary,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='All47/five frozen heads. Fit errors are optimistic diagnostics, not validation. Source-selection outcomes already reused. Fits/queries overlap across folds; do not treat235cells as independent tests. Contrasts within roles; absolute cross-role gaps confound case mix. Shuffled actual training assignment separated from own-state inference. No fitting, outer evaluation, causal, forecast or promotion claim.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
