"""Source-component-held-out complexity selection for fixed boosted readouts."""

import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from scripts.diagnose_nonlinear_fit_gap import summarize
from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_nonlinear_probe import load_target, PARAMETERS
from whole_body_jepa.multisource_intervals import sha256


STAGES = (1, 5, 10, 25, 50, 100)
HEADS = ('body_probe', 'raw_probe', 'process_summary', 'trained_shuffle')
ARMS = ('native', 'population', *HEADS, *(a+'_prefix' for a in HEADS))
REFERENCES = ('native', 'body_probe', 'population', 'process_summary_prefix', 'trained_shuffle_prefix')


def stage_predictions(model, features):
    if model.n_iter_ != STAGES[-1] or not np.isfinite(features).all():
        raise ValueError('Invalid staged model/input')
    selected = []
    with threadpool_limits(limits=4):
        for step, prediction in enumerate(model.staged_predict(features), 1):
            if step in STAGES:
                selected.append(np.asarray(prediction, float).copy())
    values = np.column_stack(selected)
    if step != STAGES[-1] or values.shape != (len(features), len(STAGES)) or not np.isfinite(values).all():
        raise ValueError('Incomplete/nonfinite stage predictions')
    return values


def crossfit_stage(truth, predictions, components):
    y, p, groups = np.asarray(truth, float), np.asarray(predictions, float), np.asarray(components)
    if (y.ndim != 1 or not len(y) or groups.shape != y.shape
            or p.shape != (len(y), len(STAGES)) or not np.isfinite(y).all()
            or not np.isfinite(p).all() or pd.isna(groups).any()):
        raise ValueError('Invalid stage selection support')
    index = np.full(len(y), len(STAGES)-1, dtype=int)
    supported = np.zeros(len(y), dtype=bool)
    records = []
    for group in sorted(set(groups)):
        held = groups == group
        train = ~held
        entry = dict(component=str(group), query_patients=int(held.sum()),
                     training_patients=int(train.sum()), training_components=len(set(groups[train])))
        if train.sum() < 30 or entry['training_components'] < 2:
            records.append(dict(**entry, status='insufficient_training_support', stage=None))
            continue
        # Construct losses from training outcomes only, not held-component labels.
        error = pd.DataFrame(abs(p[train]-y[train, None]))
        scores = error.groupby(groups[train]).mean().mean().to_numpy()
        chosen = int(np.argmin(scores))
        index[held], supported[held] = chosen, True
        records.append(dict(**entry, status='complete', stage=STAGES[chosen],
                            training_component_mae=scores.tolist()))
    return p[np.arange(len(y)), index], np.asarray(STAGES)[index], supported, records


def curve(truth, native, population, staged, groups):
    result = {}
    for i, stage in enumerate(STAGES):
        row = summarize(truth, dict(native=native, population=population, body_probe=staged[:, i]), groups)
        result[str(stage)] = {k: row[k] for k in ('patients', 'components', 'normalized_mae', 'bands')}
    return result


def assess(frame, seed, repetitions=2000):
    if (frame.empty or frame.target.nunique()!=1 or not frame.role.eq('selection').all()
            or frame.duplicated(['patient','fold']).any()
            or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth',*ARMS]]).all().all()):
        raise ValueError('Invalid paired source predictions')
    loss = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    loss['patient'] = frame.patient.to_numpy()
    patient = loss.groupby('patient')[list(ARMS)].mean()
    patient['component'] = frame.drop_duplicates('patient').set_index('patient').component.reindex(patient.index)
    component = patient.groupby('component')[list(ARMS)].mean()
    contrasts = {}
    for i, reference in enumerate(REFERENCES):
        paired = dict(patient=paired_summary(patient.body_probe_prefix-patient[reference],seed+i,repetitions),
            component=paired_summary(component.body_probe_prefix-component[reference],seed+100+i,repetitions))
        passes = bool(frame.supported.all() and len(patient)>=30 and len(component)>=3 and all(
            s['mean']<0 and s['p_less'] is not None and s['p_less']<.05/(47*len(REFERENCES))
            and s['ci95'][1]<0 for s in paired.values()))
        contrasts[reference] = dict(paired=paired, diagnostic_family_screen=passes)
    stages = {}
    supported = frame.loc[frame.supported]
    for arm in HEADS:
        stages[arm] = None if supported.empty else {
            str(stage):float((supported[arm+'_stage']==stage).groupby(supported.patient).mean().mean())
            for stage in STAGES}
    return dict(patients=len(patient),occurrences=len(frame),components=len(component),
        unsupported_occurrences=int((~frame.supported).sum()),
        normalized_mae=dict(patient=patient[list(ARMS)].mean().to_dict(),component=component.mean().to_dict()),
        comparisons=contrasts, stage_patient_fractions=stages,
        per_component=[dict(component=str(c),patients=int((patient.component==c).sum()),normalized_mae=r.to_dict())
                       for c,r in component.iterrows()])


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    features = root/'nonlinear_probe_20261001'
    dest = features/'prefix_20261003'
    out = Path('artifacts/native_reconstruction_nonlinear_prefix_20261003.json')
    if out.exists(): raise ValueError('Completed prefix intervention exists')
    dest.mkdir(exist_ok=True)
    gap_path = Path('artifacts/native_reconstruction_nonlinear_fit_gap_20261003.json')
    source_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    gap, source = [json.loads(p.read_text()) for p in (gap_path,source_path)]
    checked = {}
    verify_bound_files(gap['bindings'],checked)
    verify_bound_files(source['bindings'],checked)
    plan_path, scale_path, prov_path = root/'partitions_20260922.pt',root/'scaling_20260922.pt',root/'provenance.parquet'
    plan = torch.load(plan_path,weights_only=False)['folds']
    scaling = torch.load(scale_path,weights_only=False)
    links = pd.read_parquet(prov_path)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous component')
    mapping = dict(links.itertuples(index=False,name=None))
    targets = sorted(source['targets'])
    if len(targets)!=47: raise ValueError('Changed target family')
    for roles in plan:
        sets = {}
        for p,r in roles.items(): sets.setdefault(mapping[p],set()).add(r)
        if any(len(r)!=1 for r in sets.values()): raise ValueError('Component role overlap')
    paths = [Path(__file__),Path('RECONSTRUCTION_PREFIX_COMPLEXITY_PROTOCOL_20261003.md'),gap_path,source_path,
             plan_path,scale_path,prov_path,Path('scripts/diagnose_nonlinear_fit_gap.py'),
             Path('scripts/train_nonlinear_probe.py'),Path('scripts/score_additional_reconstruction.py')]
    for fold in range(5):
        report_path = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_selection_20261001.json')
        report = json.loads(report_path.read_text())
        verify_bound_files(report['bindings'],checked)
        cache = features/f'fold{fold}_selection.sqlite'
        if report['fold']!=fold or report['role']!='selection' or sha256(cache)!=report['private_sha256']:
            raise ValueError('Changed source feature cache')
        paths.extend((report_path,cache))
    bindings = {str(p):sha256(p) for p in paths}
    cp = dest/'contract.json'
    if cp.exists() and json.loads(cp.read_text())!=bindings: raise ValueError('Changed prefix resume contract')
    if not cp.exists(): cp.write_text(json.dumps(bindings,indent=2)+'\n')
    for fold in range(5):
        dbs = {role:sqlite3.connect((features/f'fold{fold}_{role}.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
               for role in ('fit','selection')}
        proof_path = Path(f'artifacts/native_reconstruction_nonlinear_placebo_fold{fold}_20261003.json')
        null_proof = json.loads(proof_path.read_text())
        verify_bound_files(null_proof['bindings'],checked)
        for k,target in enumerate(targets):
            saved_path, pred_path = dest/f'fold{fold}_{target}.json',dest/f'fold{fold}_{target}.parquet'
            if saved_path.exists():
                done = json.loads(saved_path.read_text())
                if done['contract_sha256']!=sha256(cp) or done['prediction_sha256']!=sha256(pred_path):
                    raise ValueError('Changed completed prefix result')
                continue
            c = scaling['variables'].index(target)
            ids, data = load_target(dbs['selection'],c)
            fit_ids, fitting = load_target(dbs['fit'],c)
            if (set(ids)&set(fit_ids) or any(plan[fold][p]!='selection' for p in ids)
                    or any(plan[fold][p]!='fit' for p in fit_ids)):
                raise ValueError('Input role leakage')
            mp = features/f'fold{fold}_probes/{target}.joblib'
            npth = features/f'placebo_20261003/fold{fold}/{target}.joblib'
            qp = npth.with_suffix('.parquet')
            if (sha256(mp)!=source['bindings'][str(mp)] or sha256(npth)!=null_proof['targets'][target]['model_sha256']
                    or sha256(qp)!=null_proof['targets'][target]['prediction_sha256']):
                raise ValueError('Changed fitted heads or predictions')
            prior = pd.read_parquet(qp,filters=[('role','==','selection')])
            if prior.patient.duplicated().any() or set(prior.patient)!=set(ids): raise ValueError('Changed query support')
            prior = prior.set_index('patient').loc[ids].reset_index()
            if (not prior.fold.eq(fold).all() or not prior.target.eq(target).all()
                    or not np.array_equal(prior.component,prior.patient.map(mapping))
                    or not np.allclose(prior.truth,data['truth'],atol=1e-10,rtol=1e-10)
                    or not np.allclose(prior.native,data['prediction'],atol=1e-10,rtol=1e-10)):
                raise ValueError('Changed labels, role or native predictions')
            bundle = joblib.load(mp)
            models = {**bundle['models'],'trained_shuffle':joblib.load(npth)}
            if (bundle['target']!=target or bundle['parameters']!=PARAMETERS
                    or any(any(m.get_params()[key]!=v for key,v in PARAMETERS.items()) for m in models.values())):
                raise ValueError('Changed readout settings')
            process = data['raw'].copy(); process[:,:106] = 0
            inputs = dict(body_probe=data['body'],raw_probe=data['raw'],process_summary=process,trained_shuffle=data['body'])
            result = prior[['patient','component','target','fold','role','truth','native','population',*HEADS]].copy()
            result['supported'] = True
            decisions = {}
            body_stages = None
            for arm in HEADS:
                pred = stage_predictions(models[arm],inputs[arm])
                if not np.allclose(pred[:,-1],prior[arm],atol=1e-10,rtol=1e-10):
                    raise ValueError('Stage100 does not reproduce original prediction')
                point, stage, supported, records = crossfit_stage(data['truth'],pred,prior.component)
                result[arm+'_prefix'], result[arm+'_stage'] = point, stage
                result['supported'] &= supported
                decisions[arm] = records
                if arm=='body_probe': body_stages = pred
            fit_stages = stage_predictions(models['body_probe'],fitting['body'])
            fit_groups = np.asarray([mapping[p] for p in fit_ids])
            population = prior.population.iloc[0]
            if not np.allclose(prior.population,population,rtol=1e-10,atol=1e-10): raise ValueError('Nonconstant baseline')
            fit_curve = curve(fitting['truth'],fitting['prediction'],np.full(len(fit_ids),population),fit_stages,fit_groups)
            expected = gap['folds'][str(fold)][target]['fit']
            if expected['patients']!=len(fit_ids): raise ValueError('Incomplete fitting curve')
            for level in ('patient','component','bounded'):
                for arm in ('native','population','body_probe'):
                    if not np.isclose(fit_curve['100']['normalized_mae'][level][arm],expected['normalized_mae'][level][arm],atol=1e-10,rtol=1e-10):
                        raise ValueError('Fit endpoint does not reproduce diagnostic')
            query_curve = curve(data['truth'],prior.native.to_numpy(),prior.population.to_numpy(),body_stages,prior.component)
            tmp = pred_path.with_suffix('.tmp'); result.to_parquet(tmp,index=False); tmp.replace(pred_path)
            saved = dict(fold=fold,target=target,fit_curve=fit_curve,selection_curve=query_curve,decisions=decisions,
                contract_sha256=sha256(cp),prediction_sha256=sha256(pred_path))
            saved_path.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
            print('Completed prefix curves',fold,target,k+1,'of47',flush=True)
        for db in dbs.values(): db.close()
    results, curves = {}, {}
    for i,target in enumerate(targets):
        frames, rows = [], {}
        for fold in range(5):
            sp,qp = dest/f'fold{fold}_{target}.json',dest/f'fold{fold}_{target}.parquet'
            row = json.loads(sp.read_text())
            if row['contract_sha256']!=sha256(cp) or row['prediction_sha256']!=sha256(qp): raise ValueError('Changed saved pair')
            frames.append(pd.read_parquet(qp)); rows[str(fold)] = row
        result = assess(pd.concat(frames,ignore_index=True),20261003+1000*i)
        for level in ('patient','component'):
            for arm in ('native','body_probe','raw_probe','process_summary','population'):
                if not np.isclose(result['normalized_mae'][level][arm],source['targets'][target]['normalized_mae'][level][arm],atol=1e-10,rtol=1e-10):
                    raise ValueError('Original pooled errors changed')
        results[target],curves[target] = result,rows
        print('Assessed prefix target',i+1,'of47',flush=True)
    summary = dict(full_five_contrast_screen=[t for t,r in results.items() if all(c['diagnostic_family_screen'] for c in r['comparisons'].values())],
        macro_normalized_mae={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()])) for a in ARMS} for level in ('patient','component')})
    out.write_text(json.dumps(dict(summary=summary,targets=results,curves=curves,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='All47/five frozen readouts; nested source component-held-out prefix selection. Reused development, overlapping models/patients; conditional diagnostics not fresh external evidence or promotion. Bootstrap does not include refit/selection uncertainty. Truth bands are retrospective only. No encoder/model-weight/runtime/calibration change.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
