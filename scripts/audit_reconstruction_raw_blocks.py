"""Fixed matched block removals, not a new deployed model or encoder repair."""
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.audit_reconstruction_stage_probe import ROOT, choose
from scripts.bounded_site_sampling import bounded_weights
from scripts.cache_nonlinear_probe import decode_features
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256

DEST = ROOT / 'raw_blocks_20261003'
NEW = ('latest_process', 'mean_process', 'values_presence')
ARMS = ('raw', *NEW, 'body', 'process')
CONTRASTS = (tuple(('raw', a) for a in NEW)
             + (('latest_process', 'mean_process'),)
             + tuple((a, 'body') for a in NEW))
PROTOCOL = Path('RECONSTRUCTION_RAW_BLOCK_PROTOCOL_20261003.md')


def block_inputs(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 265 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Invalid raw schema')
    result = {name: raw.copy() for name in ('raw', *NEW)}
    result['latest_process'][:, 53:106] = 0
    result['mean_process'][:, :53] = 0
    result['values_presence'][:, 159:] = 0
    return result


def assess(frame, seed=20261003, repetitions=5000):
    if (frame.empty or frame.duplicated(['patient', 'fold']).any()
            or frame.target.nunique() != 1
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired block predictions')
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    patients = errors.groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = patients.groupby(links.reindex(patients.index)).mean()
    comparisons = {}
    for i, (left, right) in enumerate(CONTRASTS):
        paired = dict(patient=paired_summary(patients[left]-patients[right], seed+i, repetitions),
                      component=paired_summary(groups[left]-groups[right], seed+100+i, repetitions))
        passed = len(patients) >= 30 and len(groups) >= 3 and all(
            s['ci95'] is not None and s['ci95'][1] < 0 and s['p_less'] is not None
            and s['p_less'] < .05/(47*7) for s in paired.values())
        comparisons[left+'_minus_'+right] = dict(paired=paired,screen_pass=bool(passed))
    return dict(query_occurrences=len(frame), patients=len(patients), components=len(groups),
                normalized_mae=dict(patient=patients.mean().to_dict(), component=groups.mean().to_dict()),
                comparisons=comparisons)


def load_sample(db, target, role, roles):
    eligible = [r[0] for r in db.execute('SELECT patient FROM features WHERE target=?', (target,))]
    ids = choose(eligible, 512 if role == 'fit' else 256)
    if not ids or any(roles[p] != role for p in ids):
        raise ValueError('Invalid sample roles')
    rows = [decode_features(*db.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?',
                                       (p, target)).fetchone()) for p in ids]
    return ids, np.stack([r['raw'] for r in rows]), np.array([float(r['truth']) for r in rows])


def verify_reference(frame, ids, truth, components, target, fold):
    if (frame.patient.tolist() != ids or not np.array_equal(frame.truth.to_numpy(), truth)
            or not (frame.target == target).all() or not (frame.fold == fold).all()
            or frame.component.tolist() != [components[p] for p in ids]
            or not np.isfinite(frame[['truth', 'hgb_raw', 'hgb_body', 'hgb_process']].to_numpy()).all()):
        raise ValueError('Changed E17 reference identity or values')


def fit_fold(fold):
    rp = Path(f'artifacts/native_reconstruction_stage_fold{fold}_20261003.json')
    report = json.loads(rp.read_text())
    old = report['contract']
    if (old['fold'] != fold or old['fit_limit'] != 512 or old['selection_limit'] != 256
            or old['parameters'] != PARAMETERS or len(report['targets']) != 47):
        raise ValueError('Changed E17 contract')
    fp, sp, pp = (ROOT / name for name in ('partitions_20260922.pt', 'scaling_20260922.pt', 'provenance.parquet'))
    bindings = {str(p): sha256(p) for p in (fp, sp, pp, rp, PROTOCOL, Path(__file__),
        Path('scripts/audit_reconstruction_stage_probe.py'), Path('scripts/train_nonlinear_probe.py'),
        Path('scripts/cache_nonlinear_probe.py'), Path('scripts/bounded_site_sampling.py'),
        Path('scripts/score_additional_reconstruction.py'), Path('whole_body_jepa/multisource_intervals.py'))}
    for p in (fp, sp, pp):
        if bindings[str(p)] != old['bindings'][str(p)]:
            raise ValueError('Changed original roles/scaling/provenance')
    roles = torch.load(fp, weights_only=False)['folds'][fold]
    variables = torch.load(sp, weights_only=False)['variables']
    links = pd.read_parquet(pp)[['uniquepid', 'component']].drop_duplicates()
    if links.uniquepid.duplicated().any():
        raise ValueError('Ambiguous component')
    components = dict(links.itertuples(index=False, name=None))
    dbs = {}
    try:
        for role in ('fit', 'selection'):
            p = ROOT / f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
            bindings[str(p)] = sha256(p)
            if bindings[str(p)] != old['bindings'][str(p)]:
                raise ValueError('Changed original feature cache')
            dbs[role] = sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True)
        target_sets = [{variables[r[0]] for r in db.execute('SELECT DISTINCT target FROM features')}
                       for db in dbs.values()]
        if any(s != set(report['targets']) for s in target_sets):
            raise ValueError('Incomplete original target family')
        contract = dict(fold=fold, bindings=bindings, parameters=PARAMETERS,
                        sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
                        sample_limits=[512,256], arms=list(ARMS), contrasts=list(map(list,CONTRASTS)))
        dest = DEST / f'fold{fold}'
        dest.mkdir(parents=True, exist_ok=True)
        cp = dest/'contract.json'
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError('Changed block contract')
        if not cp.exists():
            cp.write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
        frames, saved = [], {}
        for number, target in enumerate(report['targets'], 1):
            c = variables.index(target)
            fit_ids, raw_fit, y = load_sample(dbs['fit'], c, 'fit', roles)
            ids, raw_query, truth = load_sample(dbs['selection'], c, 'selection', roles)
            if set(fit_ids) & set(ids):
                raise ValueError('Overlapping patient roles')
            meta = report['targets'][target]
            ref_path = ROOT / f'stage_probe_20261003/fold{fold}/{target}.parquet'
            ref_contract = ref_path.parent/'contract.json'
            if (meta['prediction_sha256'] != sha256(ref_path)
                    or meta['contract_sha256'] != sha256(ref_contract)
                    or json.loads(ref_contract.read_text()) != old
                    or meta['fit_patients'] != len(fit_ids) or meta['selection_patients'] != len(ids)):
                raise ValueError('Changed E17 private reference')
            ref = pd.read_parquet(ref_path)
            verify_reference(ref, ids, truth, components, target, fold)
            pred_path, model_path, meta_path = (dest/f'{target}{ext}' for ext in ('.parquet','.joblib','.json'))
            if meta_path.exists():
                item = json.loads(meta_path.read_text())
                if (item['contract_sha256'] != sha256(cp) or item['prediction_sha256'] != sha256(pred_path)
                        or item['model_sha256'] != sha256(model_path)
                        or item['reference_sha256'] != sha256(ref_path)):
                    raise ValueError('Changed saved block artifacts')
                frame = pd.read_parquet(pred_path)
            else:
                fitting, query = block_inputs(raw_fit), block_inputs(raw_query)
                weights = bounded_weights([components[p] for p in fit_ids])
                models, predictions = {}, {}
                with threadpool_limits(limits=4):
                    for arm in ('raw', *NEW):
                        model = HistGradientBoostingRegressor(**PARAMETERS).fit(fitting[arm], y, sample_weight=weights)
                        prediction = model.predict(query[arm])
                        if model.n_iter_ != 100 or not np.isfinite(prediction).all():
                            raise ValueError('Incomplete or nonfinite block fit')
                        models[arm], predictions[arm] = model, prediction
                if not np.allclose(predictions['raw'], ref.hgb_raw.to_numpy(), atol=1e-12, rtol=0):
                    raise ValueError('Original raw prediction did not reproduce')
                frame = ref[['patient','component','target','fold','truth']].copy()
                for arm, values in predictions.items(): frame[arm] = values
                frame['body'], frame['process'] = ref.hgb_body.to_numpy(), ref.hgb_process.to_numpy()
                frame.to_parquet(pred_path, index=False)
                joblib.dump(models, model_path)
                item = dict(target=target, fold=fold, fit_patients=len(fit_ids), selection_patients=len(ids),
                    contract_sha256=sha256(cp), prediction_sha256=sha256(pred_path),
                    model_sha256=sha256(model_path), reference_sha256=sha256(ref_path),
                    fit_normalized_mae={a:float(np.mean(np.abs(m.predict(fitting[a])-y))) for a,m in models.items()})
                meta_path.write_text(json.dumps(item, indent=2, allow_nan=False)+'\n')
            if (frame.patient.tolist() != ids or not np.array_equal(frame.truth.to_numpy(),truth)
                    or not (frame.target==target).all() or not (frame.fold==fold).all()
                    or frame.component.tolist()!=[components[p] for p in ids]
                    or not np.allclose(frame.raw,ref.hgb_raw,rtol=0,atol=1e-12)
                    or not np.array_equal(frame.body,ref.hgb_body)
                    or not np.array_equal(frame.process,ref.hgb_process)
                    or not np.isfinite(frame[list(ARMS)].to_numpy()).all()):
                raise ValueError('Invalid saved paired frame')
            frames.append(frame); saved[target] = item
            print('Raw block diagnostic',fold,number,'/47',target,flush=True)
        out = Path(f'artifacts/native_reconstruction_raw_blocks_fold{fold}_20261003.json')
        out.write_text(json.dumps(dict(contract=contract,targets=saved,runtime_modified=False,
                                      promotion_allowed=False),indent=2,allow_nan=False)+'\n')
        return frames, {str(out):sha256(out), str(cp):sha256(cp)}
    finally:
        for db in dbs.values(): db.close()


def main():
    output = Path('artifacts/native_reconstruction_raw_blocks_assessment_20261003.json')
    if output.exists(): raise ValueError('Existing completed block assessment')
    frames, bindings = [], {}
    for fold in range(5):
        rows, verified = fit_fold(fold)
        frames.extend(rows); bindings.update(verified)
    data = pd.concat(frames, ignore_index=True)
    if data.target.nunique()!=47 or len(frames)!=235:
        raise ValueError('Incomplete block family')
    results = {t:assess(data[data.target==t],20261003+i*1000) for i,t in enumerate(sorted(data.target.unique()))}
    summary = dict(query_occurrences=len(data),patients=int(data.patient.nunique()),
        normalized_target_macro={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
                                       for a in ARMS} for level in ('patient','component')},
        screens={a+'_minus_'+b:[t for t,r in results.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
                 for a,b in CONTRASTS})
    output.write_text(json.dumps(dict(summary=summary,targets=results,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        limitation='Same E17 bounded reused development patients. Refitted raw block removals, fixed HGB. Not encoder-loss proof, forecast, causal or release validation.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
