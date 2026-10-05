"""Observed-value accessibility at last-observed event versus final body state."""
import hashlib
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.audit_reconstruction_stage_probe import ROOT
from scripts.audit_reconstruction_raw_state import summarize
from scripts.bounded_site_sampling import bounded_weights
from scripts.cache_nonlinear_probe import decode_features
from scripts.encoder_fold_scaling import transform_example
from scripts.score_additional_reconstruction import paired_summary
from scripts.train_encoder_crossfit import fresh_encoder
from scripts.train_nonlinear_probe import PARAMETERS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache

DEST = ROOT/'observed_content_20261003'
PROTOCOL = Path('RECONSTRUCTION_OBSERVED_CONTENT_PROTOCOL_20261003.md')
ARMS = tuple(f'{r}_{s}' for r in ('ridge','hgb') for s in ('event','body','process')) + ('median',)
CONTRASTS = tuple((f'{r}_{a}',f'{r}_{b}') for r in ('ridge','hgb')
                  for a,b in (('event','body'),('event','process'),('body','process')))


def digest(value):
    return hashlib.sha256(('observed-content-20261003:'+value).encode()).hexdigest()


def contexts(keys, limit):
    if limit < 1 or not keys or len(set(keys)) != len(keys):
        raise ValueError('Invalid context population')
    by_patient = {}
    for p,c in keys:
        by_patient.setdefault(p,[]).append(c)
    people = sorted(by_patient,key=lambda p:(digest(str(p)),str(p)))[:limit]
    return [(p,min(by_patient[p],key=lambda c:(digest(str(p)+'/'+str(c)),c))) for p in people]


def last_observed(tokens,times,anchor,n):
    if (tokens.ndim!=2 or tokens.shape[1]<2*n or times.shape!=(len(tokens),)
            or not len(times) or not torch.isfinite(times).all() or not np.isfinite(anchor)
            or (times>anchor).any() or (times[1:]<times[:-1]).any()):
        raise ValueError('Invalid event history')
    mask = tokens[:,n:2*n].cpu().numpy()
    if not np.isin(mask,[0,1]).all(): raise ValueError('Nonbinary observation mask')
    present = mask.any(0)
    positions = np.full(n,-1,dtype=int)
    for c in np.flatnonzero(present): positions[c]=np.flatnonzero(mask[:,c])[-1]
    return present,positions


def assess(frame,seed,repetitions=5000):
    if (frame.empty or frame.duplicated(['patient','fold']).any() or frame.variable.nunique()!=1
            or frame.groupby('patient').component.nunique().max()!=1
            or not np.isfinite(frame[['truth',*ARMS]].to_numpy()).all()):
        raise ValueError('Invalid observed-content pairs')
    error = frame[list(ARMS)].subtract(frame.truth,axis=0).abs()
    patients = error.groupby(frame.patient).mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    groups = patients.groupby(links.reindex(patients.index)).mean()
    comparisons = {}
    for i,(a,b) in enumerate(CONTRASTS):
        stats = dict(patient=paired_summary(patients[a]-patients[b],seed+i,repetitions),
                     component=paired_summary(groups[a]-groups[b],seed+100+i,repetitions))
        passed = len(patients)>=30 and len(groups)>=3 and all(s['ci95'] is not None
            and s['ci95'][1]<0 and s['p_less'] is not None and s['p_less']<.05/(53*6) for s in stats.values())
        comparisons[a+'_minus_'+b]=dict(paired=stats,screen_pass=bool(passed))
    return dict(patients=len(patients),components=len(groups),query_occurrences=len(frame),
                folds=sorted(map(int,frame.fold.unique())),
                normalized_mae=dict(patient=patients.mean().to_dict(),component=groups.mean().to_dict()),
                comparisons=comparisons)


def extract(fold):
    fp,sp,pp,hp = [ROOT/name for name in ('partitions_20260922.pt','scaling_20260922.pt',
                                       'provenance.parquet','full_histories_20260922.sqlite')]
    schema_path=Path('models/whole_body_joint_jepa.pt')
    sf=0 if fold==2 else fold
    wp=ROOT/f'monitored_continuation_20260929/fold{sf}_real_epoch10.pt'
    paths=[Path(__file__),PROTOCOL,fp,sp,pp,hp,schema_path,wp,
           Path('scripts/encoder_fold_scaling.py'),Path('scripts/train_encoder_crossfit.py'),
           Path('scripts/audit_reconstruction_raw_state.py'),Path('scripts/train_nonlinear_probe.py'),
           Path('scripts/cache_nonlinear_probe.py'),Path('scripts/bounded_site_sampling.py'),
           Path('scripts/score_additional_reconstruction.py'),Path('whole_body_jepa/native_body_reconstruction.py'),
           Path('whole_body_jepa/native_organ_observations.py'),Path('whole_body_jepa/native_organ_state_residual.py'),
           Path('whole_body_jepa/physiology_state.py'),Path('whole_body_joint_jepa.py')]
    bindings={str(p):sha256(p) for p in paths}
    for role in ('fit','selection'):
        rp=Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_{role}_20261001.json')
        r=json.loads(rp.read_text())
        if (r['fold'],r['role'],r['targets'])!=(fold,role,47): raise ValueError('Changed cache scope')
        for p in (fp,sp,hp,wp):
            if bindings[str(p)]!=r['bindings'][str(p)]: raise ValueError('Changed original dependency')
        cp=ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
        bindings[str(cp)]=sha256(cp)
        if bindings[str(cp)]!=r['private_sha256']: raise ValueError('Changed features')
        bindings[str(rp)]=sha256(rp)
    plan=torch.load(fp,weights_only=False)['folds'][fold]
    scaling=torch.load(sp,weights_only=False)
    variables=scaling['variables']
    links=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any(): raise ValueError('Ambiguous patient provenance')
    components=dict(links.itertuples(index=False,name=None))
    dest=DEST/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'contract.json'
    contract=dict(fold=fold,variables=variables,bindings=bindings,parameters=PARAMETERS,
                  sklearn=sklearn.__version__,numpy=np.__version__,ridge_alpha=100.,limits=[2048,1024])
    if cp.exists() and json.loads(cp.read_text())!=contract: raise ValueError('Changed extraction contract')
    if not cp.exists(): cp.write_text(json.dumps(contract,indent=2,allow_nan=False)+'\n')
    schema=torch.load(schema_path,weights_only=False)
    ck=torch.load(wp,weights_only=False)
    if variables!=schema['variables'] or len(variables)!=53 or (ck['fold'],ck['epoch'],ck['arm'])!=(sf,10,'real'):
        raise ValueError('Wrong frozen encoder')
    model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
    model.load_state_dict(ck['state_dict'],strict=True)
    source=ReconstructionCache(hp)
    data={};metadata={}
    try:
        for role,limit in (('fit',2048),('selection',1024)):
            cache=ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
            db=sqlite3.connect(cache.resolve().as_uri()+'?mode=ro',uri=True)
            try:
                selected=contexts(db.execute('SELECT patient,target FROM features').fetchall(),limit)
                if any(plan[p]!=role for p,c in selected): raise ValueError('Wrong role')
                saved_path,mp=dest/f'{role}.joblib',dest/f'{role}.json'
                if mp.exists():
                    m=json.loads(mp.read_text())
                    if m['sha256']!=sha256(saved_path) or m['contract_sha256']!=sha256(cp):
                        raise ValueError('Changed content cache')
                    d=joblib.load(saved_path)
                    if list(zip(d['patient'],map(int,d['context'])))!=selected:
                        raise ValueError('Changed sampled contexts')
                else:
                    rows=[]
                    for i,(p,c) in enumerate(selected):
                        old=decode_features(*db.execute('SELECT payload,digest FROM features WHERE patient=? AND target=?',(p,c)).fetchone())
                        e=transform_example(source.get(p,c),scaling['scalers'][fold])
                        raw,_=summarize(e['tokens'],e['event_hours'],e['anchor_hour'],variables,variables[c])
                        if not np.array_equal(raw,old['raw']): raise ValueError('Raw summary mismatch')
                        clean,times=mask_native_target(e['tokens'],e['event_hours'],variables,variables[c])
                        present,positions=last_observed(clean,times,e['anchor_hour'],53)
                        if not np.array_equal(present,raw[106:159].astype(bool)):
                            raise ValueError('Mask support mismatch')
                        events=np.zeros((53,64),dtype=np.float32)
                        unique=np.unique(positions[present])
                        with torch.inference_mode():
                            if len(unique):
                                encoded,_,_=encode_native_organ_observations(model.trajectory.jepa,variables,clean[unique])
                                values=encoded.cpu().numpy()
                                for v in np.flatnonzero(present):
                                    events[v]=values[np.searchsorted(unique,positions[v])]
                                    if not np.isclose(float(clean[positions[v],v]),raw[v],atol=0,rtol=0):
                                        raise ValueError('Observed label differs from input')
                            if i==0:
                                pred=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[c])[0])
                                if not np.isclose(pred,old['prediction'],atol=1e-7,rtol=0):
                                    raise ValueError('Original frozen forward mismatch')
                        if not np.isfinite(events).all(): raise ValueError('Nonfinite representation')
                        rows.append(dict(patient=p,context=c,component=components[p],body=old['body'],raw=raw,event=events))
                    d={k:np.stack([r[k] for r in rows]) for k in rows[0]}
                    joblib.dump(d,saved_path)
                    m=dict(patients=len(rows),sha256=sha256(saved_path),contract_sha256=sha256(cp))
                    mp.write_text(json.dumps(m,indent=2)+'\n')
                data[role]=d;metadata[role]=m
                print('Observed content extracted',fold,role,len(selected),flush=True)
            finally: db.close()
    finally: source.close()
    if set(data['fit']['patient'])&set(data['selection']['patient']): raise ValueError('Patient role overlap')
    return variables,data,contract,metadata


def run_fold(fold):
    variables,data,contract,extraction=extract(fold)
    dest=DEST/f'fold{fold}';cp=dest/'contract.json'
    frames=[];results={}
    for c,v in enumerate(variables):
        masks={role:d['raw'][:,106+c].astype(bool) for role,d in data.items()}
        f,q=[data[r] for r in ('fit','selection')];fm,qm=[masks[r] for r in ('fit','selection')]
        support=dict(fit=int(fm.sum()),selection=int(qm.sum()),components=len(set(q['component'][qm])))
        y=f['raw'][fm,c];truth=q['raw'][qm,c]
        if support['fit']<50 or support['selection']<30 or support['components']<3 or np.ptp(y)==0:
            results[v]=dict(status='unsupported',support=support)
            continue
        paths={ext:dest/f'{v}.{ext}' for ext in ('json','joblib','parquet')}
        if paths['json'].exists():
            info=json.loads(paths['json'].read_text())
            if (info['contract_sha256']!=sha256(cp) or info['model_sha256']!=sha256(paths['joblib'])
                    or info['prediction_sha256']!=sha256(paths['parquet'])): raise ValueError('Changed probe artifact')
            frame=pd.read_parquet(paths['parquet'])
        else:
            x={};query={}
            for role,d,m,out in (('fit',f,fm,x),('selection',q,qm,query)):
                process=d['raw'][m,106:]
                out['body']=np.c_[d['body'][m],process]
                out['event']=np.c_[d['event'][m,c],process]
                out['process']=process
            w=bounded_weights(f['component'][fm].tolist())
            models={};pred={}
            with threadpool_limits(limits=4):
                for name in ('event','body','process'):
                    scaler=StandardScaler().fit(x[name],sample_weight=w)
                    ridge=Ridge(alpha=100.,solver='cholesky').fit(scaler.transform(x[name]),y,sample_weight=w)
                    hgb=HistGradientBoostingRegressor(**PARAMETERS).fit(x[name],y,sample_weight=w)
                    models[name]=dict(scaler=scaler,ridge=ridge,hgb=hgb)
                    pred['ridge_'+name]=ridge.predict(scaler.transform(query[name]))
                    pred['hgb_'+name]=hgb.predict(query[name])
            pred['median']=np.full(len(truth),np.median(y))
            frame=pd.DataFrame(dict(patient=q['patient'][qm],component=q['component'][qm],variable=v,
                                    context=q['context'][qm],fold=fold,truth=truth,**pred))
            frame.to_parquet(paths['parquet'],index=False);joblib.dump(models,paths['joblib'])
            info=dict(status='fitted',support=support,contract_sha256=sha256(cp),
                      model_sha256=sha256(paths['joblib']),prediction_sha256=sha256(paths['parquet']))
            paths['json'].write_text(json.dumps(info,indent=2)+'\n')
        if (frame.patient.tolist()!=q['patient'][qm].tolist() or not np.array_equal(frame.truth,truth)
                or not np.array_equal(frame.context,q['context'][qm]) or not (frame.fold==fold).all()
                or not (frame.variable==v).all() or frame.component.tolist()!=q['component'][qm].tolist()
                or not np.isfinite(frame[['truth',*ARMS]].to_numpy()).all()): raise ValueError('Changed scored pairs')
        frames.append(frame);results[v]=info
        print('Observed probes',fold,c+1,'/53',v,flush=True)
    out=Path(f'artifacts/native_reconstruction_observed_content_fold{fold}_20261003.json')
    out.write_text(json.dumps(dict(contract=contract,extraction=extraction,targets=results,
        runtime_modified=False,promotion_allowed=False),indent=2,allow_nan=False)+'\n')
    return frames,{str(out):sha256(out)}


def main():
    out=Path('artifacts/native_reconstruction_observed_content_assessment_20261003.json')
    if out.exists(): raise ValueError('Existing completed observed-content assessment')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    frames=[];bindings={}
    for fold in range(5):
        rows,checked=run_fold(fold);frames.extend(rows);bindings.update(checked)
    if not frames: raise ValueError('No supported observed variables')
    data=pd.concat(frames,ignore_index=True)
    results={v:assess(data[data.variable==v],20261003+i*1000) for i,v in enumerate(sorted(data.variable.unique()))}
    summary=dict(variables_with_support=len(results),full_five_fold_variables=[v for v,r in results.items() if len(r['folds'])==5],
        query_occurrences=len(data),patients=int(data.patient.nunique()),
        normalized_macro={level:{a:float(np.mean([r['normalized_mae'][level][a] for r in results.values()]))
                                 for a in ARMS} for level in ('patient','component')},
        screens={a+'_minus_'+b:[v for v,r in results.items() if r['comparisons'][a+'_minus_'+b]['screen_pass']]
                 for a,b in CONTRASTS})
    out.write_text(json.dumps(dict(summary=summary,targets=results,bindings=bindings,runtime_modified=False,
        promotion_allowed=False,limitation='Observed-input recovery, not missing-variable prediction or future forecast. Bounded reused source. Reader-accessibility diagnostic, not irreversible information-loss proof.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
