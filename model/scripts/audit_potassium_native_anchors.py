"""Matched-patient native-anchor refit versus frozen glucose-anchor training."""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.audit_potassium_raw_history import history_features
from scripts.evaluate_potassium_tree_transfer import compare
from scripts.audit_event_aware_patient_state import paired
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.raw_lab_replay import canonical_events
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def native_queries(events, identities, variable='potassium'):
    """Queries depend on target result arrival, never future-label support."""
    identities=identities[['stay_id','subject_id']].drop_duplicates()
    if identities.isna().any().any() or identities.stay_id.duplicated().any():
        raise ValueError('ambiguous patient identity')
    ev=canonical_events(events.loc[events.variable.eq(variable)])
    q=ev[['stay_id','available_at']].drop_duplicates().rename(columns={'available_at':'t'})
    return q.merge(identities,on='stay_id',how='inner',validate='many_to_one').sort_values(
        ['stay_id','t'],kind='stable').reset_index(drop=True)


def prepare(old, raw, assets, origins):
    q=native_queries(raw,old)
    lab=future_pairs(q,raw,['potassium']);keep=lab['paired'][:,0]
    q=q.loc[keep].reset_index(drop=True);q['truth']=lab['future'][keep,0]
    current=lab['current'][keep,0]
    a,r=build_arrays(q,raw,assets,origins)
    col=assets.checkpoint['variables'].index('potassium')
    np.testing.assert_allclose(a['current'][r,col]*assets.scaler.scales[col]+assets.scaler.medians[col],current,atol=1e-5)
    old_keys=pd.MultiIndex.from_frame(old[['stay_id','t']])
    overlap=int(pd.MultiIndex.from_frame(q[['stay_id','t']]).isin(old_keys).sum())
    summary=dict(old_rows=len(old),native_rows=len(q),exact_query_overlap=overlap,
        old_patients=int(old.subject_id.nunique()),native_patients=int(q.subject_id.nunique()),
        current_age_median=float(np.median(lab['measurement_age'][keep,0])))
    return q,current,a,r,summary


def score(frame, new, old, key):
    result=compare(frame,new,key)
    result['old_fit']=compare(frame,old,key+'-old')
    result['refit_vs_old']={mode:paired(frame,new[mode],old[mode],key+'-refit-'+mode,1000)
                            for mode in ('own','full','support')}
    return result


def main():
    root=Path('/private/tmp');out=Path('artifacts/potassium_native_anchors_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    prior_path=Path('artifacts/potassium_tree_transfer_20260911.json')
    prior=json.loads(prior_path.read_text())
    if not prior.get('complete'):raise ValueError('prior incomplete')
    source_report=Path('artifacts/potassium_tree_history_20260911.json')
    selections=json.loads(source_report.read_text())['seeds']
    label=root/'auxiliary_supervision_20260910.npz'
    raw_paths=[root/'availability_jepa_eicu_labs.parquet',root/'encoder_source_expansion_20260910/labs.parquet']
    for path in [source_report,label]+raw_paths:
        if sha256(path)!=prior['binding'][str(path)]:raise ValueError('input binding changed')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    scale=assets.scaler.scales[col];module_ids=assets.model.variable_module_ids.numpy()
    y=np.load(label);valid=y['paired'][:,col]
    old=pd.DataFrame(dict(stay_id=y['stay_id'][valid],subject_id=y['subject_id'][valid],t=y['query_time'][valid]))
    all_source_subjects=set(y['subject_id'])
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(old.stay_id)]
    frame,current,a,r,support=prepare(old,raw,assets,{s:pd.Timestamp('2100-01-01') for s in old.stay_id.unique()})
    delta=(frame.truth.to_numpy()-current)/scale
    report=dict(status='diagnostic_only',runtime_modified=False,promotion_allowed=False,pristine_lockbox=False,
        binding={str(p):sha256(p) for p in [Path(__file__),prior_path,source_report,label]+raw_paths+
                 [Path('POTASSIUM_NATIVE_ANCHOR_PROTOCOL_20260911.md')]},source_support=support,seeds=[],roles={})
    fitted={};old_models={};private=root/'potassium_native_anchors_20260911';private.mkdir(exist_ok=True)
    for selection in selections:
        seed=selection['seed'];of,ot,oe=three_way_partition(old.subject_id,np.ones(len(old),bool),seed)
        partition=np.load(root/f'potassium_three_way_20260911/partition_{seed}.npz')
        for key,m in zip(('fit','tune','evaluation'),(of,ot,oe)):
            np.testing.assert_array_equal(m,partition[key][valid])
        fit=frame.subject_id.isin(set(old.loc[of,'subject_id'])).to_numpy()
        evaluation=frame.subject_id.isin(set(old.loc[oe,'subject_id'])).to_numpy()
        if np.any(fit & evaluation):raise ValueError('patient leakage')
        path=root/f'potassium_tree_transfer_20260911/seed_{seed}.joblib'
        if sha256(path)!=prior['binding'][str(path)]:raise ValueError('model binding changed')
        report['binding'][str(path)]=sha256(path);previous=joblib.load(path);old_models[seed]=previous
        allowed=source_patient_support(a,r,frame.subject_id,fit)>0
        view=restrict_input_support(a,allowed,module_ids)
        old_view=restrict_input_support(a,previous['allowed'],module_ids)
        subjects=frame.loc[fit,'subject_id'];w=1./subjects.groupby(subjects).transform('size').to_numpy(float)
        w*=len(w)/w.sum();models={};points={'persistence':current[evaluation]};old_points=dict(points)
        for mode in ('own','full','support'):
            x=history_features(view,r,col,mode).astype(np.float64)
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                max_leaf_nodes=selection['selection'][mode]['max_leaf_nodes'],min_samples_leaf=30,
                l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            model.fit(x[fit],delta[fit],sample_weight=w);models[mode]=model
            points[mode]=current[evaluation]+scale*model.predict(x[evaluation])
            ox=history_features(old_view,r[evaluation],col,mode).astype(np.float64)
            old_points[mode]=current[evaluation]+scale*previous['models'][mode].predict(ox)
        fitted[seed]=(allowed,models)
        dest=private/f'seed_{seed}.joblib';joblib.dump(dict(allowed=allowed,models=models),dest)
        report['binding'][str(dest)]=sha256(dest)
        report['seeds'].append(dict(seed=seed,fit_patients=int(frame.loc[fit,'subject_id'].nunique()),
            evaluation=score(frame.loc[evaluation],points,old_points,f'native-source-{seed}')))
        print(f'Native seed {seed} completed',flush=True)
    metadata=duckdb.sql("SELECT stay_id,intime,first_careunit FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    metadata['stay_id']='mimic:'+metadata.stay_id.astype(str)
    origins=dict(zip(metadata.stay_id,metadata.intime));units=dict(zip(metadata.stay_id,metadata.first_careunit))
    for role,db in (('eicu_hospital','eicu'),('mimic','mimic')):
        qpath=root/f'availability_glucose_{role}.parquet';rpath=root/f'availability_jepa_{db}_labs.parquet'
        for path in (qpath,rpath):
            if sha256(path)!=prior['binding'][str(path)]:raise ValueError('external changed')
            report['binding'][str(path)]=sha256(path)
        ef=pd.read_parquet(qpath);ef=ef.loc[ef.fresh & ef.truth.notna()].reset_index(drop=True)
        eraw=pd.read_parquet(rpath);eraw=eraw.loc[eraw.stay_id.isin(ef.stay_id)]
        original_labels=future_pairs(ef,eraw,['potassium'])
        ef=ef.loc[original_labels['paired'][:,0]].reset_index(drop=True)
        if set(ef.subject_id)&all_source_subjects:raise ValueError('external patient overlap')
        eraw=eraw.loc[eraw.stay_id.isin(ef.stay_id)]
        starts=origins if db=='mimic' else {s:pd.Timestamp('2100-01-01') for s in ef.stay_id.unique()}
        ef,ec,ea,er,es=prepare(ef,eraw,assets,starts);results=[]
        for seed,(allowed,models) in fitted.items():
            points={'persistence':ec};old_points=dict(points)
            ev=restrict_input_support(ea,allowed,module_ids)
            ov=restrict_input_support(ea,old_models[seed]['allowed'],module_ids)
            for mode in ('own','full','support'):
                points[mode]=ec+scale*models[mode].predict(history_features(ev,er,col,mode).astype(np.float64))
                old_points[mode]=ec+scale*old_models[seed]['models'][mode].predict(history_features(ov,er,col,mode).astype(np.float64))
            result=dict(seed=seed,pooled=score(ef,points,old_points,f'native-{role}-{seed}'),care_units={})
            if db=='mimic':
                group=ef.stay_id.map(units)
                for name in sorted(group.dropna().unique()):
                    mask=group.eq(name).to_numpy()
                    if ef.loc[mask,'subject_id'].nunique()>=30:
                        result['care_units'][name]=score(ef.loc[mask],{k:v[mask] for k,v in points.items()},
                            {k:v[mask] for k,v in old_points.items()},f'native-{role}-{seed}-{name}')
            results.append(result)
        report['roles'][role]=dict(support=es,seeds=results)
        print(f'Native {role} completed',flush=True)
    report['complete']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
