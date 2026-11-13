"""Source-frozen nonlinear comparator for incremental context transfer."""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.audit_potassium_raw_history import history_features
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def compare(frame, points, key):
    return dict(rows=len(frame), patients=int(frame.subject_id.nunique()),
        mae={k:patient_equal_mae(frame, frame.truth, p) for k,p in points.items()},
        comparisons={f'{a}_vs_{b}':paired(frame,points[a],points[b],f'{key}-{a}-{b}',1000)
            for a,b in (('full','own'),('full','support'),('full','persistence'),('own','persistence'))})


def main():
    output=Path('artifacts/potassium_tree_transfer_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp'); private=root/'potassium_tree_transfer_20260911'
    private.mkdir(exist_ok=True)
    prior_path=Path('artifacts/potassium_tree_history_20260911.json')
    prior=json.loads(prior_path.read_text())
    if not prior.get('complete'):raise ValueError('source search incomplete')
    training_path=Path('artifacts/potassium_three_way_20260911.json')
    training=json.loads(training_path.read_text())
    label=root/'auxiliary_supervision_20260910.npz'
    raw_paths=[root/'availability_jepa_eicu_labs.parquet',root/'encoder_source_expansion_20260910/labs.parquet']
    for p in [label]+raw_paths:
        if sha256(p)!=training['binding'][str(p)]:raise ValueError('source changed')
    if sha256(training_path)!=prior['binding'][str(training_path)]:raise ValueError('training binding changed')
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    module_ids=assets.model.variable_module_ids.numpy();scale=assets.scaler.scales[col]
    y=np.load(label)
    frame=pd.DataFrame(dict(stay_id=y['stay_id'],subject_id=y['subject_id'],t=y['query_time'],truth=y['future'][:,col]))
    source_subjects=set(frame.subject_id)
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    arrays,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    valid=y['paired'][:,col];f=frame.loc[valid].reset_index(drop=True);r=rows[valid]
    current=y['current'][valid,col];delta=(f.truth.to_numpy()-current)/scale
    report=dict(status='diagnostic_only',runtime_modified=False,promotion_allowed=False,pristine_lockbox=False,
        binding={str(p):sha256(p) for p in [Path(__file__),prior_path,training_path,label]+raw_paths+
                 [Path('POTASSIUM_TREE_TRANSFER_PROTOCOL_20260911.md')]},source_reproduction=[],roles={})
    models={}
    for entry in prior['seeds']:
        seed=entry['seed'];fit,tune,evaluation=three_way_partition(f.subject_id,np.ones(len(f),bool),seed)
        partition=np.load(root/f'potassium_three_way_20260911/partition_{seed}.npz')
        for key,m in zip(('fit','tune','evaluation'),(fit,tune,evaluation)):
            np.testing.assert_array_equal(m,partition[key][valid])
        full_fit=np.zeros(len(frame),bool);full_fit[valid]=fit
        allowed=source_patient_support(arrays,rows,frame.subject_id,full_fit)>0
        view=restrict_input_support(arrays,allowed,module_ids)
        subjects=f.loc[fit,'subject_id'];w=1./subjects.groupby(subjects).transform('size').to_numpy(float)
        w*=len(w)/w.sum();arms={};checks={}
        for mode in ('own','full','support'):
            x=history_features(view,r,col,mode).astype(np.float64)
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                max_leaf_nodes=entry['selection'][mode]['max_leaf_nodes'],min_samples_leaf=30,
                l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            model.fit(x[fit],delta[fit],sample_weight=w)
            p=current[evaluation]+scale*model.predict(x[evaluation])
            mae=patient_equal_mae(f.loc[evaluation],f.loc[evaluation,'truth'],p)
            if abs(mae-entry['mae'][mode])>1e-10:raise ValueError('source reproduction failed')
            checks[mode]=mae;arms[mode]=model
        models[seed]=(allowed,arms)
        path=private/f'seed_{seed}.joblib'
        joblib.dump(dict(allowed=allowed,models=arms,variables=assets.checkpoint['variables']),path)
        report['binding'][str(path)]=sha256(path)
        report['source_reproduction'].append(dict(seed=seed,mae=checks))
        print(f'Source seed {seed} reproduced',flush=True)
    metadata=duckdb.sql("SELECT stay_id,intime,first_careunit FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    metadata['stay_id']='mimic:'+metadata.stay_id.astype(str)
    origins=dict(zip(metadata.stay_id,metadata.intime));units=dict(zip(metadata.stay_id,metadata.first_careunit))
    for role,db in (('eicu_hospital','eicu'),('mimic','mimic')):
        query_path=root/f'availability_glucose_{role}.parquet';raw_path=root/f'availability_jepa_{db}_labs.parquet'
        report['binding'].update({str(p):sha256(p) for p in (query_path,raw_path)})
        ef=pd.read_parquet(query_path);ef=ef.loc[ef.fresh & ef.truth.notna()].reset_index(drop=True)
        if set(ef.subject_id)&source_subjects:raise ValueError('source overlap')
        eraw=pd.read_parquet(raw_path);eraw=eraw.loc[eraw.stay_id.isin(ef.stay_id)]
        labels=future_pairs(ef,eraw,['potassium']);keep=labels['paired'][:,0]
        ef=ef.loc[keep].reset_index(drop=True);ef['truth']=labels['future'][keep,0]
        ec=labels['current'][keep,0]
        starts=origins if db=='mimic' else {s:pd.Timestamp('2100-01-01') for s in ef.stay_id.unique()}
        ea,er=build_arrays(ef,eraw,assets,starts)
        np.testing.assert_allclose(ea['current'][er,col]*scale+assets.scaler.medians[col],ec,atol=1e-5)
        results=[]
        for seed,(allowed,arms) in models.items():
            ev=restrict_input_support(ea,allowed,module_ids)
            points={'persistence':ec}
            for mode,model in arms.items():
                points[mode]=ec+scale*model.predict(history_features(ev,er,col,mode).astype(np.float64))
            result=dict(seed=seed,pooled=compare(ef,points,f'{role}-{seed}'),care_units={})
            if db=='mimic':
                group=ef.stay_id.map(units)
                for name in sorted(group.dropna().unique()):
                    mask=group.eq(name).to_numpy()
                    if ef.loc[mask,'subject_id'].nunique()>=30:
                        result['care_units'][name]=compare(ef.loc[mask],{k:v[mask] for k,v in points.items()},f'{role}-{seed}-{name}')
            results.append(result)
        report['roles'][role]=results
        print(f'{role} completed',flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
