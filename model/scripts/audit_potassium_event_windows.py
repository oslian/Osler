"""Test lost within-window histories on fixed source patients and labels."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from scripts.audit_potassium_raw_history import history_features
from scripts.audit_potassium_native_anchors import score
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.event_window_history import event_windows, select_event_features, WINDOWS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release, HISTORY_LAGS_HOURS
from whole_body_jepa.availability_inputs import restrict_input_support
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    output=Path('artifacts/potassium_event_windows_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp');label=root/'auxiliary_supervision_20260910.npz'
    transfer_path=Path('artifacts/potassium_tree_transfer_20260911.json')
    transfer=json.loads(transfer_path.read_text());prior_path=Path('artifacts/potassium_tree_history_20260911.json')
    prior=json.loads(prior_path.read_text());raw_paths=[root/'availability_jepa_eicu_labs.parquet',root/'encoder_source_expansion_20260910/labs.parquet']
    for p in [label,prior_path]+raw_paths:
        if sha256(p)!=transfer['binding'][str(p)]:raise ValueError('source changed')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('potassium')
    y=np.load(label);valid=y['paired'][:,col]
    frame=pd.DataFrame(dict(stay_id=y['stay_id'][valid],subject_id=y['subject_id'][valid],
                            t=y['query_time'][valid],truth=y['future'][valid,col]))
    current=y['current'][valid,col];scale=assets.scaler.scales[col]
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    a,r=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    events=event_windows(frame,raw,variables,assets.scaler.medians,assets.scaler.scales)
    private=root/'potassium_event_windows_20260911';private.mkdir(exist_ok=True)
    np.save(private/'events.npy',events)
    report=dict(status='source_diagnostic_only',runtime_modified=False,promotion_allowed=False,
        external_data_read=False,binding={str(p):sha256(p) for p in [Path(__file__),label,prior_path,transfer_path]+raw_paths+
             [Path('whole_body_jepa/event_window_history.py'),Path('POTASSIUM_EVENT_WINDOWS_PROTOCOL_20260911.md')]},seeds=[],excursions={})
    h=a['history'][r];values=a['current'][h];masks=a['mask'][h]>0
    for wi,window in enumerate(WINDOWS):
        take=np.asarray(HISTORY_LAGS_HOURS)<=window
        low=np.where(masks[:,take],values[:,take],np.inf).min(axis=1)
        high=np.where(masks[:,take],values[:,take],-np.inf).max(axis=1)
        supported=np.isfinite(low)&(events[:,:,wi,6]>0)
        extends=supported&((events[:,:,wi,2]<low-1e-6)|(events[:,:,wi,3]>high+1e-6))
        report['excursions'][str(window)]=dict(supported_entries=int(supported.sum()),
            outside_lag_extrema=int(extends.sum()),fraction=float(extends.sum()/max(supported.sum(),1)))
    delta=(frame.truth.to_numpy()-current)/scale
    for entry in prior['seeds']:
        seed=entry['seed'];fit,tune,evaluation=three_way_partition(frame.subject_id,np.ones(len(frame),bool),seed)
        partition=np.load(root/f'potassium_three_way_20260911/partition_{seed}.npz')
        for k,m in zip(('fit','tune','evaluation'),(fit,tune,evaluation)):
            np.testing.assert_array_equal(m,partition[k][valid])
        path=root/f'potassium_tree_transfer_20260911/seed_{seed}.joblib'
        if sha256(path)!=transfer['binding'][str(path)]:raise ValueError('old model changed')
        report['binding'][str(path)]=sha256(path);old=joblib.load(path)
        view=restrict_input_support(a,old['allowed'],assets.model.variable_module_ids.numpy())
        subjects=frame.loc[fit,'subject_id'];w=1./subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={'persistence':current[evaluation]};old_points=dict(points);models={}
        for mode in ('own','full','support'):
            base=history_features(view,r,col,mode).astype(np.float64)
            old_points[mode]=current[evaluation]+scale*old['models'][mode].predict(base[evaluation])
            old_mae=patient_equal_mae(frame.loc[evaluation],frame.loc[evaluation,'truth'],old_points[mode])
            if abs(old_mae-entry['mae'][mode])>1e-7:raise ValueError('old score failed reproduction')
            x=np.concatenate([base,select_event_features(events,old['allowed'],col,mode)],axis=1)
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                max_leaf_nodes=entry['selection'][mode]['max_leaf_nodes'],min_samples_leaf=30,
                l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            model.fit(x[fit],delta[fit],sample_weight=w);models[mode]=model
            points[mode]=current[evaluation]+scale*model.predict(x[evaluation])
        dest=private/f'seed_{seed}.joblib';joblib.dump(dict(allowed=old['allowed'],models=models),dest)
        report['binding'][str(dest)]=sha256(dest)
        report['seeds'].append(dict(seed=seed,evaluation=score(frame.loc[evaluation],points,old_points,f'eventwindows-{seed}')))
        print(f'Event window seed {seed} completed',flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
