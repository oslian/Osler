"""Localize frozen event-summary gains without outcome-driven routing."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.audit_auxiliary_learning import patient_quantile,movement_metrics
from scripts.audit_potassium_native_anchors import score
from scripts.audit_potassium_raw_history import history_features
from scripts.train_potassium_matched_encoder import three_way_partition
from whole_body_jepa.availability_inputs import build_arrays,restrict_input_support
from whole_body_jepa.event_window_history import event_windows,select_event_features
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def fit_global_gain(delta, prediction, subjects):
    """Patient-equal LAD scale, nonnegative but not capped at one."""
    delta=np.asarray(delta,float);prediction=np.asarray(prediction,float)
    if delta.shape!=prediction.shape or len(subjects)!=len(delta):raise ValueError('unaligned gain data')
    if not np.isfinite(delta).all() or not np.isfinite(prediction).all():raise ValueError('nonfinite gain data')
    weights=1./pd.Series(subjects).groupby(pd.Series(subjects)).transform('size').to_numpy(float)
    use=prediction!=0
    if not use.any():return 1.
    ratios=delta[use]/prediction[use];weights=weights[use]*abs(prediction[use])
    order=np.argsort(ratios,kind='stable')
    index=np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)
    return max(0.,float(ratios[order[index]]))


def decomposition(frame,current,prediction,reference,large):
    weights=1./frame.groupby('subject_id')['subject_id'].transform('size').to_numpy(float)
    weights/=weights.sum()
    errors=abs(prediction-frame.truth.to_numpy())-abs(reference-frame.truth.to_numpy())
    result=dict(pooled=float(np.dot(weights,errors)),large=float(np.dot(weights[large],errors[large])),
                smaller=float(np.dot(weights[~large],errors[~large])),large_weight=float(weights[large].sum()))
    if not np.isclose(result['pooled'],result['large']+result['smaller']):raise ValueError('decomposition failed')
    return result


def directions(frame,current,prediction):
    delta=frame.truth.to_numpy()-current;keep=abs(delta)>1e-8
    if not keep.any():return None
    f=frame.loc[keep];w=1./f.groupby('subject_id')['subject_id'].transform('size').to_numpy(float)
    return float(np.average(np.sign(prediction[keep]-current[keep])==np.sign(delta[keep]),weights=w))


def large_change_mask(delta,threshold):
    magnitude=abs(np.asarray(delta,float))
    return (magnitude>threshold)&~np.isclose(magnitude,threshold,rtol=1e-8,atol=1e-8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gain',action='store_true');args=parser.parse_args()
    output=Path('artifacts/potassium_event_gain_20260911.json' if args.gain else 'artifacts/potassium_event_changes_20260911_v2.json')
    if output.exists():raise ValueError('fresh output required')
    source=Path('artifacts/potassium_event_windows_20260911.json');prior=json.loads(source.read_text())
    if not prior.get('complete'):raise ValueError('incomplete source')
    root=Path('/private/tmp');label=root/'auxiliary_supervision_20260910.npz'
    paths=[root/'availability_jepa_eicu_labs.parquet',root/'encoder_source_expansion_20260910/labs.parquet']
    for path in [label]+paths:
        if sha256(path)!=prior['binding'][str(path)]:raise ValueError('source changed')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('potassium')
    y=np.load(label);valid=y['paired'][:,col]
    frame=pd.DataFrame(dict(stay_id=y['stay_id'][valid],subject_id=y['subject_id'][valid],
                            t=y['query_time'][valid],truth=y['future'][valid,col]))
    current=y['current'][valid,col];scale=assets.scaler.scales[col]
    raw=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True);raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    a,r=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    events=event_windows(frame,raw,variables,assets.scaler.medians,assets.scaler.scales)
    report=dict(status='retrospective_diagnostic_only',runtime_modified=False,promotion_allowed=False,
        external_data_read=False,binding={str(p):sha256(p) for p in [Path(__file__),source,label]+paths+
             [Path('POTASSIUM_EVENT_CHANGE_PROTOCOL_20260911.md')]},seeds=[])
    for entry in prior['seeds']:
        seed=entry['seed'];fit,tune,evaluation=three_way_partition(frame.subject_id,np.ones(len(frame),bool),seed)
        points={'persistence':current[evaluation]};old_points=dict(points);gains={}
        old_path=root/f'potassium_tree_transfer_20260911/seed_{seed}.joblib'
        new_path=root/f'potassium_event_windows_20260911/seed_{seed}.joblib'
        for p in (old_path,new_path):
            if sha256(p)!=prior['binding'][str(p)]:raise ValueError('checkpoint changed')
            report['binding'][str(p)]=sha256(p)
        old=joblib.load(old_path);new=joblib.load(new_path)
        np.testing.assert_array_equal(old['allowed'],new['allowed'])
        view=restrict_input_support(a,old['allowed'],assets.model.variable_module_ids.numpy())
        for mode in ('own','full','support'):
            base=history_features(view,r[evaluation],col,mode).astype(np.float64)
            ext=select_event_features(events[evaluation],new['allowed'],col,mode)
            old_points[mode]=current[evaluation]+scale*old['models'][mode].predict(base)
            points[mode]=current[evaluation]+scale*new['models'][mode].predict(np.concatenate([base,ext],axis=1))
            if args.gain:
                tb=history_features(view,r[tune],col,mode).astype(np.float64)
                te=select_event_features(events[tune],new['allowed'],col,mode)
                td=scale*new['models'][mode].predict(np.concatenate([tb,te],axis=1))
                gains[mode]=fit_global_gain(frame.loc[tune,'truth'].to_numpy()-current[tune],td,frame.loc[tune,'subject_id'].to_numpy())
        f=frame.loc[evaluation];c=current[evaluation]
        pooled=score(f,points,old_points,f'eventchange-{seed}')
        for mode,value in pooled['mae'].items():
            if abs(value-entry['evaluation']['mae'][mode])>1e-10:raise ValueError('new score reproduction failed')
            if abs(pooled['old_fit']['mae'][mode]-entry['evaluation']['old_fit']['mae'][mode])>1e-10:
                raise ValueError('old score reproduction failed')
        if args.gain:
            calibrated={'persistence':c}
            for mode,gain in gains.items():calibrated[mode]=c+gain*(points[mode]-c)
            report['seeds'].append(dict(seed=seed,gains=gains,evaluation=score(f,calibrated,points,f'gain-{seed}')))
            continue
        threshold=patient_quantile(abs(frame.loc[fit,'truth'].to_numpy()-current[fit]),frame.loc[fit,'subject_id'],.75)
        large=large_change_mask(f.truth.to_numpy()-c,threshold)
        item=dict(seed=seed,threshold=threshold,pooled=pooled,groups={},contributions={})
        for mode in ('own','full','support'):
            item['contributions'][mode]=decomposition(f,c,points[mode],old_points[mode],large)
        for name,mask in (('large',large),('smaller',~large)):
            g=f.loc[mask];p={k:v[mask] for k,v in points.items()};op={k:v[mask] for k,v in old_points.items()}
            result=score(g,p,op,f'eventchange-{seed}-{name}') if g.subject_id.nunique()>=30 else dict(supported=False)
            result['movement']={k:movement_metrics(g,c[mask],v) for k,v in p.items()}
            result['direction']={k:directions(g,c[mask],v) for k,v in p.items()}
            item['groups'][name]=result
        report['seeds'].append(item)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')
    print('Frozen change decomposition complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
