"""Matched predictions on reconstructed rate-record queries."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.event_window_history import event_windows
from whole_body_jepa.potassium_treatment_context import context_features,rate_history_features,rate_transition_features
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--influence',action='store_true');parser.add_argument('--snapshot-repair',action='store_true');args=parser.parse_args()
    if args.influence and args.snapshot_repair:raise ValueError('choose one audit')
    reference_path=Path('artifacts/rate_anchored_prediction_20260911.json')
    reference=json.loads(reference_path.read_text()) if args.influence or args.snapshot_repair else None
    output=Path('artifacts/rate_anchored_influence_20260911.json') if args.influence else reference_path
    if args.snapshot_repair:output=Path('artifacts/rate_snapshot_repair_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    source=Path('artifacts/rate_anchored_potassium_20260911.json')
    prior=json.loads(source.read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(root/name)!=prior['binding'][str(root/name)]:raise ValueError('changed cache')
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0];current=labels['current'][valid,0]
    after=labels['bracket_after'][valid]
    rawpath=Path('/private/tmp/insulin_potassium_pairs_20260911/labs.parquet')
    if sha256(rawpath)!=prior['binding'][str(rawpath)]:raise ValueError('changed labs')
    raw=pd.read_parquet(rawpath)
    base=np.column_stack([current,labels['measurement_age'][valid,0],event_windows(q,raw,['potassium'],np.array([0.]),np.array([1.])).reshape(len(q),-1)])
    rates=read_insulin_rates(Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz'))
    rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    rates=rates.rename(columns={'label':'original_label'});rates['action']='insulin';rates['dimension']='drug_units_per_hour'
    parts=[]
    for fn in (context_features,rate_history_features,rate_transition_features):
        values,cols=fn(q,rates);parts.append(values[:,[i for i,c in enumerate(cols) if c[0]=='insulin']])
    treatment=np.column_stack(parts);support=np.where(np.isfinite(treatment),0.,np.nan)
    if args.snapshot_repair:
        old_treatment=treatment.copy()
        _,columns=context_features(q.iloc[:0],rates)
        index=[c for c in columns if c[0]=='insulin'].index(('insulin','drug_units_per_hour'))
        rate_groups=dict(tuple(rates.groupby('stay_id')))
        for i,row in enumerate(q.itertuples()):
            e=rate_groups[row.stay_id]
            e=e[(e.offset>row.offset-360)&(e.offset<=row.offset)].sort_values(['offset','source_id'])
            if len(e):old_treatment[i,index]=e.value.iloc[-1]
    report=dict(complete=False,runtime_modified=False,promotion_allowed=False,treatment_availability_verified=False,
                status='retrospective_record_anchor_diagnostic',seeds=[],binding={str(p):sha256(p) for p in [Path(__file__),source,Path('RATE_ANCHORED_PREDICTION_PROTOCOL_20260911.md')]})
    if args.influence:report.update(fitting_performed=False,diagnostic='patient_error_influence')
    for entry in prior['seeds']:
        seed=entry['seed'];path=root/f'partition_{seed}.npz'
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        roles={k:v[valid] for k,v in dict(np.load(path)).items()}
        sets=[set(q.loc[m,'subject_id']) for m in roles.values()]
        if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError('patient overlap')
        fit=roles['fit'];subjects=q.loc[fit,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w*=len(w)/w.sum()
        points={'persistence':current};models={}
        if args.influence or args.snapshot_repair:
            saved=root/f'model_{seed}.joblib'
            if sha256(saved)!=reference['binding'][str(saved)]:raise ValueError('weights changed')
            frozen=joblib.load(saved)
        if args.snapshot_repair:points['original']=current+frozen['real'].predict(np.column_stack([base,old_treatment]))
        for name,x in [('own',base),('real',np.column_stack([base,treatment])),('support',np.column_stack([base,support]))]:
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            if args.influence:model=frozen[name]
            else:model.fit(x[fit],(q.truth.to_numpy()-current)[fit],sample_weight=w)
            points[name]=current+model.predict(x);models[name]=model
        result=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            result['roles'][role]={}
            for group,g in [('all',np.ones(len(q),bool)),('bracket_after',after)]:
                mask=roles[role]&g;f=q.loc[mask];p={k:v[mask] for k,v in points.items()}
                result['roles'][role][group]=dict(rows=len(f),patients=int(f.subject_id.nunique()),mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},comparisons={k:paired(f,p['real'],p[k],f'rateanchor-{seed}-{role}-{group}-{k}',1000) for k in ('own','support','persistence')})
                if args.snapshot_repair:
                    expected=next(s for s in reference['seeds'] if s['seed']==seed)['roles'][role][group]['mae']
                    if abs(patient_equal_mae(f,f.truth,p['original'])-expected['real'])>1e-10:raise ValueError('old snapshot reproduction failed')
                    if abs(patient_equal_mae(f,f.truth,p['own'])-expected['own'])>1e-10:raise ValueError('own control changed')
                    result['roles'][role][group]['comparisons']['original']=paired(f,p['real'],p['original'],f'snapshotrepair-{seed}-{role}-{group}',1000)
                if args.influence:
                    expected=next(s for s in reference['seeds'] if s['seed']==seed)['roles'][role][group]['mae']
                    for k,v in result['roles'][role][group]['mae'].items():
                        if abs(v-expected[k])>1e-10:raise ValueError('prediction reproduction failed')
                    errors=pd.DataFrame({'subject':f.subject_id.to_numpy(),'delta':np.abs(p['real']-f.truth.to_numpy())-np.abs(p['own']-f.truth.to_numpy())}).groupby('subject').delta.mean().to_numpy()
                    leave_one=(errors.sum()-errors)/(len(errors)-1)
                    result['roles'][role][group]['influence']=dict(patient_gain_fraction=float((errors<0).mean()),median_patient_delta=float(np.median(errors)),mean_patient_delta=float(errors.mean()),single_patient_sign_reversals=int((np.sign(leave_one)!=np.sign(errors.mean())).sum()),leave_one_mean_range=[float(leave_one.min()),float(leave_one.max())])
        path=root/f'{"snapshot_repair" if args.snapshot_repair else "model"}_{seed}.joblib'
        if not args.influence:joblib.dump(models,path)
        report['binding'][str(path)]=sha256(path)
        report['seeds'].append(result);print(f'Rate-anchor seed {seed} complete',flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
