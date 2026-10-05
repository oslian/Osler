"""Fixed expanded-cohort information diagnostic; not release promotion."""
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
from whole_body_jepa.potassium_treatment_context import context_features, rate_history_features, shuffled_context, rate_transition_features
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--transitions',action='store_true')
    parser.add_argument('--support-control',action='store_true')
    parser.add_argument('--postdraw-audit',action='store_true')
    args=parser.parse_args()
    if args.postdraw_audit:args.support_control=True
    if args.support_control:args.transitions=True
    name='expanded_insulin_transitions_20260911' if args.transitions else 'expanded_insulin_potassium_20260911'
    if args.support_control:name='expanded_insulin_support_control_20260911'
    if args.postdraw_audit:name='expanded_insulin_postdraw_20260911'
    output=Path(f'artifacts/{name}.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/insulin_potassium_pairs_20260911')
    partition_path=Path('artifacts/insulin_potassium_partition_20260911.json')
    partition=json.loads(partition_path.read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(root/name)!=partition['binding'][str(root/name)]:raise ValueError('cache changed')
    labels=np.load(root/'labels.npz');valid=labels['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][valid,0]
    current=labels['current'][valid,0];sequence=labels['sequence'][valid]
    raw=pd.read_parquet(root/'labs.parquet')
    base=np.column_stack([current,labels['measurement_age'][valid,0],event_windows(q,raw,['potassium'],np.array([0.]),np.array([1.])).reshape(len(q),-1)])
    rates=read_insulin_rates(Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz'))
    rates['stay_id']='eicu:'+rates.stay_id.astype(str)
    rates=rates.rename(columns={'label':'original_label'})
    rates['action']='insulin';rates['dimension']='drug_units_per_hour'
    snapshot,cols=context_features(q,rates)
    history,hcols=rate_history_features(q,rates)
    keep=[i for i,c in enumerate(cols) if c[0]=='insulin']
    hkeep=[i for i,c in enumerate(hcols) if c[0]=='insulin']
    treatment=np.column_stack([snapshot[:,keep],history[:,hkeep]])
    original_treatment=treatment.copy()
    reference_path=Path('artifacts/expanded_insulin_potassium_20260911.json')
    reference=json.loads(reference_path.read_text()) if args.transitions else None
    transition_path=Path('artifacts/expanded_insulin_transitions_20260911.json')
    transition_report=json.loads(transition_path.read_text()) if args.support_control else None
    if args.transitions:
        transitions,tcols=rate_transition_features(q,rates)
        treatment=np.column_stack([treatment,transitions[:,[i for i,c in enumerate(tcols) if c[0]=='insulin']]])
    if args.postdraw_audit:
        last_change_age=transitions[:,tcols.index(('insulin','drug_units_per_hour','last_change_age_hours'))]
        postdraw=np.isfinite(last_change_age)&(last_change_age<labels['measurement_age'][valid,0])
        support_path=Path('artifacts/expanded_insulin_support_control_20260911.json')
        support_report=json.loads(support_path.read_text())
    if not np.array_equal(sequence,np.isfinite(history[:,hcols.index(('insulin','drug_units_per_hour','change'))])):raise ValueError('sequence mismatch')
    report=dict(complete=False,promotion_allowed=False,runtime_modified=False,treatment_availability_verified=False,
                status='retrospective_information_diagnostic',pristine_lockbox=False,
                placebo_limitation='donor history may change availability support; not an exact support-matched placebo',
                binding={str(p):sha256(p) for p in [Path(__file__),partition_path,Path('EXPANDED_INSULIN_POTASSIUM_PROTOCOL_20260911.md')]},seeds=[])
    if args.transitions:
        report['binding'][str(reference_path)]=sha256(reference_path)
        report['binding']['TRANSITION_ACCURACY_PROTOCOL_20260911.md']=sha256(Path('TRANSITION_ACCURACY_PROTOCOL_20260911.md'))
    if args.support_control:
        report['control']='finite mask only; shuffled key denotes support-only trained control, not donor placebo'
        report['binding'][str(transition_path)]=sha256(transition_path)
    if args.postdraw_audit:
        report['fitting_performed']=False
        report['binding'][str(support_path)]=sha256(support_path)
    for entry in partition['seeds']:
        seed=entry['seed'];path=root/f'partition_{seed}.npz'
        if sha256(path)!=entry['partition_sha256']:raise ValueError('partition changed')
        roles=dict(np.load(path));fit=roles['fit']
        shuffled=shuffled_context(q,treatment,[roles[k] for k in ('fit','tune','evaluation','hospital')],seed)
        if args.support_control:shuffled=np.where(np.isfinite(treatment),0.,np.nan)
        subjects=q.loc[fit,'subject_id'];weights=1/subjects.groupby(subjects).transform('size').to_numpy(float)
        weights*=len(weights)/weights.sum()
        points={'persistence':current};models={}
        if args.transitions:
            old_path=root/f'diagnostic_{seed}.joblib'
            if sha256(old_path)!=reference['binding'][str(old_path)]:raise ValueError('old weights changed')
            old=joblib.load(old_path)
            points['original']=current+old['real'].predict(np.column_stack([base,original_treatment]))
        if args.support_control:
            tp=root/f'transition_{seed}.joblib'
            if sha256(tp)!=transition_report['binding'][str(tp)]:raise ValueError('transition weights changed')
            frozen_transition=joblib.load(tp)
        if args.postdraw_audit:
            sp=root/f'support_control_{seed}.joblib'
            if sha256(sp)!=support_report['binding'][str(sp)]:raise ValueError('support weights changed')
            frozen_support=joblib.load(sp)
        for name,x in [('own',base),('real',np.column_stack([base,treatment])),('shuffled',np.column_stack([base,shuffled]))]:
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10.,max_bins=63,early_stopping=False,random_state=seed)
            if args.transitions and name=='own':model=old['own']
            elif args.support_control and name=='real':model=frozen_transition['real']
            elif args.postdraw_audit:model=frozen_support[name]
            else:model.fit(x[fit],(q.truth.to_numpy()-current)[fit],sample_weight=weights)
            points[name]=current+model.predict(x);models[name]=model
        result=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            result['roles'][role]={}
            groups=[('all',np.ones(len(q),bool)),('sequence',sequence),('without_sequence',~sequence)]
            if args.postdraw_audit:groups += [('change_after_draw',postdraw),('sequence_without_postdraw_change',sequence&~postdraw)]
            for group,gmask in groups:
                mask=roles[role]&gmask;f=q.loc[mask];p={k:v[mask] for k,v in points.items()}
                if args.transitions and group in ('all','sequence','without_sequence'):
                    prior=next(s for s in reference['seeds'] if s['seed']==seed)['roles'][role][group]['mae']
                    for key,oldkey in [('original','real'),('own','own')]:
                        if abs(patient_equal_mae(f,f.truth,p[key])-prior[oldkey])>1e-10:raise ValueError('reference reproduction failed')
                    if args.support_control:
                        expected=next(s for s in transition_report['seeds'] if s['seed']==seed)['roles'][role][group]['mae']['real']
                        if abs(patient_equal_mae(f,f.truth,p['real'])-expected)>1e-10:raise ValueError('transition reproduction failed')
                comparisons=('own','shuffled','persistence','original') if args.transitions else ('own','shuffled','persistence')
                result['roles'][role][group]=dict(patients=int(f.subject_id.nunique()),rows=len(f),mae={k:patient_equal_mae(f,f.truth,v) for k,v in p.items()},comparisons={k:paired(f,p['real'],p[k],f'expanded-{seed}-{role}-{group}-{k}',1000) for k in comparisons})
        dest=root/f'{"support_control" if args.support_control else "transition" if args.transitions else "diagnostic"}_{seed}.joblib'
        if not args.postdraw_audit:joblib.dump(models,dest)
        report['binding'][str(dest)]=sha256(dest)
        report['seeds'].append(result);print(f'Expanded diagnostic seed {seed} complete',flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
