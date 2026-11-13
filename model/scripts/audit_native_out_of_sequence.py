"""Count delayed older measurements actually present in native history."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256
from scripts.audit_event_aware_patient_state import paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accuracy',action='store_true')
    args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/native_out_of_sequence_accuracy_20260911.json' if args.accuracy else 'artifacts/native_out_of_sequence_20260911_v2.json')
    if out.exists():raise ValueError('fresh output required')
    raw=pd.read_parquet(root/'native_events.parquet')
    labs=raw.loc[raw.source_kind.eq('lab_available')].copy()
    age=(labs.event_time-labs.measured_at).dt.total_seconds()/3600
    labs=labs.loc[age.between(0,6)]
    # Match token selection: latest draw within a simultaneous variable group.
    keys=['stay_id','variable','event_time']
    latest=labs.groupby(keys).measured_at.transform('max')
    labs=labs.loc[labs.measured_at.eq(latest)]
    unambiguous=labs.groupby(keys).value.transform('nunique').eq(1)
    labs=labs.loc[unambiguous].drop_duplicates(keys).sort_values(['stay_id','variable','event_time'])
    maximum=labs.groupby(['stay_id','variable']).measured_at.cummax()
    previous=maximum.groupby([labs.stay_id,labs.variable]).shift()
    labs['late_older']=labs.measured_at<previous
    inventory=[]
    for variable,g in labs.groupby('variable'):
        delayed=g.loc[g.late_older]
        inventory.append(dict(variable=variable,encoded_groups=len(g),older_groups=len(delayed),
                              affected_stays=int(delayed.stay_id.nunique())))
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    groups={s:g for s,g in labs.groupby('stay_id')}
    affected=np.zeros(len(q),bool);glucose=np.zeros(len(q),bool)
    for i,row in enumerate(q.itertuples()):
        g=groups.get(row.stay_id)
        if g is None:continue
        window=g.loc[g.event_time.between(row.t-pd.Timedelta(hours=48),row.t,inclusive='right')]
        maxima=window.groupby('variable').measured_at.cummax()
        prior=maxima.groupby(window.variable).shift()
        older=window.measured_at<prior
        affected[i]=older.any();glucose[i]=(older&window.variable.eq('glucose')).any()
    report=dict(complete=True,status='chronology_inventory',runtime_modified=False,
        inventory=inventory,queries=len(q),queries_with_older_arrival=int(affected.sum()),
        queries_with_older_glucose_arrival=int(glucose.sum()),
        binding={str(p):sha256(p) for p in (root/'native_events.parquet',root/'queries.parquet',root/'labels.npz',Path(__file__))})
    if args.accuracy:
        reference=Path('artifacts/rate_glucose_jepa_20260911.json')
        ref=json.loads(reference.read_text())
        for name,digest in ref['binding'].items():
            if sha256(Path(name))!=digest:raise ValueError('changed source input')
        labels=np.load(root/'multitarget_labels.npz');k=list(labels['variables']).index('glucose')
        q=q.rename(columns={'uniquepid':'subject_id'});q['truth']=labels['future'][:,k]
        report['status']='preanchor_stratified_accuracy';report['seeds']=[]
        report['binding'][str(reference)]=sha256(reference)
        for entry in ref['seeds']:
            seed=entry['seed'];points=np.load(root/f'glucose_native_predictions_{seed}.npz')
            partitions=np.load(root/f'partition_{seed}.npz');item=dict(seed=seed,roles={})
            for role in ('evaluation','hospital'):
                groups={}
                for name,condition in [('older_glucose',glucose),('no_older_glucose',~glucose)]:
                    mask=partitions[role][valid]&labels['paired'][:,k]&condition;f=q.loc[mask]
                    groups[name]=dict(rows=int(mask.sum()),patients=int(f.subject_id.nunique()),
                        comparison=paired(f,points['real'][mask],points['own'][mask],f'older-arrival:{seed}:{role}:{name}',2000))
                item['roles'][role]=groups
            report['seeds'].append(item)
        for role in ('evaluation','hospital'):
            for group in ('older_glucose','no_older_glucose'):
                z=[s['roles'][role][group] for s in report['seeds']]
                print(role,group,'patients',[g['patients'] for g in z],
                    'delta',[round(g['comparison']['mean'],4) for g in z],
                    'gain/harm',sum(g['comparison']['ci95'][1]<0 for g in z),sum(g['comparison']['ci95'][0]>0 for g in z))
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('queries',len(q),'any older',affected.sum(),'glucose older',glucose.sum())
    print(sorted(inventory,key=lambda x:x['older_groups'],reverse=True)[:8])


if __name__=='__main__':main()
