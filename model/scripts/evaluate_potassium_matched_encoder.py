"""Frozen source-trained potassium transfer, with identical available-time labels."""
import json
import argparse
from copy import deepcopy
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from scripts.train_availability_encoder import predict
from whole_body_jepa.availability_inputs import build_arrays, restrict_input_support
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256, SEEDS
from whole_body_jepa.runtime import load_release


def comparison(frame,predictions,key):
    return dict(rows=len(frame),patients=int(frame.subject_id.nunique()),
        mae={n:patient_equal_mae(frame,frame.truth,p) for n,p in predictions.items()},
        comparisons={f'{a}_vs_{b}':paired(frame,predictions[a],predictions[b],f'{key}-{a}-{b}',1000)
            for a,b in (('whole','own'),('whole','persistence'),('own','persistence'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freshness',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    destination=Path('artifacts/potassium_freshness_20260911.json' if args.freshness else
                     'artifacts/potassium_matched_external_20260911.json')
    if destination.exists():raise ValueError('fresh output required')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables'];col=variables.index('potassium')
    training=Path('artifacts/potassium_matched_encoder_20260911.json')
    if not json.loads(training.read_text()).get('complete'):raise ValueError('training incomplete')
    labels_path=Path('/private/tmp/auxiliary_supervision_20260910.npz')
    source_subjects=set(np.load(labels_path)['subject_id'])
    metadata=duckdb.sql("SELECT stay_id,intime,first_careunit FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    metadata['stay_id']='mimic:'+metadata.stay_id.astype(str)
    origins=dict(zip(metadata.stay_id,metadata.intime));units=dict(zip(metadata.stay_id,metadata.first_careunit))
    protocol=Path('POTASSIUM_EXTERNAL_PROTOCOL_20260911.md')
    report=dict(status='validation_only',runtime_modified=False,promotion_allowed=False,
        pristine_lockbox=False,roles={},binding={str(p):sha256(p) for p in [Path(__file__),protocol,training,labels_path]})
    for role,db in (('eicu_hospital','eicu'),('mimic','mimic')):
        query_path=Path(f'/private/tmp/availability_glucose_{role}.parquet')
        raw_path=Path(f'/private/tmp/availability_jepa_{db}_labs.parquet')
        report['binding'].update({str(p):sha256(p) for p in (query_path,raw_path)})
        frame=pd.read_parquet(query_path)
        frame=frame.loc[frame.fresh & frame.truth.notna()].reset_index(drop=True)
        if set(frame.subject_id)&source_subjects:raise ValueError('source patient overlap')
        raw=pd.read_parquet(raw_path);raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
        label=future_pairs(frame,raw,['potassium'])
        keep=label['paired'][:,0]
        frame=frame.loc[keep].reset_index(drop=True)
        frame['truth']=label['future'][keep,0]
        current=label['current'][keep,0]
        ages=label['measurement_age'][keep,0]
        draw_intervals=ages+label['actual_delta_t'][keep,0]
        if frame.subject_id.nunique()<30:raise ValueError('insufficient evaluation support')
        starts=origins if db=='mimic' else {s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()}
        arrays,rows=build_arrays(frame,raw,assets,starts)
        recovered=arrays['current'][rows,col]*assets.scaler.scales[col]+assets.scaler.medians[col]
        if not np.allclose(recovered,current,atol=1e-5):raise ValueError('current encoding differs')
        results=[]
        for seed in SEEDS:
            points={'persistence':current}
            for arm in ('own','whole'):
                path=Path(f'/private/tmp/potassium_matched_encoder_20260911/{arm}_{seed}.pt')
                report['binding'][str(path)]=sha256(path)
                c=torch.load(path,map_location='cpu',weights_only=False)
                if c['variables']!=variables or c['target']!='potassium':raise ValueError('checkpoint contract mismatch')
                model=deepcopy(assets.model);model.load_state_dict(c['state_dict'])
                model.value_delta_activation=torch.nn.Identity();model.belief_input_mode=c['belief_input_mode']
                view=restrict_input_support(arrays,c['allowed_variables'],assets.model.variable_module_ids.numpy())
                points[arm]=predict(model,view,rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
            result=dict(seed=int(seed),pooled=comparison(frame,points,f'{role}-{seed}'),care_units={})
            if args.freshness:
                prior=json.loads(Path('artifacts/potassium_matched_external_20260911.json').read_text())
                previous=next(s for s in prior['roles'][role] if s['seed']==seed)
                if result['pooled']!=previous['pooled']:raise ValueError('pooled reproduction failed')
                result['age_summary']=dict(median_hours=float(np.median(ages)),
                    draw_interval_median_hours=float(np.median(draw_intervals)))
                result['age_groups']={}
                for name,mask in (('le1',ages<=1),('gt1_le3',(ages>1)&(ages<=3)),('gt3_le6',ages>3)):
                    n=int(frame.loc[mask,'subject_id'].nunique())
                    result['age_groups'][name]=comparison(frame.loc[mask],{k:v[mask] for k,v in points.items()},
                        f'{role}-{seed}-age-{name}') if n>=30 else dict(patients=n,rows=int(mask.sum()),supported=False)
            if db=='mimic':
                group=frame.stay_id.map(units)
                for name in sorted(group.dropna().unique()):
                    mask=group.eq(name).to_numpy()
                    if frame.loc[mask,'subject_id'].nunique()>=30:
                        result['care_units'][name]=comparison(frame.loc[mask],{k:v[mask] for k,v in points.items()},f'{role}-{seed}-{name}')
                        if args.freshness:
                            result['care_units'][name]['age_median_hours']=float(np.median(ages[mask]))
            results.append(result)
        report['roles'][role]=results
    report['complete']=True
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print('Frozen transfer complete',flush=True)


if __name__=='__main__':main()
