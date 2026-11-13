"""Re-encode real availability-safe labs in frozen JEPA; matched glucose labels."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from extractors.eicu_sepsis_transition_extract import LAB_TO_STATE, PLAUSIBLE
from extractors.mimiciv_observation_transition_extract import LAB_ITEMIDS, EXTRA_PLAUSIBLE
from whole_body_jepa.availability_inputs import UNITS, filter_units, build_arrays
from whole_body_jepa.multisource_intervals import SEEDS, sha256
from whole_body_jepa.runtime import load_release
from scripts.audit_continual_validated_cell import release_features
from scripts.audit_event_aware_patient_state import patient_partition, paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def extract(db, wanted, variables, *, output):
    root = Path('/Users/chunyouchang/Downloads')
    c = duckdb.connect()
    c.execute('SET threads=4')
    ids = pd.DataFrame({'stay_id':sorted(wanted)})
    ids['native'] = ids.stay_id.str.split(':').str[-1].astype(int)
    c.register('wanted',ids)
    if db == 'eicu':
        mapping = pd.DataFrame([(k,v) for k,v in LAB_TO_STATE.items() if v in variables and v in UNITS],columns=['name','variable'])
        c.register('mapping',mapping)
        raw = c.execute('''SELECT w.stay_id, cast(l.labid as varchar) source_id, m.variable,
            l.labname source_test_name,
            timestamp '2100-01-01'+l.labresultoffset*interval '1 minute' measured_at,
            timestamp '2100-01-01'+greatest(l.labresultoffset,l.labresultrevisedoffset)*interval '1 minute' available_at,
            cast(l.labresult as double) AS "value", coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface) unit
            FROM read_csv_auto(?) l JOIN wanted w ON l.patientunitstayid=w.native
            JOIN mapping m ON l.labname=m.name WHERE l.labresultrevisedoffset IS NOT NULL''',
            [str(root/'eicu-collaborative-research-database-2.0/lab.csv.gz')]).df()
        origins = {s:pd.Timestamp('2100-01-01') for s in wanted}
    else:
        mapping = pd.DataFrame([(i,v) for v,items in LAB_ITEMIDS.items() if v in variables and v in UNITS for i in items],columns=['itemid','variable'])
        c.register('mapping',mapping)
        stays = root/'mimic-iv-3.1/icu/icustays.csv.gz'
        raw = c.execute('''SELECT w.stay_id, cast(l.labevent_id as varchar) source_id, m.variable,
            l.charttime measured_at, greatest(l.charttime,l.storetime) available_at,
            cast(l.valuenum as double) AS "value", l.valueuom unit
            FROM read_csv_auto(?) l JOIN mapping m ON l.itemid=m.itemid
            JOIN read_csv_auto(?) s ON l.subject_id=s.subject_id AND l.hadm_id=s.hadm_id
            AND l.charttime BETWEEN s.intime AND s.outtime JOIN wanted w ON s.stay_id=w.native
            WHERE l.storetime IS NOT NULL''',[str(root/'mimic-iv-3.1/hosp/labevents.csv.gz'),str(stays)]).df()
        starts = c.execute('SELECT w.stay_id, s.intime FROM read_csv_auto(?) s JOIN wanted w ON s.stay_id=w.native',[str(stays)]).df()
        origins = dict(zip(starts.stay_id,starts.intime))
    c.close()
    kept = filter_units(raw,{**PLAUSIBLE,**EXTRA_PLAUSIBLE})
    audit = dict(raw_rows=len(raw), accepted_rows=len(kept),
        variable_counts={k:int(v) for k,v in kept.variable.value_counts().items()},
        unmapped_or_unverified_variables=sorted(set(variables)-set(kept.variable)),
        unit_inventory=raw.groupby(['variable','unit'],dropna=False).size().reset_index(name='rows').fillna('').to_dict('records'))
    kept.to_parquet(output,index=False)
    return kept,origins,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('refusing overwrite')
    torch.set_num_threads(4)
    assets=load_release(Path('models'))
    roles={r:pd.read_parquet(f'/private/tmp/availability_glucose_{r}.parquet') for r in ('development','eicu_hospital','mimic')}
    raw,origins,support={},{},{}
    for db in ('eicu','mimic'):
        wanted=set(s for r,f in roles.items() if (r=='mimic')==(db=='mimic') for s in f.stay_id)
        print('extract '+db,flush=True)
        raw[db],origins[db],support[db]=extract(db,wanted,assets.checkpoint['variables'],
            output=Path(f'/private/tmp/availability_jepa_{db}_labs.parquet'))
        print(json.dumps(support[db]),flush=True)
    features,predictions,data={},{},{}
    for role,frame in roles.items():
        db='mimic' if role=='mimic' else 'eicu'
        valid=(frame.fresh & frame.truth.notna()).to_numpy()
        data[role]=frame.loc[valid].reset_index(drop=True)
        features[role],predictions[role]={},{}
        for mode,own,truthful in [('own',True,True),('whole',False,True),('legacy_masks',False,False)]:
            print('encode '+role+' '+mode,flush=True)
            arrays,rows=build_arrays(frame,raw[db],assets,origins[db],target_only=own,truthful=truthful)
            index=assets.checkpoint['variables'].index('glucose')
            reconstructed=arrays['current'][rows,index]*assets.scaler.scales[index]+assets.scaler.medians[index]
            if not np.allclose(reconstructed[valid],frame.current.to_numpy()[valid],atol=.001):
                raise ValueError('glucose reconstruction differs from matched anchor')
            f,y=release_features(assets,arrays,rows[valid])
            features[role][mode],predictions[role][mode]=f,y
        np.savez_compressed(f'/private/tmp/availability_jepa_{role}_features.npz',**features[role],
                            **{'prediction_'+k:v for k,v in predictions[role].items()})
    report=dict(status='candidate_only',runtime_modified=False,promotion_allowed=False,causal_claim_allowed=False,
        untouched_lockbox=False, scope='full checkpoint schema; only time/unit verified laboratory channels populated',
        extraction=support, frozen={},seeds=[],binding={str(f):sha256(f) for f in
        [Path(__file__),Path('whole_body_jepa/availability_inputs.py'),Path('models/whole_body_joint_jepa.pt'),Path('AVAILABILITY_JEPA_PROTOCOL_20260910.md')]})
    for role,f in data.items():
        report['frozen'][role]=dict(rows=len(f),patients=int(f.subject_id.nunique()),
            mae={k:patient_equal_mae(f,f.truth,y) for k,y in {**predictions[role],'persistence':f.current.to_numpy()}.items()},
            whole_vs_own=paired(f,predictions[role]['whole'],predictions[role]['own'],'availability-frozen:'+role,1000))
    dev=data['development']
    for seed in SEEDS:
        fit,tune=patient_partition(dev.subject_id,seed)
        models={}
        for mode in ('own','whole','legacy_masks'):
            scaler=StandardScaler().fit(features['development'][mode][fit])
            x=scaler.transform(features['development'][mode])
            w=dev.loc[fit].subject_id.map(dev.loc[fit].subject_id.value_counts()).rdiv(1.).to_numpy(copy=True)
            w/=w.mean()
            best=None
            for alpha in (1.,10.,100.):
                model=Ridge(alpha=alpha).fit(x[fit],(dev.truth-dev.current).to_numpy()[fit],sample_weight=w)
                pred=dev.current.to_numpy()[tune]+model.predict(x[tune])
                mae=patient_equal_mae(dev.loc[tune],dev.truth.to_numpy()[tune],pred)
                if best is None or mae<best[0]: best=(mae,model,alpha)
            models[mode]=(scaler,best[1],best[2])
        evaluation={}
        for role in ('eicu_hospital','mimic'):
            f=data[role]
            pred={k:f.current.to_numpy()+m.predict(s.transform(features[role][k])) for k,(s,m,a) in models.items()}
            pred['persistence']=f.current.to_numpy()
            pred['frozen_whole']=predictions[role]['whole']
            evaluation[role]=dict(mae={k:patient_equal_mae(f,f.truth,y) for k,y in pred.items()},
                whole_comparisons={k:paired(f,pred['whole'],pred[k],f'availability-jepa:{role}:{seed}:{k}',1000) for k in ('own','legacy_masks','persistence','frozen_whole')})
        result=dict(seed=int(seed),alpha={k:m[2] for k,m in models.items()},evaluation=evaluation)
        report['seeds'].append(result)
        print(json.dumps(result),flush=True)
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
