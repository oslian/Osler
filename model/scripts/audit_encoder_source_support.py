"""Fixed-weight source-supported input ablation for repaired JEPA."""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import torch

from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import patient_partition,paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import build_arrays,source_patient_support,restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    torch.set_num_threads(4)
    assets=load_release(Path('models'))
    variables=assets.checkpoint['variables']
    idx=variables.index('glucose')
    median=float(assets.scaler.medians[idx]);scale=float(assets.scaler.scales[idx])
    ids=assets.model.variable_module_ids.numpy()
    raw={db:pd.read_parquet(f'/private/tmp/availability_jepa_{db}_labs.parquet') for db in ('eicu','mimic')}
    c=duckdb.connect()
    starts=c.execute("SELECT stay_id,intime FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    c.close()
    origins={'mimic':dict(zip('mimic:'+starts.stay_id.astype(str),starts.intime)),
             'eicu':{s:pd.Timestamp('2100-01-01') for s in raw['eicu'].stay_id.unique()}}
    arrays,rows,data={},{},{}
    for role in ('development','eicu_hospital','mimic'):
        frame=pd.read_parquet(f'/private/tmp/availability_glucose_{role}.parquet')
        valid=(frame.fresh & frame.truth.notna()).to_numpy()
        data[role]=frame.loc[valid].reset_index(drop=True)
        db='mimic' if role=='mimic' else 'eicu'
        a,r=build_arrays(frame,raw[db],assets,origins[db],truthful=True)
        arrays[role],rows[role]=a,r[valid]
        print('built '+role,flush=True)
    report=dict(status='candidate_only',runtime_modified=False,causal_claim_allowed=False,
                protocol='ENCODER_SOURCE_SUPPORT_PROTOCOL_20260910.md',seeds=[],
                binding={str(p):sha256(p) for p in (Path(__file__),Path('whole_body_jepa/availability_inputs.py'),
                         Path('ENCODER_SOURCE_SUPPORT_PROTOCOL_20260910.md'))})
    destination=Path('/private/tmp/encoder_supported_weights_20260910')
    destination.mkdir(exist_ok=True)
    for seed in (7,19,31):
        fit,_=patient_partition(data['development'].subject_id,seed)
        support=source_patient_support(arrays['development'],rows['development'],data['development'].subject_id,fit)
        if support[idx]<20:raise ValueError('insufficient target support')
        entry=dict(seed=seed,source_patient_support=dict(zip(variables,map(int,support))),evaluation={},
                   changed_variables={},checkpoints={})
        for role in ('eicu_hospital','mimic'):
            f=data[role];r=rows[role]
            a=arrays[role]
            inputs={'full':a,'seen':restrict_input_support(a,support>=1,ids),
                    'supported20':restrict_input_support(a,support>=20,ids)}
            entry['changed_variables'][role]={mode:[v for i,v in enumerate(variables)
                if any(np.any(a[k][:,i]!=inputs[mode][k][:,i]) for k in ('current','mask','ages'))]
                for mode in ('seen','supported20')}
            metrics={}
            for arm in ('whole_unbounded','whole_unbounded_balanced'):
                saved=torch.load(f'/private/tmp/encoder_repair_weights_20260910/{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                if saved['value_delta_activation']!='identity':raise ValueError('wrong candidate activation')
                if role=='eicu_hospital':
                    path=destination/f'{arm}_{seed}.pt'
                    torch.save(dict(saved,input_policy='source_seen_only',variables=variables,
                                    allowed_variables=(support>=1).tolist(),source_patient_support=support.tolist()),path)
                    entry['checkpoints'][arm]=dict(path=str(path),sha256=sha256(path),
                        original_sha256=sha256(Path(f'/private/tmp/encoder_repair_weights_20260910/{arm}_{seed}.pt')))
                model=assets.model
                model.value_delta_activation=torch.nn.Identity()
                model.load_state_dict(saved['state_dict'])
                predictions={k:predict(model,v,r,idx,median,scale) for k,v in inputs.items()}
                metrics[arm]=dict(mae={k:patient_equal_mae(f,f.truth,v) for k,v in predictions.items()},
                    comparisons={k:paired(f,predictions[k],predictions['full'],f'support:{seed}:{role}:{arm}:{k}',1000)
                                 for k in ('seen','supported20')},
                    min_max={k:[float(v.min()),float(v.max())] for k,v in predictions.items()},
                    negative_predictions={k:int(np.sum(v<0)) for k,v in predictions.items()})
            entry['evaluation'][role]=metrics
        report['seeds'].append(entry)
        print('completed seed '+str(seed),flush=True)
    report['complete']=True
    Path('artifacts/encoder_source_support_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
