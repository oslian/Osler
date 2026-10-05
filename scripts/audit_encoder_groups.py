"""Localize fixed whole-context sensitivity after input and amplitude repairs."""
from pathlib import Path
import argparse
import json
import duckdb
import numpy as np
import pandas as pd
import torch

from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import build_arrays,restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def neutralize_component(arrays,mask,component):
    """Diagnostic intervention, never an observation or runtime imputation."""
    result={k:v.copy() for k,v in arrays.items()}
    n=arrays['current'].shape[1]
    if component=='value':
        result['current'][:,mask]=0.  # Checkpoint median in normalized space.
        result['input'][:,:n]=result['current']
    elif component=='age':
        result['ages'][:,mask]=np.where(arrays['mask'][:,mask]>0,3./24.,arrays['ages'][:,mask])
        result['input'][:,2*n:3*n]=result['ages']
    else:raise ValueError('unknown component')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source=Path('artifacts/encoder_budget_20260910.json')
    trained=json.loads(source.read_text())
    if not trained.get('complete'):raise ValueError('training incomplete')
    for name,digest in trained['data_binding'].items():
        if sha256(Path(name))!=digest: raise ValueError('input changed')
    assets=load_release(Path('models'))
    variables=assets.checkpoint['variables'];index=variables.index('glucose')
    ids=assets.model.variable_module_ids.numpy()
    groups={name:ids==i for i,name in enumerate(assets.checkpoint['modules'])}
    groups['all_non_target']=np.ones(len(variables),bool)
    for mask in groups.values():mask[index]=False
    interventions={f'{name}:{component}':(mask,component) for name,mask in groups.items()
                   for component in (('value','age') if args.components else ('drop',))}
    with duckdb.connect() as con:
        starts=con.execute("SELECT stay_id,intime FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    origins=dict(zip('mimic:'+starts.stay_id.astype(str),starts.intime))
    data={};arrays={};rows={}
    for role,db in (('eicu_hospital','eicu'),('mimic','mimic')):
        f=pd.read_parquet(f'/private/tmp/availability_glucose_{role}.parquet')
        raw=pd.read_parquet(f'/private/tmp/availability_jepa_{db}_labs.parquet')
        start=origins if db=='mimic' else {s:pd.Timestamp('2100-01-01') for s in raw.stay_id.unique()}
        valid=(f.fresh & f.truth.notna()).to_numpy()
        a,r=build_arrays(f,raw,assets,start,truthful=True)
        data[role]=f.loc[valid].reset_index(drop=True);arrays[role]=a;rows[role]=r[valid]
    report=dict(status='diagnostic_only',promotion_allowed=False,runtime_modified=False,
        binding={str(p):sha256(p) for p in (source,Path(__file__),Path('ENCODER_COMPONENT_PROTOCOL_20260910.md' if args.components else 'ENCODER_GROUP_PROTOCOL_20260910.md'))},
        groups={k:[v for v,b in zip(variables,m) if b] for k,m in groups.items()},seeds=[])
    for entry in trained['seeds']:
        seed=entry['seed'];path=Path(entry['training']['whole_unbounded']['checkpoint'])
        if sha256(path)!=entry['training']['whole_unbounded']['sha256']:raise ValueError('weights changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['variables']!=variables or saved['value_delta_activation']!='identity':raise ValueError('contract mismatch')
        model=assets.model;model.value_delta_activation=torch.nn.Identity();model.load_state_dict(saved['state_dict'])
        allowed=np.asarray(saved['allowed_variables'],bool)
        result=dict(seed=seed,roles={})
        for role,f in data.items():
            a=restrict_input_support(arrays[role],allowed,ids);r=rows[role]
            full=predict(model,a,r,index,float(assets.scaler.medians[index]),float(assets.scaler.scales[index]))
            with np.load(path.parent/f'predictions_{role}_{seed}.npz') as cache:
                np.testing.assert_allclose(full,cache['whole_unbounded'],rtol=0,atol=1e-5)
            result['roles'][role]=dict(full_mae=patient_equal_mae(f,f.truth,full),groups={})
            for key,(mask,component) in interventions.items():
                name=key if args.components else key.rsplit(':',1)[0]
                changed=bool(a['mask'][:,mask].any())
                altered=restrict_input_support(a,allowed & ~mask,ids) if component=='drop' else neutralize_component(a,mask,component)
                for field in ('current','mask','ages'):
                    np.testing.assert_array_equal(a[field][:,index],altered[field][:,index])
                prediction=predict(model,altered,r,index,float(assets.scaler.medians[index]),float(assets.scaler.scales[index])) if changed else full.copy()
                result['roles'][role]['groups'][name]=dict(changed=changed,
                    mae=patient_equal_mae(f,f.truth,prediction),mean_absolute_shift=float(np.abs(prediction-full).mean()),
                    paired=paired(f,prediction,full,f'group:{seed}:{role}:{name}',1000))
        report['seeds'].append(result)
        print('completed group audit seed '+str(seed),flush=True)
    report['complete']=True
    Path('artifacts/encoder_components_20260910.json' if args.components else 'artifacts/encoder_groups_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
