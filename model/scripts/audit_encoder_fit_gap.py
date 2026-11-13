"""Locate own/whole failure before external transfer; no fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from whole_body_jepa.runtime import load_release
from whole_body_jepa.availability_inputs import build_arrays,restrict_input_support
from whole_body_jepa.multisource_intervals import sha256
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import patient_partition,paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    torch.set_num_threads(4)
    source=Path('artifacts/encoder_budget_20260910.json')
    original=json.loads(source.read_text())
    if not original.get('complete'):raise ValueError('incomplete source experiment')
    paths=[Path('/private/tmp/availability_glucose_development.parquet'),Path('/private/tmp/availability_jepa_eicu_labs.parquet')]
    for p in paths:
        if sha256(p)!=original['data_binding'][str(p)]:raise ValueError('changed source data')
    frame=pd.read_parquet(paths[0]);raw=pd.read_parquet(paths[1])
    valid=(frame.fresh & frame.truth.notna()).to_numpy()
    f=frame.loc[valid].reset_index(drop=True)
    assets=load_release(Path('models'))
    origins={s:pd.Timestamp('2100-01-01') for s in raw.stay_id.unique()}
    arrays={};rows={}
    for view in ('own','whole'):
        arrays[view],r=build_arrays(frame,raw,assets,origins,target_only=view=='own',truthful=True)
        rows[view]=r[valid]
    idx=assets.checkpoint['variables'].index('glucose')
    median=float(assets.scaler.medians[idx]);scale=float(assets.scaler.scales[idx])
    report=dict(status='diagnostic_only',runtime_modified=False,promotion_allowed=False,
                source_tune_is_selection_data=True,fit_metrics_are_in_sample=True,
                binding={str(p):sha256(p) for p in [source,Path(__file__),*paths]},seeds=[])
    for s in original['seeds']:
        seed=s['seed'];fit,tune=patient_partition(f.subject_id,seed)
        pred={};entry=dict(seed=seed,roles={})
        for view in ('own','whole'):
            arm=view+'_unbounded';meta=s['training'][arm];p=Path(meta['checkpoint'])
            if sha256(p)!=meta['sha256']:raise ValueError('changed checkpoint')
            saved=torch.load(p,map_location='cpu',weights_only=False)
            model=assets.model;model.load_state_dict(saved['state_dict'])
            model.value_delta_activation=torch.nn.Identity()
            model.belief_input_mode='observed'
            a=restrict_input_support(arrays[view],saved['allowed_variables'],model.variable_module_ids.numpy())
            pred[view]=predict(model,a,rows[view],idx,median,scale)
            measured=patient_equal_mae(f.loc[tune],f.truth.to_numpy()[tune],pred[view][tune])
            if not np.isclose(measured,meta['tune_mae'],rtol=0,atol=1e-5):raise ValueError('source tune did not reproduce')
        for name,keep in [('fit',fit),('tune',tune)]:
            subset=f.loc[keep].reset_index(drop=True)
            entry['roles'][name]=dict(patients=int(subset.subject_id.nunique()),rows=len(subset),
                mae={view:patient_equal_mae(subset,subset.truth,y[keep]) for view,y in pred.items()},
                whole_vs_own=paired(subset,pred['whole'][keep],pred['own'][keep],f'fit-gap:{seed}:{name}',1000))
        report['seeds'].append(entry)
    report['complete']=True
    Path('artifacts/encoder_fit_gap_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
