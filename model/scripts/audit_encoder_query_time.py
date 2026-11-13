"""Source-only fixed-weight response to requested future time; not a repair trial."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from whole_body_jepa.runtime import load_release
from whole_body_jepa.availability_inputs import build_arrays, restrict_input_support
from whole_body_jepa.multisource_intervals import sha256
from scripts.train_availability_encoder import predict
from scripts.audit_event_aware_patient_state import patient_partition, paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae


def main():
    torch.set_num_threads(4)
    source=Path('artifacts/encoder_source2250_64_20260910.json')
    report0=json.loads(source.read_text())
    paths=[Path('/private/tmp/availability_glucose_development.parquet'),Path('/private/tmp/availability_jepa_eicu_labs.parquet')]
    for p in paths:
        if sha256(p)!=report0['data_binding'][str(p)]:raise ValueError('input changed')
    frame=pd.read_parquet(paths[0]);raw=pd.read_parquet(paths[1])
    valid=(frame.fresh & frame.truth.notna()).to_numpy()
    f=frame.loc[valid].reset_index(drop=True)
    assets=load_release(Path('models'))
    origins={s:pd.Timestamp('2100-01-01') for s in raw.stay_id.unique()}
    report=dict(status='diagnostic_only',promotion_allowed=False,runtime_modified=False,
                external_data_used=False,retrospective_time_query_not_deployment_policy=True,
                binding={str(p):sha256(p) for p in [source,Path(__file__),*paths]},seeds=[],
                actual_hours_quantiles=np.quantile(f.actual_delta_t,[0,.25,.5,.75,1]).tolist(),
                fraction_more_than_15min_from_nominal=float((abs(f.actual_delta_t-6)>.25).mean()))
    for view in ('own','whole'):
        arrays,r=build_arrays(frame,raw,assets,origins,target_only=view=='own')
        rows=r[valid];idx=assets.checkpoint['variables'].index('glucose')
        for s in report0['seeds']:
            _,tune=patient_partition(f.subject_id,s['seed'])
            meta=s['training'][view+'_unbounded'];p=Path(meta['checkpoint'])
            if sha256(p)!=meta['sha256']:raise ValueError('checkpoint changed')
            saved=torch.load(p,map_location='cpu',weights_only=False)
            model=assets.model;model.load_state_dict(saved['state_dict'])
            model.value_delta_activation=torch.nn.Identity();model.belief_input_mode=saved['belief_input_mode']
            a=restrict_input_support(arrays,saved['allowed_variables'],model.variable_module_ids.numpy())
            predictions={}
            for name,h in [('nominal',np.full(len(f),6)),('at5h',np.full(len(f),5)),('at7h',np.full(len(f),7)),('actual',f.actual_delta_t.to_numpy())]:
                a['horizon'][rows]=h
                predictions[name]=predict(model,a,rows[tune],idx,float(assets.scaler.medians[idx]),float(assets.scaler.scales[idx]))
            sub=f.loc[tune].reset_index(drop=True)
            mae=patient_equal_mae(sub,sub.truth,predictions['nominal'])
            if not np.isclose(mae,meta['tune_mae'],atol=1e-5,rtol=0):raise ValueError('source tune reproduction failed')
            report['seeds'].append(dict(seed=s['seed'],view=view,nominal_mae=mae,
                actual_query_mae=patient_equal_mae(sub,sub.truth,predictions['actual']),
                actual_vs_nominal=paired(sub,predictions['actual'],predictions['nominal'],f'time:{view}:{s["seed"]}',1000),
                mean_absolute_5_to_7_response=float(np.mean(abs(predictions['at7h']-predictions['at5h'])))))
    report['complete']=True
    Path('artifacts/encoder_query_time_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
