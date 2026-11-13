"""Fixed-result feature-shift diagnostics; does not select or change a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.audit_event_aware_patient_state import patient_partition,paired


def main():
    path=Path('artifacts/availability_jepa_shift_20260910.json')
    if path.exists(): raise ValueError('refusing overwrite')
    report=json.loads(Path('artifacts/availability_jepa_20260910_v2.json').read_text())
    data={}
    cache={}
    for role in ('development','eicu_hospital','mimic'):
        f=pd.read_parquet(f'/private/tmp/availability_glucose_{role}.parquet')
        data[role]=f.loc[f.fresh & f.truth.notna()].reset_index(drop=True)
        cache[role]=np.load(f'/private/tmp/availability_jepa_{role}_features.npz')
    output={'no_model_change':True,'frozen_comparisons':{},'shift':[]}
    for role in ('eicu_hospital','mimic'):
        f=data[role]
        output['frozen_comparisons'][role]={
            'whole_vs_persistence':paired(f,cache[role]['prediction_whole'],f.current.to_numpy(),'availability-check:'+role,1000),
            'whole_vs_legacy_masks':paired(f,cache[role]['prediction_whole'],cache[role]['prediction_legacy_masks'],'availability-mask:'+role,1000)}
    for entry in report['seeds']:
        fit,_=patient_partition(data['development'].subject_id,entry['seed'])
        modes={}
        for mode in ('own','whole','legacy_masks'):
            scaler=StandardScaler().fit(cache['development'][mode][fit])
            x=np.abs(scaler.transform(cache['mimic'][mode]))
            q95=np.quantile(x,.95,axis=0)
            top=np.argsort(q95)[-5:][::-1]
            modes[mode]={'fraction_abs_z_above_8':float((x>8).mean()),
                         'top_columns_by_p95_abs_z':[{'column':int(i),'p95_abs_z':float(q95[i])} for i in top]}
        output['shift'].append({'seed':entry['seed'],'modes':modes})
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
