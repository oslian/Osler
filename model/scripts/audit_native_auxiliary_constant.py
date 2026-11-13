"""Compare native auxiliary forecasts against a source-fit fixed delta."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=Path('artifacts/native_auxiliary_constant_20260911.json')
    if path.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_truth_auxiliary_20260911.json').read_text())
    for n,h in ref['binding'].items():
        if sha256(Path(n))!=h:raise ValueError('changed input')
    with np.load(root/'labels.npz') as f:valid=f['paired'][:,0]
    with np.load(root/'multitarget_labels.npz') as f:labels={k:f[k] for k in f.files}
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    vs=labels['variables'].tolist();col=vs.index('glucose')
    report=dict(runtime_modified=False,binding=ref['binding'].copy(),seeds=[])
    report['binding'][str(Path(__file__))]=sha256(Path(__file__))
    for entry in [x for x in ref['seeds'] if x['arm']=='real']:
        seed=entry['seed'];item=dict(seed=seed,roles={})
        with np.load(root/f'glucose_truth_auxiliary_predictions_real_{seed}.npz') as f:final=f['final']
        with np.load(root/f'partition_{seed}.npz') as f:fit=f['fit'][valid]&labels['paired'][:,col]
        offsets={}
        for j,v in enumerate(vs):
            m=fit&labels['paired'][:,j]
            if q.loc[m].subject_id.nunique()<30:continue
            d=labels['future'][m,j]-labels['current'][m,j]
            subjects=q.loc[m,'subject_id'];w=(1/subjects.map(subjects.value_counts())).to_numpy()
            order=np.argsort(d,kind='stable');offsets[v]=float(d[order[np.searchsorted(np.cumsum(w[order]),w.sum()/2)]])
        item['source_offsets']=offsets
        with np.load(root/f'partition_{seed}.npz') as f:parts={k:f[k][valid] for k in ('evaluation','hospital')}
        for role,selection in parts.items():
            result={}
            for j,v in enumerate(vs):
                m=selection&labels['paired'][:,col]&labels['paired'][:,j];frame=q.loc[m].copy();frame['truth']=labels['future'][m,j]
                if frame.subject_id.nunique()<30 or v not in offsets:continue
                old=labels['current'][:,j]+offsets[v]
                result[v]=dict(mae=patient_equal_mae(frame,frame.truth,final[m,j]),original=patient_equal_mae(frame,frame.truth,old[m]),comparison=paired(frame,final[m,j],old[m],f'auxconstant:{seed}:{role}:{v}',2000))
            item['roles'][role]=result
        report['seeds'].append(item)
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

