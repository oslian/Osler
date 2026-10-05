"""Actual other-target outcomes after shared-future-frozen training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=Path('artifacts/frozen_future_retention_20260911.json')
    if path.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_frozen_future_20260911.json').read_text())
    for n,h in ref['binding'].items():
        if sha256(Path(n))!=h:raise ValueError('changed input')
    with np.load(root/'labels.npz') as f:valid=f['paired'][:,0]
    with np.load(root/'multitarget_labels.npz') as f:labels={k:f[k] for k in f.files}
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    vs=labels['variables'].tolist();col=vs.index('glucose')
    report=dict(runtime_modified=False,binding=ref['binding'].copy(),seeds=[])
    report['binding'][str(Path(__file__))]=sha256(Path(__file__))
    for entry in ref['seeds']:
        seed=entry['seed'];item=dict(seed=seed,roles={})
        with np.load(root/f'glucose_frozen_future_all_whole_{seed}.npz') as f:final=f['final']
        with np.load(root/f'glucose_retention_predictions_{seed}.npz') as f:old=f['original']
        with np.load(root/f'partition_{seed}.npz') as f:parts={k:f[k][valid] for k in ('evaluation','hospital')}
        for role,selection in parts.items():
            result={}
            for j,v in enumerate(vs):
                m=selection&labels['paired'][:,col]&labels['paired'][:,j];frame=q.loc[m].copy();frame['truth']=labels['future'][m,j]
                if frame.subject_id.nunique()<30:continue
                result[v]=dict(mae=patient_equal_mae(frame,frame.truth,final[m,j]),original=patient_equal_mae(frame,frame.truth,old[m,j]),comparison=paired(frame,final[m,j],old[m,j],f'frozenfuture:{seed}:{role}:{v}',2000))
            item['roles'][role]=result
        report['seeds'].append(item)
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
