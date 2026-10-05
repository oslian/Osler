"""Compare inherited snapshot support with actual native history support."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_event_encoder import native_arrays
from whole_body_jepa.availability_inputs import source_patient_support
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_magnesium_matched_20260911.json')
    ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():
        assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_support_contract_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    base=dict(np.load(root/'body_arrays.npz'));native=dict(np.load(root/'native_tokens.npz'))
    labels=dict(np.load(root/'multitarget_labels.npz'));variables=list(labels['variables'])
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    arrays,rows,_=native_arrays(base,native,len(variables),base['presence'].shape[1])
    eligible=labels['paired'][:,variables.index('magnesium')]&labels['paired'][:,variables.index('glucose')]
    items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]&eligible
        old=source_patient_support(base,base['query_rows'],q.uniquepid,fit)
        new=source_patient_support(arrays,rows,q.uniquepid,fit)
        items.append(dict(seed=seed,snapshot_patients=old.tolist(),native_patients=new.tolist(),
            wrongly_excluded=[v for j,v in enumerate(variables) if old[j]==0 and new[j]>0],
            allowed_but_no_native=[v for j,v in enumerate(variables) if old[j]>0 and new[j]==0],
            native_supported_variables=int((new>0).sum()),
            native_supported_30patients=int((new>=30).sum())))
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,seeds=items,variables=variables,
        binding={str(p):sha256(p) for p in (parent,Path(__file__))}),indent=2)+'\n')
    for s in items:print({k:v for k,v in s.items() if not k.endswith('_patients')})


if __name__=='__main__':main()
