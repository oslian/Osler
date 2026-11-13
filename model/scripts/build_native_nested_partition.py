"""Separate baseline fitting/selection from residual development patients."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/native_nested_partition_20260913.json')
    private=root/'nested_partition_7.npz'
    if output.exists() or private.exists():raise ValueError('fresh outputs required')
    q=pd.read_parquet(root/'queries.parquet');labels=dict(np.load(root/'labels.npz'))
    roles=dict(np.load(root/'partition_7.npz'));rng=np.random.default_rng(20260913)
    chosen={}
    for role,count in (('fit',64),('tune',32),('evaluation',64),('hospital',64)):
        groups=[np.asarray(g.index) for _,g in q.loc[roles[role]&labels['paired'][:,0]].groupby('uniquepid',sort=True)]
        chosen[role]=np.array([int(rng.choice(groups[i])) for i in rng.choice(len(groups),count,replace=False)])
    excluded=set(q.loc[np.concatenate(list(chosen.values())),'uniquepid'])
    new={k:v.copy() for k,v in roles.items()}
    for role in ('fit','tune'):new[role]&=~q.uniquepid.isin(excluded).to_numpy()
    assert not set(q.loc[new['fit']|new['tune'],'uniquepid'])&excluded
    assert not set(q.loc[new['fit'],'uniquepid'])&set(q.loc[new['tune'],'uniquepid'])
    np.savez_compressed(private,**new,**{f'residual_{k}':v for k,v in chosen.items()})
    output.write_text(json.dumps(dict(complete=True,patients={k:int(q.loc[v,'uniquepid'].nunique()) for k,v in new.items()},
        residual_patients={k:len(v) for k,v in chosen.items()},baseline_residual_patient_overlap=0,
        training_performed=False,runtime_modified=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),root/'queries.parquet',root/'labels.npz',root/'partition_7.npz',private)}),indent=2)+'\n')
    print(json.loads(output.read_text())['patients'])


if __name__=='__main__':main()
