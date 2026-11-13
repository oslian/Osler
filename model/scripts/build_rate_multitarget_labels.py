"""Measured auxiliary labels on unchanged rate-query patient roles."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');dest=root/'multitarget_labels.npz'
    out=Path('artifacts/rate_multitarget_labels_20260911.json')
    if out.exists() or dest.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/rate_anchor_body_inputs_20260911.json').read_text())
    if sha256(root/'body_labs.parquet')!=ref['binding'][str(root/'body_labs.parquet')]:raise ValueError('raw labels changed')
    old=np.load(root/'labels.npz');valid=old['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    labels=future_pairs(q,pd.read_parquet(root/'body_labs.parquet'),variables,horizon=6,tolerance=1,max_age=6)
    k=variables.index('potassium')
    for name in ('current','future','actual_delta_t','measurement_age'):
        np.testing.assert_allclose(labels[name][:,k],old[name][valid,0],atol=1e-5,rtol=1e-6)
    prior=json.loads(Path('artifacts/rate_anchored_potassium_20260911.json').read_text());roles=[]
    for entry in prior['seeds']:
        path=root/f"partition_{entry['seed']}.npz"
        if sha256(path)!=entry['partition_sha256']:raise ValueError('roles changed')
        roles.append({a:b[valid] for a,b in dict(np.load(path)).items()})
    support={}
    for i,v in enumerate(variables):
        pair=labels['paired'][:,i];support[v]=dict(pairs=int(pair.sum()),patients=int(q.loc[pair,'uniquepid'].nunique()),roles={})
        for role in ('fit','tune','evaluation','hospital'):
            support[v]['roles'][role]=[int(q.loc[pair&r[role],'uniquepid'].nunique()) for r in roles]
    np.savez_compressed(dest,**labels,variables=np.asarray(variables))
    report=dict(complete=True,training_performed=False,runtime_modified=False,query_count=len(q),potassium_reproduced=True,
                support=support,policy=dict(horizon=6,tolerance=1,max_age=6),
                binding={str(p):sha256(p) for p in (Path(__file__),dest,root/'body_labs.parquet',root/'queries.parquet',root/'labels.npz')})
    out.write_text(json.dumps(report,indent=2)+'\n')
    print({v:s for v,s in support.items() if s['pairs']})


if __name__=='__main__':main()
