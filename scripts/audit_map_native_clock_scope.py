"""Quantify clipped time coordinates on current MAP histories, without fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    out=Path('artifacts/map_native_clock_scope_20260913.json')
    if out.exists():raise ValueError('fresh output required')
    parent=Path('artifacts/map_native_query_history_complete_20260912.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed input')
    q=pd.read_parquet(root/'queries.parquet')
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q));n=raw.query['current'].shape[1]
    transforms=np.load(root/'neural_transforms.npz')
    eligible=np.load(root/'labels.npz')['paired'][:,0]
    clock=raw.input[:,-1]
    stats=[]
    for seed in (7,19,31,43,59,71,83):
        active=raw.input[:,n:2*n][:,transforms[f'{seed}_allowed'].astype(bool)].astype(bool).any(1)
        active|=np.isfinite(raw.event_values[:,2])
        repeated=[];saturated=[];transitions=[]
        for seq in raw.sequences:
            times=clock[seq[active[seq]]]
            d=np.diff(times)
            transitions.append(len(d));repeated.append(int((d==0).sum()))
            saturated.append(bool(len(times)>1 and np.all(times==times[0]) and times[0] in (0.,1.)))
        frame=pd.DataFrame({'patient':q.uniquepid,'zero_delta':repeated,'transitions':transitions,'fully_saturated':saturated})
        frame['zero_delta_fraction']=np.divide(frame.zero_delta,frame.transitions,
            out=np.zeros(len(frame)),where=frame.transitions>0)
        roles=np.load(root/f'partition_{seed}.npz');summary={}
        for role in ('fit','tune','evaluation','hospital'):
            f=frame.loc[roles[role]&eligible]
            summary[role]=f.groupby('patient')[['zero_delta_fraction','fully_saturated']].mean().mean().to_dict()
        stats.append(dict(seed=seed,roles=summary))
    result=dict(complete=True,results=stats,training_performed=False,runtime_modified=False,
        token_fraction_at_zero=float(np.mean(clock[1:]==0)),token_fraction_at_one=float(np.mean(clock[1:]==1)),
        limitation='Equal encoded clocks may include float32 rounding; saturation is not accuracy causation.',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),root/'neural_transforms.npz')})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['results'][0],indent=2))
    print('token bounds',result['token_fraction_at_zero'],result['token_fraction_at_one'])


if __name__=='__main__':main()
