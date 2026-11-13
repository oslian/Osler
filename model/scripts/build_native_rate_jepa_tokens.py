"""Materialize sparse new-event tokens on a hash-bound native schedule."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/native_rate_jepa_tokens_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    ref=json.loads(Path('artifacts/native_rate_jepa_schedule_20260911_v2.json').read_text())
    for name in ('native_events.parquet','native_schedule.npz'):
        if sha256(root/name)!=ref['binding'][str(root/name)]:raise ValueError('schedule changed')
    events=pd.read_parquet(root/'native_events.parquet');assets=load_release(Path('models'))
    with np.load(root/'native_schedule.npz') as archive:s={k:archive[k] for k in archive.files}
    tokens=[];treatments=[];lookup={};indices=[]
    for left,right in zip(s['group_ptr'][:-1],s['group_ptr'][1:]):
        ids=s['event_indices'][left:right];key=tuple(ids.tolist())
        if key not in lookup:
            token,treatment=encode_group(events.iloc[ids],assets.checkpoint['variables'],assets.scaler.medians,assets.scaler.scales,
                                         assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules']),assets.max_age_hours)
            lookup[key]=len(tokens);tokens.append(token);treatments.append(treatment)
        indices.append(lookup[key])
    token=np.asarray(tokens,np.float32);treatment=np.asarray(treatments,np.float64)
    if token.shape[1]!=assets.checkpoint['input_dim']:raise ValueError('input width mismatch')
    path=root/'native_tokens.npz'
    np.savez_compressed(path,input=token,event_values=treatment,query_ptr=s['query_ptr'],history_indices=np.asarray(indices,np.int64))
    n=len(assets.checkpoint['variables']);labs=token[:,n:2*n].any(axis=1);rates=np.isfinite(treatment[:,2])
    report=dict(complete=True,training_performed=False,runtime_modified=False,unique_groups=len(token),
                lab_groups=int(labs.sum()),rate_groups=int(rates.sum()),rate_only_groups=int((rates&~labs).sum()),
                empty_groups=int((~rates&~labs).sum()),input_width=token.shape[1],
                binding={str(p):sha256(p) for p in [path,root/'native_events.parquet',root/'native_schedule.npz',Path('whole_body_jepa/native_event_tokens.py'),Path(__file__)]})
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
