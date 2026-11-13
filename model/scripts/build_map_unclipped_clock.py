"""Restore raw event clocks while reproducing every existing MAP native token."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.native_history_schedule import conflict_checked_history_schedule
from whole_body_jepa.native_event_tokens import encode_group
from whole_body_jepa.native_clock import restore_clock
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards',type=int,nargs='+',default=list(range(60)))
    args=parser.parse_args()
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    private=root/'map_native_anchor_support'
    queries=pd.read_parquet(private/'queries.parquet')
    original=pd.read_parquet(root/'native_events.parquet');original['quality_valid']=True
    bedside=pd.read_parquet(root/'qualified_bedside_events.parquet');bedside['event_time']=bedside.available_at
    all_events=pd.concat([original,bedside],ignore_index=True)
    assets=load_release(Path('models'));n=len(assets.checkpoint['variables'])
    destination=private/'unclipped_clock_shards';destination.mkdir(exist_ok=True)
    for shard in args.shards:
        source=private/'native_query_shards'/f'shard_{shard:03d}.npz'
        output=destination/source.name
        report=Path(f'artifacts/map_unclipped_clock_{shard:03d}_20260913.json')
        if output.exists() or report.exists():raise ValueError('fresh shard required')
        parent=Path(f'artifacts/map_native_query_tokens_{shard:03d}_20260912.json')
        for name,h in json.loads(parent.read_text())['bindings'].items():
            if sha256(Path(name))!=h:raise ValueError(f'changed input: {name}')
        old=dict(np.load(source));q=queries.iloc[old['global_query_rows']].reset_index(drop=True)
        events,schedules=conflict_checked_history_schedule(all_events.loc[all_events.stay_id.isin(q.stay_id.unique())],q)
        lookup={};hours=[];ids=[];ptr=[0]
        for groups in schedules:
            for group in groups:
                key=tuple(group.tolist())
                if key not in lookup:
                    part=events.iloc[group];index=len(lookup)
                    token,treatment=encode_group(part,assets.checkpoint['variables'],assets.scaler.medians,
                        assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules']),assets.max_age_hours)
                    np.testing.assert_array_equal(token,old['input'][index])
                    np.testing.assert_array_equal(treatment,old['event_values'][index])
                    lookup[key]=index
                    hours.append((pd.Timestamp(part.event_time.iloc[0])-pd.Timestamp('2100-01-01')).total_seconds()/3600)
                ids.append(lookup[key])
            ptr.append(len(ids))
        np.testing.assert_array_equal(ids,old['history_indices'])
        np.testing.assert_array_equal(ptr,old['query_ptr'])
        if len(hours)!=len(old['input']):raise ValueError('token coverage mismatch')
        corrected=restore_clock(old['input'],np.arange(len(hours)),hours,n)
        keep=np.ones(corrected.shape[1],bool);keep[[3*n,corrected.shape[1]-1]]=False
        np.testing.assert_array_equal(corrected[:,keep],old['input'][:,keep])
        payload={**old,'input':corrected,'event_hours':np.asarray(hours,np.float64)}
        np.savez_compressed(output,**payload)
        paths=[parent,source,output,Path(__file__),Path('whole_body_jepa/native_clock.py')]
        item=dict(complete=True,shard=shard,queries=len(q),tokens=len(hours),old_tokens_reproduced=True,
            nonclock_exact=True,changed_clock_tokens=int(np.any(corrected!=old['input'],axis=1).sum()),
            training_performed=False,runtime_modified=False,bindings={str(p):sha256(p) for p in paths})
        report.write_text(json.dumps(item,indent=2)+'\n')
        print('complete',shard,len(q),len(hours),item['changed_clock_tokens'],flush=True)


if __name__=='__main__':main()
