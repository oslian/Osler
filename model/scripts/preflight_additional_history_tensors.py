"""Fixed prior parity cohort: exact training tensor and storage preflight."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import transform_example
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',action='store_true')
    args=parser.parse_args()
    root=Path('data/research_recovery/additional_development_domains_20260921')
    suffix='cache' if args.cache else 'preflight'
    output=Path(f'artifacts/native_reconstruction_additional_tensor_{suffix}_20260922.json')
    if output.exists():raise ValueError('Existing preflight')
    sr=Path('artifacts/native_reconstruction_additional_domain_scaling_20260922.json')
    report=json.loads(sr.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed scaling dependency')
    sp=root/'scaling_20260922.pt'
    if sha256(sp)!=report['private_sha256']:raise ValueError('Changed scaler')
    scaling=torch.load(sp,weights_only=False)
    variables=scaling['variables'];n=len(variables)
    centers={p:i for i,p in enumerate(scaling['people'])}
    cp=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)
    module_ids=cp['state_dict']['variable_module_ids'].numpy()
    pilot=root/'masked_history_pilot/index.pt'
    pr=Path('artifacts/native_reconstruction_additional_masked_history_pilot_20260921.json')
    parent=json.loads(pr.read_text())
    if sha256(pilot)!=parent['private_index_sha256']:raise ValueError('Changed pilot')
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed pilot dependency')
    indices={kind:{s['hospitalid']:s for s in torch.load(root/kind/'event_index.pt',weights_only=False)}
             for kind in ('labs','bedside')}
    torch.set_num_threads(2)
    cache_path=root/'pilot_histories_20260922.sqlite'
    if args.cache and cache_path.exists():raise ValueError('Existing pilot cache')
    cache=ReconstructionCache(cache_path,report['bindings']) if args.cache else None
    sizes=[];lengths=[];people=set();unsupported=set();transformed=0
    for shard in torch.load(pilot,weights_only=False):
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Changed decisions')
        q=pd.read_parquet(shard['path']);q=q.loc[q.status.eq('accepted')]
        if q.empty:continue
        events=[]
        for kind in indices:
            source=indices[kind][shard['hospitalid']]
            if sha256(Path(source['path']))!=source['sha256']:raise ValueError('Changed events')
            events.append(pd.read_parquet(source['path'],filters=[('stay_id','in',q.stay_id.unique().tolist())]))
        events=pd.concat(events,ignore_index=True)
        for stay,requests in q.groupby('stay_id'):
            requests=requests.sort_values(['t','variable']).reset_index(drop=True)
            count=0
            for e in reconstruction_examples(events.loc[events.stay_id.eq(stay)],requests,
                    variables,np.zeros(n),np.ones(n),module_ids,len(cp['modules'])):
                row=requests.iloc[e['query_position']];col=e['target_index']
                if variables[col]!=row.variable:continue
                if cache is not None:
                    cache.add(row.uniquepid,e)
                    restored=cache.get(row.uniquepid,col)
                    for key in ('tokens','event_hours'):
                        assert torch.equal(e[key],restored[key])
                    for key in ('truth_normalized','anchor_hour','target_index'):
                        assert e[key]==restored[key]
                expected=scaling['centers'][centers[row.uniquepid],col]
                np.testing.assert_allclose(e['truth_normalized'],expected,rtol=1e-6,atol=1e-6)
                mask=e['tokens'][:,n:2*n].bool()
                forbidden={row.variable}|related_siblings(row.variable)
                assert not any(mask[:,i].any() for i,v in enumerate(variables) if v in forbidden)
                assert len(mask)==row.independent_event_groups
                for scaler in scaling['scalers']:
                    absent=mask.any(0).numpy() & (scaler['counts']==0)
                    unsupported.update(v for i,v in enumerate(variables) if absent[i])
                    if absent.any() or scaler['counts'][col]==0:continue
                    converted=transform_example(e,scaler)
                    assert torch.isfinite(converted['tokens']).all()
                    assert torch.equal(converted['tokens'][:,n:],e['tokens'][:,n:])
                    transformed+=1
                sizes.append(sum(e[k].numel()*e[k].element_size() for k in ('tokens','event_hours')))
                lengths.append(len(mask));people.add(row.uniquepid);count+=1
            if count!=len(requests):raise ValueError('Lost accepted examples')
        if cache is not None:cache.finish_shard(shard['hospitalid'])
    if cache is not None:cache.close()
    result=dict(patients=len(people),examples=len(sizes),fold_transforms=transformed,
        unsupported_observed_variables=sorted(unsupported),tensor_bytes=sum(sizes),
        mean_tensor_bytes=float(np.mean(sizes)),max_history_events=max(lengths),
        median_history_events=float(np.median(lengths)),
        illustrative_full_tensor_gib=float(np.mean(sizes)*report['accepted_labels']/1024**3),
        limitation='Fixed 38-person parity pilot, not representative storage estimate or model accuracy. No checkpoint weights used; module mapping only. No full-history cache or training yet.',
        runtime_modified=False,promotion_allowed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),sr,sp,pr,pilot,Path('scripts/encoder_fold_scaling.py'),Path('whole_body_jepa/native_reconstruction_examples.py'))})
    if args.cache:
        result.update(cache_bytes=cache_path.stat().st_size,cache_sha256=sha256(cache_path),
            cache_roundtrip_exact=True)
        result['bindings']['whole_body_jepa/reconstruction_cache.py']=sha256(Path('whole_body_jepa/reconstruction_cache.py'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
