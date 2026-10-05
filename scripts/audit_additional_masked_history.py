"""Exact availability replay and masked-history acceptance of frozen anchors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.reconstruction_support import reconstruction_support


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true',help='One hash-fixed patient/component, reference parity before full replay')
    args=parser.parse_args()
    root=Path('data/research_recovery/additional_development_domains_20260921')
    suffix='pilot' if args.pilot else 'full'
    output=Path(f'artifacts/native_reconstruction_additional_masked_history_{suffix}_20260921.json')
    folder=root/f'masked_history_{suffix}'
    if output.exists() or folder.exists():
        raise ValueError('Existing audit outputs')
    parent=Path('artifacts/native_reconstruction_additional_domain_anchors_20260921.json')
    report=json.loads(parent.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed anchor dependency')
    qp=root/'anchors/queries.parquet'
    if sha256(qp)!=report['private_queries_sha256']:raise ValueError('Changed fixed anchors')
    queries=pd.read_parquet(qp)
    population_pairs=len(queries);population_people=queries.uniquepid.nunique()
    if args.pilot:
        people=queries[['uniquepid','component']].drop_duplicates().copy()
        people['priority']=people.uniquepid.map(lambda p:hashlib.sha256(('20260921-parity:'+p).encode()).hexdigest())
        selected=set(people.sort_values('priority').drop_duplicates('component').uniquepid)
        queries=queries.loc[queries.uniquepid.isin(selected)].copy()
    variables=torch.load('models/whole_body_joint_jepa.pt',weights_only=False)['variables']
    indices={};paths=[Path(__file__),parent,qp,Path('whole_body_jepa/reconstruction_support.py'),
        Path('whole_body_jepa/native_reconstruction_examples.py'),Path('whole_body_jepa/native_history_schedule.py'),
        Path('whole_body_jepa/native_event_tokens.py'),Path('whole_body_jepa/native_body_reconstruction.py'),Path('whole_body_jepa/nowcast.py')]
    for kind in ('labs','bedside'):
        ip=root/kind/'event_index.pt';paths.append(ip)
        index=torch.load(ip,weights_only=False)
        indices[kind]={item['hospitalid']:item for item in index}
    folder.mkdir();torch.set_num_threads(2)
    started=time.monotonic();shards=[];totals=[];parity_pairs=0
    for number,(site,q) in enumerate(queries.groupby('hospitalid',sort=True)):
        events=[]
        stays=q.stay_id.unique().tolist()
        for kind in indices:
            item=indices[kind][site]
            if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Changed event shard')
            events.append(pd.read_parquet(item['path'],filters=[('stay_id','in',stays)]))
        events=pd.concat(events,ignore_index=True)
        by_stay={stay:g for stay,g in events.groupby('stay_id',sort=False)}
        results=[]
        for stay,requests in q.groupby('stay_id',sort=False):
            requests=requests.sort_values(['t','variable']).reset_index(drop=True)
            history=by_stay[stay]
            decision=reconstruction_support(history,requests,variables)
            if args.pilot:
                n=len(variables)
                reference=list(reconstruction_examples(history,requests,variables,np.zeros(n),np.ones(n),np.arange(n),n))
                expected={(e['query_position'],variables[e['target_index']]):e for e in reference}
                for i,row in enumerate(requests.itertuples()):
                    sample=expected.get((i,row.variable))
                    if (decision.iloc[i].status=='accepted') != (sample is not None):
                        raise ValueError('Reference acceptance mismatch')
                    if sample is not None:
                        mask=sample['tokens'][:,n:2*n].bool()
                        if (int(mask.any(0).sum())!=decision.iloc[i].independent_variables
                                or len(mask)!=decision.iloc[i].independent_event_groups):
                            raise ValueError('Reference mask support mismatch')
                    parity_pairs+=1
            results.append(pd.concat([requests,decision],axis=1))
        result=pd.concat(results,ignore_index=True)
        dest=folder/f'site_{number:03d}.parquet';result.to_parquet(dest,index=False)
        shards.append(dict(path=str(dest),sha256=sha256(dest),hospitalid=site))
        totals.append(result[['uniquepid','variable','component','status']])
    frame=pd.concat(totals,ignore_index=True)
    if len(frame)!=len(queries) or frame.duplicated(['uniquepid','variable']).any():
        raise ValueError('Lost or repeated anchor')
    counts={}
    for target,g in frame.groupby('variable'):
        accepted=g.loc[g.status.eq('accepted')]
        domains=accepted.groupby('component').uniquepid.nunique()
        counts[target]=dict(proposed=len(g),accepted=len(accepted),
            reasons={str(k):int(v) for k,v in g.status.value_counts().items()},
            accepted_components=len(domains),accepted_components_at_least30=int((domains>=30).sum()))
    index_path=folder/'index.pt';torch.save(shards,index_path)
    summary=dict(pilot=args.pilot,patients=int(frame.uniquepid.nunique()),pairs=len(frame),
        accepted_pairs=int(frame.status.eq('accepted').sum()),
        rejection_counts={str(k):int(v) for k,v in frame.status.value_counts().items()},
        reference_parity_pairs=parity_pairs,seconds=time.monotonic()-started,
        full_population_pairs=population_pairs,full_population_people=int(population_people))
    output.write_text(json.dumps(dict(summary=summary,targets=counts,
        bindings={str(p):sha256(p) for p in paths},private_index_sha256=sha256(index_path),
        runtime_modified=False,promotion_allowed=False,roles_assigned=False,
        limitation='Same source-version replay, <=1h age and >anchor-48h history; all target/sibling observations removed. Pilot uses one hash-fixed patient/component and is NOT a population acceptance estimate. Full mode checks every frozen anchor. No fit-only vocabulary or model/scaler training yet; history availability is not predictive efficacy.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
