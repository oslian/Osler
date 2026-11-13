"""Fixed-budget independent-patient exposure plan, with restartable queues."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def draw_queue(keys,state,rng,count):
    if not keys or count<0:raise ValueError('Nonempty population and nonnegative count required')
    result=[]
    for _ in range(count):
        if state.get('position',0)>=len(state.get('order',[])):
            state['order']=rng.permutation(len(keys)).tolist();state['position']=0
        result.append(keys[state['order'][state['position']]])
        state['position']+=1
    return result


def main():
    out=Path('artifacts/native_reconstruction_full_exposure_preflight_20260918.json')
    if out.exists():raise ValueError('Existing preflight')
    rp=Path('artifacts/native_reconstruction_full_broad_samples_20260918.json')
    report=json.loads(rp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Source changed')
    ip=Path('data/research_recovery/broad_source_20260917/full_samples/index.pt')
    index=torch.load(ip,weights_only=False);variables=index['variables']
    targets=list(report['accepted']['fit']);groups={t:[]for t in targets}
    for s in index['shards']:
        if s['role']!='fit':continue
        p=Path(s['path'])
        if sha256(p)!=s['sha256']:raise ValueError('Changed sample')
        for c in torch.load(p,weights_only=False)['samples']:
            groups[variables[c]].append(s['patient'])
    rng=np.random.default_rng(19);result={}
    for t,keys in groups.items():
        state={};counts=Counter();epochs=[]
        for epoch in range(8):
            counts.update(draw_queue(keys,state,rng,536))
            epochs.append(len(counts))
        result[t]=dict(available=len(keys),draws=sum(counts.values()),unique_seen=len(counts),
            coverage=len(counts)/len(keys),minimum_exposure_including_unseen=min([counts[k]for k in keys]),
            maximum_exposure=max(counts.values()),cumulative_unique_by_epoch=epochs)
        if len(counts)!=min(len(keys),4288):raise ValueError('Queue diversity failure')
    out.write_text(json.dumps(dict(targets=result,bindings={str(p):sha256(p)for p in (Path(__file__),rp,ip)},
        limitation='Exposure preflight only, no model updates. Fixed budget cannot cover every dense-target patient.',runtime_modified=False),indent=2)+'\n')
    for t in ('glucose','lipase','cpk','troponin_i','fibrinogen'):print(t,result[t])


if __name__=='__main__':main()
