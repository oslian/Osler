"""Exact replay of matched reconstruction sampler exposure."""
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    parent=Path('artifacts/native_reconstruction_sampling_20260913.json')
    report=json.loads(parent.read_text())
    for name,h in report['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('training input changed')
    out=Path('artifacts/native_reconstruction_exposure_20260913.json')
    if out.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support/multitarget_sampling_7')
    saved=torch.load(root/'samples.pt',weights_only=False);targets=saved['targets']
    keys=[k for k in saved['data']['fit'] if k[1] in targets]
    bytarget={c:[k for k in keys if k[1]==c] for c in targets}
    prob=np.array([saved['frequencies']['fit'][k] for k in keys],float);prob/=prob.sum()
    variables=torch.load(root/'balanced.pt',weights_only=False)['variables'];results={}
    for arm in ('event_weighted','balanced'):
        draw=np.random.default_rng(7);count={k:0 for k in keys};epochs={}
        for epoch in range(1,9):
            for _ in range(64):
                if arm=='event_weighted':key=keys[int(draw.choice(len(keys),p=prob))]
                else:
                    c=int(draw.choice(targets));key=bytarget[c][int(draw.integers(len(bytarget[c])))]
                count[key]+=1
            epochs[epoch]=dict(steps=epoch*64,available_pairs=len(keys),seen_pairs=sum(v>0 for v in count.values()),
                targets={variables[c]:dict(draws=sum(count[k] for k in bytarget[c]),
                    seen_patients=sum(count[k]>0 for k in bytarget[c]),available_patients=len(bytarget[c])) for c in targets})
        selected=report['selection'][arm]['selected_epoch']
        results[arm]=dict(selected_epoch=selected,selected=epochs[selected] if selected else None,final=epochs[8])
    out.write_text(json.dumps(dict(complete=True,results=results,training_performed=False,runtime_modified=False,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__))}),indent=2)+'\n')
    print(json.dumps({a:{stage:dict(seen_pairs=r[stage]['seen_pairs'],total_pairs=len(keys),
        untrained_targets=[t for t,v in r[stage]['targets'].items() if v['draws']==0],
        median_target_draws=float(np.median([v['draws'] for v in r[stage]['targets'].values()])))
        for stage in ('selected','final')} for a,r in results.items()},indent=2))


if __name__=='__main__':main()
