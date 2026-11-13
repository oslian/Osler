"""Patient-paired descriptive bootstrap of source epoch7 versus16."""
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def interval(values, seed=20260920, repetitions=4000):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid deltas')
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,repetitions,100):
        size=min(100,repetitions-start)
        means.extend(x[rng.integers(len(x),size=(size,len(x)))].mean(axis=1).tolist())
    lo,hi=np.quantile(means,[.025,.975])
    return dict(patients=len(x),mean=float(x.mean()),lower=float(lo),upper=float(hi))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    a=root/'expanded_selection_replay/epoch7.pt';b=root/'continued_selection_replay/epoch16.pt'
    rp=Path('artifacts/native_reconstruction_continued_selection_20260920.json')
    previous=Path('artifacts/native_reconstruction_expanded_selection_20260919.json')
    out=Path('artifacts/native_reconstruction_continued_selection_pairs_20260920.json')
    if out.exists(): raise ValueError('Existing audit')
    r,old=[json.loads(p.read_text()) for p in (rp,previous)]
    if sha256(a)!=old['private_predictions'][str(a)] or sha256(b)!=r['private_predictions'][str(b)]:
        raise ValueError('Predictions changed')
    frames=[torch.load(p,weights_only=False)['rows'] for p in (a,b)]
    maps=[{(x['patient'],x['target']):x for x in rows} for rows in frames]
    if any(len(m)!=len(rows) for m,rows in zip(maps,frames)) or maps[0].keys()!=maps[1].keys():
        raise ValueError('Pair mismatch')
    grouped={s:{t:[] for t in r['trained_targets']} for s in ('original','additional','all')}
    for key,x in maps[0].items():
        y=maps[1][key]
        if x['truth']!=y['truth'] or x['scope']!=y['scope']: raise ValueError('Truth/scope changed')
        delta=abs(y['prediction']-y['truth'])-abs(x['prediction']-x['truth'])
        grouped[x['scope']][x['target']].append(delta);grouped['all'][x['target']].append(delta)
    result={s:{t:interval(v) for t,v in ts.items()} for s,ts in grouped.items()}
    for scope,ts in result.items():
        for t,x in ts.items():
            expected=r['epochs']['16'][scope]['targets'][t]['mae']-r['epochs']['7'][scope]['targets'][t]['mae']
            if not np.isclose(x['mean'],expected,atol=1e-10,rtol=0): raise ValueError('Score mismatch')
    out.write_text(json.dumps(dict(groups=result,repetitions=4000,
        bindings={str(p):sha256(p) for p in (a,b,rp,previous,Path(__file__))},runtime_modified=False,
        limitation='Post-selection descriptive 95% paired patient bootstrap, unadjusted for 33 targets and epoch selection. Not a promotion test.'),indent=2)+'\n')
    for s,ts in result.items():print(s,{t:ts[t] for t in ('creatinine','lipase','glucose','bun')})


if __name__=='__main__':main()
