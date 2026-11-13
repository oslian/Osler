"""Check whether recorded numerical rates collapse under source scaling."""
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');out=Path('artifacts/native_rate_scale_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/rate_anchor_jepa_native_20260911.json').read_text())
    path=root/'native_tokens.npz'
    if sha256(path)!=ref['binding'][str(path)]:raise ValueError('cache changed')
    with np.load(path) as f:t={k:f[k] for k in f.files}
    valid=np.load(root/'labels.npz')['paired'][:,0];items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];cp=root/f'jepa_native_real_{seed}.pt'
        if sha256(cp)!=ref['binding'][str(cp)]:raise ValueError('checkpoint changed')
        s=torch.load(cp,weights_only=True);center=s['event_center'][2];scale=s['event_scale'][2]
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()};item=dict(seed=seed,center=center,scale=scale,roles={})
        for role,m in roles.items():
            ids=np.unique(np.concatenate([t['history_indices'][t['query_ptr'][i]:t['query_ptr'][i+1]] for i in np.flatnonzero(m)]))
            v=t['event_values'][ids,2];v=v[np.isfinite(v)];z=(v-center)/scale
            item['roles'][role]=dict(unique_supported_tokens=len(v),zero_fraction=float(np.mean(v==0)),
                raw_quantiles=np.quantile(v,[0,.25,.5,.75,.99,1]).tolist(),
                normalized_iqr=float(np.quantile(z,.75)-np.quantile(z,.25)),
                normalized_std=float(np.std(z)),abs_z_over5_fraction=float(np.mean(np.abs(z)>5)))
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,fitting_performed=False,runtime_modified=False,seeds=items,
        binding={str(p):sha256(p) for p in (Path(__file__),path,Path('artifacts/rate_anchor_jepa_native_20260911.json'))}),indent=2)+'\n')
    for role in ('fit','evaluation','hospital'):
        print(role,{k:[round(v['roles'][role][k],6) for v in items] for k in ('normalized_iqr','normalized_std','abs_z_over5_fraction')})


if __name__=='__main__':main()
