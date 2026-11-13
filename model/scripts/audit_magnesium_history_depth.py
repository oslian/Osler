"""Pre-anchor history-depth strata; descriptive, not routing selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_magnesium_matched_20260911.json');ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_history_depth_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    n=dict(np.load(root/'native_tokens.npz'));l=dict(np.load(root/'multitarget_labels.npz'));v=list(l['variables']);k=v.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];ok=l['paired'][:,k]&l['paired'][:,v.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=l['future'][:,k]
    counts=np.array([(n['input'][n['history_indices'][a:b],len(v)+k]>0).sum() for a,b in zip(n['query_ptr'][:-1],n['query_ptr'][1:])])
    groups={'one_two':counts<=2,'three_four':(counts>=3)&(counts<=4),'five_plus':counts>=5}
    items=[]
    for e in ref['seeds']:
        seed=e['seed'];p=dict(np.load(root/f'magnesium_matched_predictions_{seed}.npz'))
        roles={a:b[valid]&ok for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for name,mask in groups.items():
                m=roles[role]&mask;f=q.loc[m]
                item['roles'][role][name]=paired(f,p['real'][m],p['own'][m],f'mg-depth-{seed}-{role}-{name}',2000)
        items.append(item)
    report=dict(complete=True,runtime_modified=False,descriptive_only=True,seeds=items,
        history_count_quantiles=np.quantile(counts[ok],[0,.25,.5,.75,1]).tolist(),
        at_most_two_fraction=float(np.mean(counts[ok]<=2)),
        binding={str(p):sha256(p) for p in (parent,Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in groups:
            rows=[s['roles'][role][group] for s in items];supported=[r for r in rows if r['supported']]
            print(role,group,'supported',len(supported),'gain',sum(r['ci95'][1]<0 for r in supported),'harm',sum(r['ci95'][0]>0 for r in supported),'mean',float(np.median([r['mean'] for r in rows])))


if __name__=='__main__':main()
