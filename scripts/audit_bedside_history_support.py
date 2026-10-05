"""Pre-anchor history support strata, not a learned routing rule."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    out=Path('artifacts/bedside_history_support_20260912.json')
    if out.exists():raise ValueError('fresh output required')
    native=dict(np.load(root/'native_tokens_bedside.npz'))
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    variables=labels['variables'].tolist();n=len(variables);col=variables.index('glucose')
    columns=[n+variables.index(v) for v in ('heart_rate','map','o2sat','respiratory_rate','temperature')]
    counts=np.asarray([(native['input'][native['history_indices'][a:b]][:,columns]>0).sum(axis=0)
        for a,b in zip(native['query_ptr'][:-1],native['query_ptr'][1:])])
    enough=(counts>=5).all(axis=1)
    valid=np.load(root/'labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col];items=[]
    for seed in (7,19,31,43,59,71,83):
        real=np.load(root/f'bedside_rescale_predictions_{seed}.npz');support=np.load(root/f'bedside_support_predictions_{seed}.npz')
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            item['roles'][role]={}
            for group,mask in [('five_each',enough),('less_support',~enough)]:
                m=roles[role]&mask
                comparison=paired_model_errors(q.loc[m],dict(real=real['bedside'][m],support=support['bedside'][m]),
                    'support',seed=20260912,repetitions=20000)['real']
                item['roles'][role][group]=dict(rows=int(m.sum()),comparison=comparison)
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,training_performed=False,runtime_modified=False,
        definition='at least five observed native groups for each of five bedside variables in pre-anchor history',
        enough_queries=int(enough.sum()),all_queries=len(q),seeds=items),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in ('five_each','less_support'):
            cs=[s['roles'][role][group]['comparison'] for s in items]
            print(role,group,'patients',[c['patients'] for c in cs],
                'gain',sum(c['ci95'] is not None and c['ci95'][1]<0 for c in cs),
                'harm',sum(c['ci95'] is not None and c['ci95'][0]>0 for c in cs))


if __name__=='__main__':main()
