"""Complete the recorded five optimizer failures without changing support."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.diagnose_seen_domain_correction import query_people, matched_fits
from scripts.try_robust_source_correction import robust_predict
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    rp=Path('artifacts/native_reconstruction_seen_domain_20260925.json')
    cp=Path('artifacts/native_reconstruction_robust_convergence_20260925.json')
    out=Path('artifacts/native_reconstruction_seen_domain_completed_20260925.json')
    if out.exists():raise ValueError('Existing completion')
    original=json.loads(rp.read_text());fix=json.loads(cp.read_text())
    for p,h in fix['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed repair inputs')
    variables=json.loads(Path('artifacts/native_reconstruction_variable_snapshot_20260925.json').read_text())['variables']
    result=copy.deepcopy(original['targets'])
    with threadpool_limits(limits=4):
        for repaired in fix['results']:
            t=repaired['target'];f=repaired['fold'];c=repaired['component']
            frame=pd.read_parquet(root/f'calibration_20260924/fold{f}.parquet');queries=query_people(frame)
            snap=pd.read_parquet(root/'variable_snapshot_20260925.parquet',filters=[('target','=',t)])
            frame=frame.loc[frame.target==t].merge(snap,on=['patient','target'],validate='one_to_one')
            comp={str(v):v for v in frame.component.unique()}[c]
            seen,unseen,query=matched_fits(frame,queries,comp)
            a,ai=robust_predict(seen,query,variables);b,bi=robust_predict(unseen,query,variables)
            if not ai['arms']['process_context']['converged'] or not repaired['stages'][1]['converged']:
                raise ValueError('Incomplete numerical repair')
            entry=next(x for x in result[t][str(f)]['components'] if x['component']==c)
            entry['seen_fit']['status']='fitted'
            entry['seen_fit']['arms']['numeric_context'].update(converged=True,
                iterations=300+repaired['stages'][1]['iterations'],residual_scale=repaired['stages'][1]['scale'])
            entry['numerical_continuation']=True
            # Numerical repair cannot create missing training-domain support.
            if bi['status']!='fitted':
                if bi['status']!=entry['unseen_fit']['status']:
                    raise ValueError('Unseen support status changed')
                continue
            mae={'unchanged':float(np.mean(abs(query.real-query.truth)))}
            for name,preds in (('seen',a),('unseen',b)):
                for arm,p in preds.items():mae[f'{name}_{arm}']=float(np.mean(abs(p-query.truth)))
            if len(query)!=repaired['query_patients'] or not np.isclose(mae['seen_numeric_context'],repaired['query_mae']['original'],atol=1e-10,rtol=1e-12):
                raise ValueError('Original comparison changed')
            mae['seen_numeric_context']=repaired['query_mae']['continued']
            entry.update(status='paired',mae=mae,numerical_continuation=True)
    for folds in result.values():
        for r in folds.values():
            rows=r['components'];r['complete']=bool(rows) and all(x['status']=='paired' for x in rows)
            r['supported']=r['complete'] and sum(x['query_patients']>=30 for x in rows)>=3
            if r['complete']:r['component_equal_mae']={a:float(np.mean([x['mae'][a] for x in rows])) for a in rows[0]['mae']}
    assessed=[r for z in result.values() for r in z.values() if r['supported']]
    categories=dict(both_better=0,seen_only=0,unseen_only=0,neither=0)
    for r in assessed:
        m=r['component_equal_mae'];a=m['seen_numeric_context']<m['unchanged'];b=m['unseen_numeric_context']<m['unchanged']
        categories[('both_better' if b else 'seen_only') if a else ('unseen_only' if b else 'neither')]+=1
    summary=dict(supported_complete=len(assessed),still_incomplete=235-len(assessed),comparisons=categories,
                 seen_better_than_unseen=sum(r['component_equal_mae']['seen_numeric_context']<r['component_equal_mae']['unseen_numeric_context'] for r in assessed))
    out.write_text(json.dumps(dict(summary=summary,targets=result,bindings={str(p):sha256(p) for p in (rp,cp,Path(__file__))},
        runtime_modified=False,promotion_allowed=False,outer_evaluation_performed=False,
        limitation='Optimizer continuation applied only to recorded nonconverged fits. No support relaxation or outcome-selected repair.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
