"""Evaluate independent readout selection after verified nested retraining."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.audit_reconstruction_readout_selection import select_source
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/native_reconstruction_nested_heads_20260917.json')
    choices_path=Path('artifacts/native_reconstruction_nested_head_choices_20260917.json')
    if output.exists() or choices_path.exists():raise ValueError('existing nested evaluation')
    training_path=Path('artifacts/native_reconstruction_nested_training_20260917.json')
    cache_report_path=Path('artifacts/native_reconstruction_raw_state_20260917.json')
    partition_path=Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    training=json.loads(training_path.read_text());cache_report=json.loads(cache_report_path.read_text())
    partition=json.loads(partition_path.read_text())
    for report in (training,cache_report,partition):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('changed dependency')
    paths=[Path(__file__),training_path,cache_report_path,partition_path,
           Path('scripts/audit_reconstruction_readout_selection.py'),Path('whole_body_jepa/change_evaluation.py')]
    for arm in ('real','process','placebo'):
        oldp=Path(f'data/research_recovery/site_reconstruction_training_20260917/{arm}.pt')
        newp=Path(f'data/research_recovery/nested_reconstruction_training_20260917/{arm}.pt')
        old=torch.load(oldp,weights_only=False);new=torch.load(newp,weights_only=False)
        if new['bindings']!=training['bindings'] or new['epoch']!=8:raise ValueError('wrong training')
        if old['best_state_dict'].keys()!=new['best_state_dict'].keys() or any(not torch.equal(v,new['best_state_dict'][k]) for k,v in old['best_state_dict'].items()):raise ValueError('cache requires recomputation for changed weights')
        paths.extend([oldp,newp])
    cp=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    sp=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    data=torch.load(cp,weights_only=False);roles=torch.load(sp,weights_only=False)['roles'];paths.extend([cp,sp])
    choices={};work={}
    eligible=set(partition['all_four_roles_supported'])
    for target in training['source_targets']:
        groups={r:[d for d in data if d['target']==target and roles[d['subject_id']]==r] for r in ('fit','readout_tune','evaluation')}
        for role,rows in groups.items():
            if len(rows)!=partition['target_counts'][target][role]:raise ValueError('role support mismatch')
        if target not in eligible:
            choices[target]=dict(selected='original',status='insufficient_four_role_support')
            work[target]=(groups['evaluation'],None,'original')
            continue
        fit,tune=groups['fit'],groups['readout_tune']
        x=np.stack([d['body'] for d in fit]);y=np.array([d['truth'] for d in fit]);median=float(np.median(y))
        tx=np.stack([d['body'] for d in tune]);ty=np.array([d['truth'] for d in tune])
        scores={'original':float(np.mean(abs(np.array([d['original'] for d in tune])-ty))),
                'population':float(np.mean(abs(median-ty)))}
        models={'original':None,'population':median}
        for alpha in (1.,10.,100.,1000.,10000.):
            name=f'ridge_{alpha:g}';m=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd')).fit(x,y)
            scores[name]=float(np.mean(abs(m.predict(tx)-ty)));models[name]=m
        selected=select_source(scores)
        choices[target]=dict(selected=selected,status='development_candidate',source_scores=scores)
        work[target]=(groups['evaluation'],models[selected],selected)
    bindings={str(p):sha256(p) for p in paths}
    choices_path.write_text(json.dumps(dict(choices=choices,bindings=bindings,promotion_allowed=False),indent=2)+'\n')
    result={}
    for t,(rows,m,name) in work.items():
        original=np.array([d['original'] for d in rows]);truth=np.array([d['truth'] for d in rows])
        selected=original.copy() if name=='original' else (np.full(len(rows),m) if name=='population' else m.predict(np.stack([d['body'] for d in rows])))
        median=float(np.median([d['truth'] for d in data if d['target']==t and roles[d['subject_id']]=='fit']))
        pred={'selected':selected,'original':original,'population':np.full(len(rows),median)}
        frame=pd.DataFrame(dict(subject_id=[d['subject_id'] for d in rows],truth=truth))
        result[t]=dict(choice=choices[t],patients=len(rows),mae={a:float(np.mean(abs(p-truth))) for a,p in pred.items()},
            comparisons={r:paired_model_errors(frame,pred,r,seed=20260917,repetitions=2000) for r in ('original','population')})
        if not np.isclose(result[t]['mae']['original'],cache_report['targets'][t]['mae']['original'],atol=1e-10,rtol=0):raise ValueError('original score mismatch')
    bindings[str(choices_path)]=sha256(choices_path)
    output.write_text(json.dumps(dict(complete=True,targets=result,unsupported=training['source_unsupported'],
        bindings=bindings,selected_weights_identical_to_previous=True,runtime_modified=False,promotion_allowed=False,
        limitation='Independent readout selection from current fine-tuning/checkpoint selection only. Historical initialization provenance caveat remains; reused development evaluation, nominal uncorrected intervals. Unsupported targets preserve original diagnostically, not promotion.'),indent=2)+'\n')
    print('complete',output)


if __name__=='__main__':main()
