"""Identical masked events: raw summary, process summary and frozen body probe."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.audit_reconstruction_source_fit import durable
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def summarize(tokens, times, anchor, variables, target):
    x,t=mask_native_target(tokens,times,variables,target)
    n=len(variables);mask=x[:,n:2*n].bool().numpy();values=x[:,:n].numpy()
    latest=np.zeros(n);mean=np.zeros(n);present=mask.any(0).astype(float)
    lag=np.full(n,49.);count=mask.sum(0)
    for c in np.flatnonzero(present):
        indices=np.flatnonzero(mask[:,c]);latest[c]=values[indices[-1],c]
        mean[c]=values[indices,c].mean();lag[c]=float(anchor-t[indices[-1]])
    raw=np.r_[latest,mean,present,lag,np.log1p(count)]
    process=raw.copy();process[:2*n]=0
    return raw,process


def main():
    output=Path('artifacts/native_reconstruction_raw_state_20260917.json')
    cache=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    if output.exists() or cache.exists():raise ValueError('existing experiment')
    ep=Path('artifacts/native_reconstruction_site_evaluation_20260917.json')
    tp=Path('artifacts/native_reconstruction_site_training_epoch8_20260917.json')
    ev=json.loads(ep.read_text());training=json.loads(tp.read_text())
    for report in (ev,training):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('changed dependency')
    split=Path('data/research_recovery/reconstruction_site_partition_20260917.pt')
    roles=torch.load(split,weights_only=False)['roles']
    ip=Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt')
    index=torch.load(ip,weights_only=False);variables=index['variables']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));expand_history_projection(assets.model,6)
    assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    cp=Path('data/research_recovery/site_reconstruction_training_20260917/real.pt')
    ck=torch.load(cp,weights_only=False)
    if ck['bindings']!=training['bindings']:raise ValueError('model identity')
    model.load_state_dict(ck['best_state_dict'])
    captured=[]
    hook=model.readout.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().copy()[0]))
    data=[]
    with torch.no_grad():
        for shard in index['shards']:
            role=roles[shard['patient']]
            if role not in ('fit','tune','evaluation'):continue
            p=durable(shard['path'])
            if sha256(p)!=shard['sha256']:raise ValueError('changed shard')
            s=torch.load(p,weights_only=False)
            if s['patient']!=shard['patient']:raise ValueError('patient mismatch')
            for c,e in s['samples'].items():
                target=variables[c]
                if target not in ev['targets']:continue
                raw,process=summarize(e['tokens'],e['event_hours'],e['anchor_hour'],variables,target)
                captured.clear()
                prediction=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],target)[0])
                if len(captured)!=1:raise ValueError('readout capture mismatch')
                data.append(dict(subject_id=s['patient'],role=role,target=target,truth=e['truth_normalized'],
                    raw=raw,raw_process=process,body=captured[0],original=prediction))
            if len(data)//1000!=(len(data)-len(s['samples']))//1000:print('extracted',len(data),flush=True)
    hook.remove();torch.save(data,cache)
    results={}
    for target in training['source_targets']:
        groups={r:[d for d in data if d['target']==target and d['role']==r] for r in ('fit','tune','evaluation')}
        ids=[{d['subject_id'] for d in group} for group in groups.values()]
        if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('role overlap')
        y={r:np.array([d['truth'] for d in group]) for r,group in groups.items()}
        median=float(np.median(y['fit']));n=len(y['evaluation'])
        pred={'population':np.full(n,median),'original':np.array([d['original'] for d in groups['evaluation']])}
        for a,key in (('original','real'),('population','population')):
            if not np.isclose(np.mean(abs(pred[a]-y['evaluation'])),ev['targets'][target]['normalized_mae'][key],atol=1e-7,rtol=0):raise ValueError('original MAE changed')
        choices={}
        for source in ('raw','raw_process','body'):
            x={r:np.stack([d[source] for d in group]) for r,group in groups.items()}
            best=float(np.mean(abs(y['tune']-median)));selected=None;prediction=np.full(n,median)
            for alpha in (1.,10.,100.,1000.,10000.):
                candidate=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd')).fit(x['fit'],y['fit'])
                score=float(np.mean(abs(candidate.predict(x['tune'])-y['tune'])))
                if score<best:
                    best=score;selected=alpha;prediction=candidate.predict(x['evaluation'])
            choices[source]=dict(alpha=selected,tune_mae=best,features=x['fit'].shape[1]);pred[source]=prediction
        frame=pd.DataFrame(dict(subject_id=[d['subject_id'] for d in groups['evaluation']],truth=y['evaluation']))
        results[target]=dict(patients=n,choices=choices,mae={a:float(np.mean(abs(p-y['evaluation']))) for a,p in pred.items()},
            comparisons={ref:paired_model_errors(frame,pred,ref,seed=20260917,repetitions=2000) for ref in ('population','original','raw_process','body')})
        print('probed',target,flush=True)
    paths=[Path(__file__),ep,tp,split,ip,cp,cache,Path('RECONSTRUCTION_RAW_STATE_PROTOCOL_20260917.md')]
    output.write_text(json.dumps(dict(complete=True,targets=results,unsupported=training['source_unsupported'],
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Same masked rows but different feature dimensions; reused development site evaluation. Source tune selects ridge/baseline, no encoder update. Nominal uncorrected CIs, not external or future promotion.'),indent=2)+'\n')


if __name__=='__main__':main()
