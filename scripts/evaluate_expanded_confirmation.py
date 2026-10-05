"""Frozen nominees on expanded source patients; no tuning on new outcomes."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    output=Path('artifacts/native_reconstruction_expanded_confirmation_evaluation_20260917.json')
    private=Path('data/research_recovery/expanded_confirmation_20260917/predictions')
    if output.exists() or private.exists():raise ValueError('Fresh evaluation required')
    sample_report=Path('artifacts/native_reconstruction_expanded_confirmation_samples_20260917.json')
    nomination_report=Path('artifacts/native_reconstruction_confirmed_heads_20260917.json')
    samples=json.loads(sample_report.read_text());nomination=json.loads(nomination_report.read_text())
    for report in (samples,nomination):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    cache_path=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    cache=torch.load(cache_path,weights_only=False)
    cp=Path('data/research_recovery/nested_reconstruction_training_20260917/real.pt')
    oldcp=Path('data/research_recovery/site_reconstruction_training_20260917/real.pt')
    checkpoint=torch.load(cp,weights_only=False);old=torch.load(oldcp,weights_only=False)
    if any(not torch.equal(v,old['best_state_dict'][k]) for k,v in checkpoint['best_state_dict'].items()):raise ValueError('Feature cache identity changed')
    heads={};medians={}
    for t,ch in nomination['choices'].items():
        fit=[d for d in cache if d['target']==t and d['role']=='fit']
        medians[t]=float(np.median([d['truth'] for d in fit]))
        if 'alpha' not in ch:continue
        alpha=ch['alpha']
        heads[t]=None if alpha is None else make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd')).fit(
            np.stack([d['body'] for d in fit]),[d['truth'] for d in fit])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    model.load_state_dict(checkpoint['best_state_dict'])
    captured=[]
    hook=model.readout.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().copy()[0]))
    ip=Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt')
    index=torch.load(ip,weights_only=False)
    if index['variables']!=variables:raise ValueError('Schema mismatch')
    private.mkdir();allrows=[];prediction_index=[]
    with torch.inference_mode():
        for i,shard in enumerate(index['shards']):
            p=Path(shard['path'])
            if sha256(p)!=shard['sha256']:raise ValueError('Changed sample')
            s=torch.load(p,weights_only=False)
            if s['patient']!=shard['patient']:raise ValueError('Patient mismatch')
            rows=[]
            for c,e in s['samples'].items():
                t=variables[c]
                if t not in medians:continue
                captured.clear()
                original=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],t)[0])
                if len(captured)!=1:raise ValueError('Missing body state')
                pred=None if t not in heads else medians[t] if heads[t] is None else float(heads[t].predict(captured[0][None])[0])
                rows.append(dict(patient=s['patient'],site=s['sample_sites'][c],target=t,truth=e['truth_normalized'],original=original,nominee=pred,population=medians[t]))
            path=private/f'patient_{i:05d}.pt';torch.save(rows,path)
            prediction_index.append(dict(path=str(path),sha256=sha256(path)));allrows.extend(rows)
            if (i+1)%100==0:print('Scored patients',i+1,'pairs',len(allrows),flush=True)
    hook.remove();torch.save(prediction_index,private/'index.pt')
    result={}
    for t in samples['accepted']:
        rows=[r for r in allrows if r['target']==t]
        if t not in medians:
            result[t]=dict(status='no_trained_source_head',samples=samples['accepted'][t]);continue
        if len(rows)!=samples['accepted'][t] or len({r['patient'] for r in rows})!=len(rows):raise ValueError('Pair coverage mismatch')
        y=np.array([r['truth'] for r in rows]);groups=[r['site'] for r in rows]
        pred={k:np.array([r[k] for r in rows]) for k in ('original','population')}
        if t in heads:pred['nominee']=np.array([r['nominee'] for r in rows])
        checks={} if t not in heads else {k:evidence(abs(pred['nominee']-y)-abs(pred[k]-y),groups) for k in ('original','population')}
        result[t]=dict(patients=len(rows),mae={k:float(np.mean(abs(v-y))) for k,v in pred.items()},
            confirmation=checks,nominee_confirmed=bool(checks) and confirmed(checks,len(rows)),
            status='fixed_nominee_scored' if checks else 'no_supported_nomination')
    paths=[Path(__file__),sample_report,nomination_report,cp,oldcp,cache_path,ip,private/'index.pt',Path('scripts/audit_confirmed_reconstruction_heads.py')]
    payload=dict(complete=True,targets=result,bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Expanded source-confirmation, historically reused hospitals. Fixed fit-only heads and encoder; same prespecified Bonferroni patient/site tests. No process/placebo/interval or new external promotion gate.')
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print('Confirmed nominees:',[t for t,r in result.items() if r.get('nominee_confirmed')],flush=True)


if __name__=='__main__':main()
