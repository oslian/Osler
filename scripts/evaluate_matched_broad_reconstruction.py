"""Matched trained source cohorts on identical expanded confirmation labels."""
import json
import copy
from pathlib import Path
import numpy as np
import torch
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    out=Path('artifacts/native_reconstruction_matched_broad_evaluation_20260917.json')
    private=Path('data/research_recovery/broad_source_20260917/matched_evaluation')
    if out.exists() or private.exists():raise ValueError('Fresh evaluation required')
    tp=Path('artifacts/native_reconstruction_matched_broad_training_20260917.json')
    sp=Path('artifacts/native_reconstruction_expanded_confirmation_samples_20260917.json')
    oldp=Path('artifacts/native_reconstruction_expanded_confirmation_evaluation_20260917.json')
    training=json.loads(tp.read_text());samples=json.loads(sp.read_text());old=json.loads(oldp.read_text())
    for report in (training,samples,old):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed bound source')
    oldindex=Path('data/research_recovery/expanded_confirmation_20260917/predictions/index.pt')
    oldrows={}
    for item in torch.load(oldindex,weights_only=False):
        p=Path(item['path'])
        if sha256(p)!=item['sha256']:raise ValueError('Changed prior predictions')
        for r in torch.load(p,weights_only=False):oldrows[(r['patient'],r['target'])]=r
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    models={};checkpoints={}
    for arm in ('real','process','placebo'):
        assets=copy.deepcopy(load_release(Path('models')));variables=assets.checkpoint['variables']
        expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
        model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
        cp=Path(f'data/research_recovery/broad_source_20260917/matched_training/{arm}.pt')
        ck=torch.load(cp,weights_only=False)
        if ck['bindings']!=training['bindings'] or ck['epoch']!=8:raise ValueError('Training mismatch')
        model.load_state_dict(ck['best_state_dict']);models[arm]=model;checkpoints[str(cp)]=sha256(cp)
    pointers=[{p.data_ptr() for p in model.parameters()} for model in models.values()]
    if any(a & b for i,a in enumerate(pointers) for b in pointers[i+1:]):
        raise ValueError('Comparison arms share parameter storage')
    ip=Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt')
    index=torch.load(ip,weights_only=False);targets=set(training['source_targets'])
    fitindex=Path('data/research_recovery/broad_source_20260917/matched_samples/index.pt')
    values={t:[] for t in targets}
    for item in torch.load(fitindex,weights_only=False)['shards']:
        if item['role']!='fit':continue
        p=Path(item['path'])
        if sha256(p)!=item['sha256']:raise ValueError('Changed fit sample')
        for c,e in torch.load(p,weights_only=False)['samples'].items():
            if variables[c] in values:values[variables[c]].append(e['truth_normalized'])
    medians={t:float(np.median(v)) for t,v in values.items()}
    private.mkdir();records=[];saved=[]
    with torch.inference_mode():
        for i,shard in enumerate(index['shards']):
            p=Path(shard['path'])
            if sha256(p)!=shard['sha256']:raise ValueError('Changed confirmation')
            s=torch.load(p,weights_only=False);rows=[]
            for c,e in s['samples'].items():
                t=variables[c]
                if t not in targets:continue
                previous=oldrows[(s['patient'],t)]
                if previous['truth']!=e['truth_normalized'] or previous['site']!=s['sample_sites'][c]:raise ValueError('Pair identity mismatch')
                row=dict(patient=s['patient'],site=previous['site'],target=t,truth=e['truth_normalized'],old=previous['original'],population=medians[t])
                for arm,model in models.items():
                    row[arm]=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],t)[0])
                rows.append(row)
            path=private/f'patient_{i:05d}.pt';torch.save(rows,path);saved.append(dict(path=str(path),sha256=sha256(path)));records.extend(rows)
            if (i+1)%100==0:print('Compared patients',i+1,flush=True)
    torch.save(saved,private/'index.pt');result={}
    for t in samples['accepted']:
        if t not in targets:result[t]=dict(status='no_trained_source_head',samples=samples['accepted'][t]);continue
        rows=[r for r in records if r['target']==t]
        if len(rows)!=samples['accepted'][t]:raise ValueError('Incomplete scoring')
        y=np.array([r['truth'] for r in rows]);groups=[r['site'] for r in rows]
        pred={k:np.array([r[k] for r in rows]) for k in ('old','population','real','process','placebo')}
        checks={k:evidence(abs(pred['real']-y)-abs(pred[k]-y),groups) for k in ('old','population','process','placebo')}
        mae={k:float(abs(p-y).mean()) for k,p in pred.items()}
        if not np.isclose(mae['old'],old['targets'][t]['mae']['original'],atol=1e-10,rtol=0):raise ValueError('Old score mismatch')
        source_supported=all(training['support'][r][t]>=30 for r in ('fit','encoder_tune'))
        result[t]=dict(patients=len(rows),mae=mae,comparisons=checks,source_supported=source_supported,
            development_gate=source_supported and confirmed(checks,len(rows)))
    paths=[Path(__file__),tp,sp,oldp,oldindex,ip,fitindex,private/'index.pt',Path('scripts/audit_confirmed_reconstruction_heads.py')]
    bindings={str(p):sha256(p) for p in paths};bindings.update(checkpoints)
    out.write_text(json.dumps(dict(complete=True,targets=result,bindings=bindings,runtime_modified=False,promotion_allowed=False,
        limitation='Single matched-source fit; reused source confirmation. Equal people/site quotas/updates, not full case mix or input exposure. No future/conformal/external promotion.'),indent=2)+'\n')
    print('Development-gate targets:',[t for t,r in result.items() if r.get('development_gate')],flush=True)


if __name__=='__main__':main()
