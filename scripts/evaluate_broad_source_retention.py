"""Fixed broad-source checkpoints on original held-out reconstruction patients."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from scripts.train_reconstruction_replication_v2 import input_tokens
from scripts.audit_confirmed_reconstruction_heads import evidence
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    out=Path('artifacts/native_reconstruction_broad_retention_20260917.json')
    save=Path('data/research_recovery/broad_source_20260917/retention.pt')
    if out.exists() or save.exists():raise ValueError('Existing retention audit')
    parent=Path('artifacts/native_reconstruction_matched_broad_evaluation_20260917.json')
    report=json.loads(parent.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Parent changed')
    split=Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    cache=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    ip=BACKUP/'map_native_anchor_support/expanded_reconstruction_20260913/index.pt'
    partition=torch.load(split,weights_only=False)
    refs={(r['subject_id'],r['target']):r for r in torch.load(cache,weights_only=False)
          if partition['roles'][r['subject_id']]=='evaluation'}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));variables=assets.checkpoint['variables']
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    models={};paths=[Path(__file__),parent,split,cache,ip]
    for arm in ('real','process','placebo'):
        cp=Path(f'data/research_recovery/broad_source_20260917/matched_training/{arm}.pt')
        model=deepcopy(template);model.load_state_dict(torch.load(cp,weights_only=False)['best_state_dict'])
        models[arm]=model;paths.append(cp)
    fi=Path('data/research_recovery/broad_source_20260917/matched_samples/index.pt');paths.append(fi)
    fit=torch.load(fi,weights_only=False)
    held=set(k[0] for k in refs)
    if held & {s['patient'] for s in fit['shards']}:raise ValueError('Patient overlap')
    values={}
    for s in fit['shards']:
        if s['role']!='fit':continue
        p=Path(s['path'])
        if sha256(p)!=s['sha256']:raise ValueError('Fit sample changed')
        for c,e in torch.load(p,weights_only=False)['samples'].items():
            values.setdefault(variables[c],[]).append(e['truth_normalized'])
    medians={t:float(np.median(v)) for t,v in values.items()}
    rows=[];seen=set()
    with torch.inference_mode():
        for s in torch.load(ip,weights_only=False)['shards']:
            if s['patient'] not in held:continue
            p=durable(s['path'])
            if sha256(p)!=s['sha256']:raise ValueError('Old sample changed')
            for c,e in torch.load(p,weights_only=False)['samples'].items():
                key=(s['patient'],variables[c])
                if key not in refs:continue
                ref=refs[key]
                if ref['truth']!=e['truth_normalized'] or key in seen:raise ValueError('Pair mismatch')
                seen.add(key)
                r=dict(patient=key[0],target=key[1],site=partition['components'][key[0]],
                       truth=ref['truth'],old=ref['original'],population=medians[key[1]])
                for arm,model in models.items():
                    r[arm]=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],key[1])[0])
                rows.append(r)
    if seen!=set(refs):raise ValueError('Incomplete retention pairs')
    torch.save(rows,save);paths.append(save)
    result={}
    for t in report['targets']:
        rr=[r for r in rows if r['target']==t]
        if not rr:result[t]=dict(status='no_trained_source_head');continue
        y=np.array([r['truth']for r in rr]);site=[r['site']for r in rr]
        err={k:abs(np.array([r[k]for r in rr])-y)for k in ('old','population','real','process','placebo')}
        result[t]=dict(patients=len(rr),mae={k:float(v.mean())for k,v in err.items()},
            comparisons={k:evidence(err['real']-err[k],site)for k in err if k!='real'})
    out.write_text(json.dumps(dict(complete=True,patients=len(held),pairs=len(rows),targets=result,
        bindings={str(p):sha256(p)for p in paths},runtime_modified=False,promotion_allowed=False,
        limitation='Fixed-weight retention on reused original evaluation. Descriptive, not noninferiority proof or new lockbox.'),indent=2)+'\n')
    print('Retention complete',len(held),len(rows),flush=True)


if __name__=='__main__':main()
