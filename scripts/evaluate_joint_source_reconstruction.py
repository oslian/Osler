"""Fixed joint-source model, separately evaluated in both prior cohorts."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from scripts.train_reconstruction_replication_v2 import input_tokens
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.audit_broad_prior_increment import load_rows
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    out=Path('artifacts/native_reconstruction_joint_source_evaluation_20260918.json')
    private=Path('data/research_recovery/broad_source_20260917/joint_evaluation')
    if out.exists() or private.exists():raise ValueError('Existing evaluation')
    tp=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    training=json.loads(tp.read_text())
    for p,h in training['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training changed')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));variables=assets.checkpoint['variables']
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    models={};paths=[Path(__file__),tp]
    for arm in ('real','process','placebo'):
        p=Path(f'data/research_recovery/broad_source_20260917/joint_training/{arm}.pt')
        ck=torch.load(p,weights_only=False)
        if ck['bindings']!=training['bindings'] or ck['epoch']!=8:raise ValueError('Checkpoint mismatch')
        model=deepcopy(template);model.load_state_dict(ck['best_state_dict']);models[arm]=model;paths.append(p)
    medians=ck['medians']
    bi=Path('data/research_recovery/broad_source_20260917/matched_evaluation/index.pt')
    op=Path('data/research_recovery/broad_source_20260917/retention.pt')
    refs={'broad':load_rows(bi),'selected':{(r['patient'],r['target']):r for r in torch.load(op,weights_only=False)}}
    indices={'broad':Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt'),
             'selected':BACKUP/'map_native_anchor_support/expanded_reconstruction_20260913/index.pt'}
    paths.extend([bi,op,*indices.values()]);private.mkdir();results={}
    for group,ip in indices.items():
        rows=[];seen=set();patients={k[0]for k in refs[group]}
        with torch.inference_mode():
            for i,item in enumerate(torch.load(ip,weights_only=False)['shards']):
                if item['patient'] not in patients:continue
                p=durable(item['path'])
                if sha256(p)!=item['sha256']:raise ValueError('Changed sample')
                s=torch.load(p,weights_only=False)
                for c,e in s['samples'].items():
                    key=(s['patient'],variables[c])
                    if key not in refs[group]:continue
                    ref=refs[group][key]
                    if key in seen or ref['truth']!=e['truth_normalized']:raise ValueError('Pair mismatch')
                    seen.add(key);row=dict(patient=key[0],target=key[1],site=ref['site'],truth=ref['truth'],
                                         old=ref['old'],broad_only=ref['real'],population=medians[c])
                    for arm,model in models.items():
                        row[arm]=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],key[1])[0])
                    rows.append(row)
                if (i+1)%500==0:print(group,'patients traversed',i+1,flush=True)
        if seen!=set(refs[group]):raise ValueError('Incomplete pairs')
        p=private/f'{group}.pt';torch.save(rows,p);paths.append(p);per={}
        for t in json.loads(Path('artifacts/native_reconstruction_matched_broad_evaluation_20260917.json').read_text())['targets']:
            rr=[r for r in rows if r['target']==t]
            if not rr:per[t]=dict(status='no_trained_source_head');continue
            y=np.array([r['truth']for r in rr]);sites=[r['site']for r in rr]
            errors={k:abs(np.array([r[k]for r in rr])-y)for k in ('old','broad_only','population','real','process','placebo')}
            checks={k:evidence(errors['real']-errors[k],sites)for k in errors if k!='real'}
            supported=all(training['support'][g][role][t]>=30 for g in training['support']for role in training['support'][g])
            per[t]=dict(patients=len(rr),mae={k:float(v.mean())for k,v in errors.items()},comparisons=checks,
                        source_supported=supported,development_gate=supported and confirmed(checks,len(rr)))
        results[group]=per
    out.write_text(json.dumps(dict(complete=True,groups=results,bindings={str(p):sha256(p)for p in paths},
        runtime_modified=False,promotion_allowed=False,limitation='Reused cohorts; double single-source training budget. Current reconstruction only.'),indent=2)+'\n')
    print('Both cohort evaluations complete',flush=True)


if __name__=='__main__':main()
