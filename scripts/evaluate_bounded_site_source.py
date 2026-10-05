"""Frozen matched site-sampling candidates on identical reused evaluation pairs."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    rp=Path('artifacts/native_reconstruction_coverage_evaluation_20260920.json')
    out=Path('artifacts/native_reconstruction_bounded_site_evaluation_20260920.json')
    private=root/'bounded_site_evaluation'
    if out.exists() or private.exists():raise ValueError('Existing evaluation')
    training,parent=[json.loads(p.read_text()) for p in (tp,rp)]
    for report in (training,parent):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError(f'Binding changed: {p}')
    assets=deepcopy(load_release(Path('models')));variables=assets.checkpoint['variables']
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    paths=[Path(__file__),tp,rp];models={}
    for arm in ('uniform','bounded'):
        p=root/f'bounded_site_training/{arm}.pt';ck=torch.load(p,weights_only=False)
        if ck['bindings']!=training['bindings'] or ck['epoch']!=8 or ck['selected_epoch']!=training['arms'][arm]['selected_epoch']:
            raise ValueError('Checkpoint mismatch')
        model=deepcopy(template);model.load_state_dict(ck['best_state_dict']);models[arm]=model;paths.append(p)
    indices={'broad':Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt'),
             'selected':BACKUP/'map_native_anchor_support/expanded_reconstruction_20260913/index.pt'}
    private.mkdir();results={}
    for group,ip in indices.items():
        pp=root/f'coverage_evaluation/{group}.pt';paths.extend([ip,pp])
        originals=torch.load(pp,weights_only=False)
        refs={(r['patient'],r['target']):r for r in originals}
        if len(refs)!=len(originals):raise ValueError('Duplicate reference pairs')
        patients={k[0] for k in refs};seen=set();rows=[]
        with torch.inference_mode():
            for shard in torch.load(ip,weights_only=False)['shards']:
                if shard['patient'] not in patients:continue
                p=durable(shard['path'])
                if sha256(p)!=shard['sha256']:raise ValueError('Sample changed')
                for c,e in torch.load(p,weights_only=False)['samples'].items():
                    key=(shard['patient'],variables[c])
                    if key not in refs:continue
                    if key in seen or refs[key]['truth']!=e['truth_normalized']:raise ValueError('Pair mismatch')
                    seen.add(key);row=dict(refs[key])
                    for arm,model in models.items():
                        row['site_'+arm]=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],variables[c])[0])
                    rows.append(row)
        if seen!=set(refs):raise ValueError('Incomplete pairs')
        saved=private/f'{group}.pt';torch.save(rows,saved);paths.append(saved);per={}
        names=('old','expanded_population','expanded_real','continued_real','site_uniform','site_bounded')
        for target,original in parent['groups'][group].items():
            rr=[r for r in rows if r['target']==target]
            if not rr:per[target]=dict(status='no_trained_source_head');continue
            y=np.array([r['truth'] for r in rr]);sites=[r['site'] for r in rr]
            errors={k:abs(np.array([r[k] for r in rr])-y) for k in names}
            checks={k:evidence(errors['site_bounded']-e,sites) for k,e in errors.items() if k!='site_bounded'}
            if any(not np.isclose(errors[k].mean(),original['mae'][k],rtol=0,atol=1e-10) for k in names if k in original['mae']):raise ValueError('Reference score mismatch')
            per[target]=dict(patients=len(rr),mae={k:float(v.mean()) for k,v in errors.items()},comparisons=checks,
                source_supported=original['source_supported'],
                paired_sampling_gate=original['source_supported'] and confirmed({'uniform':checks['site_uniform']},len(rr)),
                diagnostic_all_comparisons=original['source_supported'] and confirmed(checks,len(rr)))
        results[group]=per
        good=[r for r in per.values() if 'mae' in r]
        print(group,{k:float(np.mean([r['mae'][k] for r in good])) for k in names},flush=True)
    out.write_text(json.dumps(dict(complete=True,groups=results,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Reused reconstruction cohorts, not untouched external or future forecasting. New matched process/placebo not trained; diagnostic comparisons are not full promotion.'),indent=2)+'\n')


if __name__=='__main__':main()
