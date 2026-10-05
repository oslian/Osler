"""Full source creatinine final-update trace, never an evaluation-selected gate."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_final_update import trace_final
from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def summarize(rows, variables):
    full=interval([r['delta'] for r in rows])
    by_variable={}
    for v in variables:
        present=[r['delta'] for r in rows if v in r['last_variables']]
        absent=[r['delta'] for r in rows if v not in r['last_variables']]
        by_variable[v]=dict(present=interval(present) if len(present)>=30 else dict(patients=len(present),status='insufficient_support'),
                            absent=interval(absent) if len(absent)>=30 else dict(patients=len(absent),status='insufficient_support'))
    return dict(paired_change=full,by_last_variable=by_variable,
        before_mae=float(np.mean([r['before_error'] for r in rows])),after_mae=float(np.mean([r['after_error'] for r in rows])),
        final_mae=float(np.mean([r['final_error'] for r in rows])),
        positive_final_gap=sum(r['gap']>0 for r in rows))


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    cp=root/'bounded_site_training/bounded.pt'
    out=Path('artifacts/native_reconstruction_creatinine_full_update_20260920.json')
    private=root/'creatinine_full_update_20260920.pt'
    if out.exists() or private.exists():raise ValueError('Existing audit')
    training=json.loads(tp.read_text())
    for p,h in training['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training changed')
    ck=torch.load(cp,weights_only=False)
    if ck['bindings']!=training['bindings']:raise ValueError('Checkpoint changed')
    ip,fp,sp=root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,fp,sp)]
    shards=source_shards(joint,full,'expanded',split['roles'])
    original={s['patient'] for s in shards if s['group']=='broad' and s['role']=='encoder_tune'}
    selected=[s for s in shards if s['group']=='selected' and s['role']=='encoder_tune']
    broad=[s for s in full['shards'] if s['role']=='encoder_tune']
    forbidden={s['patient'] for s in shards if s['role']=='fit'}
    allids=[s['patient'] for s in selected+broad]
    if len(allids)!=len(set(allids)) or set(allids)&forbidden:raise ValueError('Source overlap')
    variables=joint['variables'];c=variables.index('creatinine');n=len(variables)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    allrows={}
    with torch.inference_mode():
        for group,source in (('selected',selected),('broad',broad)):
            store=SourceStore(source,[c]);rows=[]
            for key,truth in store.labels.items():
                e=store[key];tokens=input_tokens(e,'real',n)
                stages,metadata=trace_final(model,tokens,e['event_hours'],e['anchor_hour'],'creatinine')
                actual=float(model(tokens,e['event_hours'],e['anchor_hour'],'creatinine')[0])
                if not np.isclose(actual,stages['final'],rtol=0,atol=1e-7):raise ValueError('Trace mismatch')
                clean,_=mask_native_target(tokens,e['event_hours'],variables,'creatinine')
                before=abs(stages['before_last_update']-truth);after=abs(stages['after_last_update']-truth)
                rows.append(dict(patient=key[0],before_error=before,after_error=after,final_error=abs(actual-truth),delta=after-before,
                    last_variables=[variables[i] for i in torch.nonzero(clean[-1,n:2*n]>0).flatten().tolist()],
                    original=key[0] in original,gap=metadata['last_gap_hours']))
            check=rows if group=='selected' else [r for r in rows if r['original']]
            expected=training['arms']['bounded']['log'][ck['selected_epoch']-1]['group_target_scores'][group]['creatinine']
            if not np.isclose(np.mean([r['final_error'] for r in check]),expected,rtol=0,atol=1e-7):raise ValueError('Source score not reproduced')
            allrows[group]=rows;print(group,'traced',len(rows),flush=True)
    results={g:summarize(rows,variables) for g,rows in allrows.items()}
    results['broad_original']=summarize([r for r in allrows['broad'] if r['original']],variables)
    torch.save(allrows,private)
    paths=[Path(__file__),tp,cp,ip,fp,sp,Path('scripts/audit_reconstruction_final_update.py'),Path('scripts/audit_continued_selection_pairs.py')]
    out.write_text(json.dumps(dict(groups=results,bindings={str(p):sha256(p) for p in paths},private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='Source selection patients, fixed checkpoint, post-selection descriptive bootstrap and overlapping unadjusted observation strata. Not causal attribution, new external validation, forecast or a deployment skip-update rule.'),indent=2)+'\n')
    print({g:r['paired_change'] for g,r in results.items()})


if __name__=='__main__':main()
