"""Source-only suffix ablations of a fixed shared reconstruction state."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from scripts.audit_continued_selection_pairs import interval
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction,mask_native_target
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def suffix(tokens,times,count):
    if count<1 or len(tokens)!=len(times) or not len(tokens):raise ValueError('Invalid suffix')
    return tokens[-count:],times[-count:]


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json');cp=root/'bounded_site_training/bounded.pt'
    out=Path('artifacts/native_reconstruction_history_suffix_20260920.json');private=root/'history_suffix_20260920.pt'
    if out.exists() or private.exists():raise ValueError('Existing audit')
    training=json.loads(tp.read_text())
    for p,h in training['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training changed')
    ck=torch.load(cp,weights_only=False)
    if ck['bindings']!=training['bindings']:raise ValueError('Checkpoint mismatch')
    ip,fp,sp=root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,fp,sp)]
    variables,targets=joint['variables'],joint['targets'];cols=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    output={};raw={}
    with torch.inference_mode():
        for group in ('selected','broad'):
            store=SourceStore([s for s in shards if s['group']==group and s['role']=='encoder_tune'],cols);per={};rows=[]
            for key,truth in store.labels.items():
                e=store[key];target=variables[key[1]];tokens=input_tokens(e,'real',len(variables))
                clean,times=mask_native_target(tokens,e['event_hours'],variables,target)
                predictions={}
                for name,count in (('full',len(clean)),('last8',8),('last1',1)):
                    x,t=suffix(clean,times,count)
                    predictions[name]=float(model(x,t,e['anchor_hour'],target)[0])
                rows.append(dict(patient=key[0],target=target,events=len(clean),errors={k:abs(v-truth) for k,v in predictions.items()}))
            for target in targets:
                chosen=[r for r in rows if r['target']==target]
                mae={k:float(np.mean([r['errors'][k] for r in chosen])) for k in ('full','last8','last1')}
                expected=training['arms']['bounded']['log'][ck['selected_epoch']-1]['group_target_scores'][group][target]
                if not np.isclose(mae['full'],expected,rtol=0,atol=1e-7):raise ValueError('Original source score mismatch')
                changes={k:interval([r['errors'][k]-r['errors']['full'] for r in chosen]) for k in ('last8','last1')}
                per[target]=dict(patients=len(chosen),mae=mae,suffix_minus_full=changes,
                    affected_patients={k:sum(r['events']>n for r in chosen) for k,n in (('last8',8),('last1',1))})
            output[group]=per;raw[group]=rows;print(group,'pairs',len(rows),'complete',flush=True)
    torch.save(raw,private)
    out.write_text(json.dumps(dict(groups=output,bindings={str(p):sha256(p) for p in (Path(__file__),tp,cp,ip,fp,sp)},private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='Source selection data; fixed checkpoint, suffix distribution shift and zero initialization change. Descriptive unadjusted paired intervals, not matched training, causal forgetting or deployment selection.'),indent=2)+'\n')
    for group,per in output.items():
        print(group,{k:float(np.mean([r['mae'][k] for r in per.values()])) for k in ('full','last8','last1')})


if __name__=='__main__':main()
