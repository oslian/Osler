"""Full-parameter autograd reachability on source-only reconstruction probes."""
from copy import deepcopy
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from scripts.reconstruction_source_store import SourceStore,source_shards
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def component(name):
    parts=name.split('.')
    return '.'.join(parts[:3]) if parts[0]=='trajectory' else parts[0]


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    tp=Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    cp=root/'bounded_site_training/bounded.pt'
    out=Path('artifacts/native_reconstruction_gradient_paths_20260920.json')
    if out.exists():raise ValueError('Existing audit')
    training=json.loads(tp.read_text())
    for p,h in training['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError(f'Binding changed: {p}')
    ck=torch.load(cp,weights_only=False)
    if ck['bindings']!=training['bindings']:raise ValueError('Checkpoint mismatch')
    ip,fp,sp=root/'joint_source_manifest.pt',root/'full_samples/index.pt',Path('data/research_recovery/reconstruction_nested_selection_20260917.pt')
    joint,full,split=[torch.load(p,weights_only=False) for p in (ip,fp,sp)]
    variables,targets=joint['variables'],joint['targets'];columns=[variables.index(t) for t in targets]
    shards=source_shards(joint,full,'expanded',split['roles'])
    torch.set_num_threads(4);torch.manual_seed(20260920)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    active=defaultdict(int);connected=defaultdict(int);norms=defaultdict(list);tasks=[]
    rng=np.random.default_rng(20260920)
    for group in ('selected','broad'):
        store=SourceStore([s for s in shards if s['group']==group and s['role']=='fit'],columns)
        for c in columns:
            keys=store.groups[c];e=store[keys[int(rng.integers(len(keys)))]]
            pred=model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],variables[c])
            loss=torch.nn.functional.smooth_l1_loss(pred,pred.new_tensor([e['truth_normalized']]))
            grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
            bycomponent=defaultdict(float)
            for (name,p),grad in zip(named,grads):
                if grad is not None:
                    if not torch.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    connected[name]+=1
                    value=float(grad.detach().double().square().sum())
                    active[name]+=int(value>0);bycomponent[component(name)]+=value
            for name,value in bycomponent.items():norms[name].append(value**.5)
            tasks.append(dict(group=group,target=variables[c],events=len(e['tokens']),gradient_components=sorted(k for k,v in bycomponent.items() if v>0)))
        print('Full gradient paths checked',group,flush=True)
    per={}
    for name,p in named:
        per[name]=dict(elements=p.numel(),connected_probes=connected[name],nonzero_probes=active[name])
    groups={}
    for name in sorted({component(n) for n,_ in named}):
        selected={n:r for n,r in per.items() if component(n)==name}
        groups[name]=dict(tensors=len(selected),elements=sum(r['elements'] for r in selected.values()),
            ever_connected_tensors=sum(r['connected_probes']>0 for r in selected.values()),ever_nonzero_tensors=sum(r['nonzero_probes']>0 for r in selected.values()),
            max_probe_norm=max(norms.get(name,[0.])))
    out.write_text(json.dumps(dict(components=groups,parameters=per,tasks=tasks,selected_epoch=ck['selected_epoch'],
        bindings={str(p):sha256(p) for p in (Path(__file__),tp,cp,ip,fp,sp)},runtime_modified=False,promotion_allowed=False,
        limitation='One source patient per target/group (66 probes), one selected checkpoint. Autograd reachability is not sufficient learning, full-distribution coverage or causal error attribution. Unused forecast branches are expected for current reconstruction.'),indent=2)+'\n')
    print('Persistent/readout components',{k:v for k,v in groups.items() if not k.startswith('trajectory.jepa.')})


if __name__=='__main__':main()
