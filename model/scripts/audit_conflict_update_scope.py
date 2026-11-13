"""Verify actual parameter changes, not just requires_grad declarations."""
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from scripts.train_availability_encoder import configure
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    output=Path('artifacts/native_conflict_update_scope_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    ref=json.loads(parent.read_text())
    for path,h in ref['binding'].items():
        if sha256(Path(path))!=h:raise ValueError('parent changed')
    root=Path('/private/tmp/rate_anchored_potassium_20260911');assets=load_release(Path('models'))
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        torch.manual_seed(seed);model=deepcopy(assets.model);expand_history_projection(model,6)
        nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
        params=configure(model,True);optimizer_ids={id(p) for p in params}
        if len(optimizer_ids)!=len(params):raise ValueError('duplicate optimizer parameter')
        item=dict(seed=seed,arms={})
        for arm in ('old_full','clean_full'):
            state=torch.load(root/f'conflict_repair_{arm}_{seed}.pt',map_location='cpu',weights_only=False)['state_dict']
            groups=defaultdict(lambda:dict(tensors=0,trainable=0,changed=0,squared_change=0.,squared_initial=0.))
            for name,p in model.named_parameters():
                g=groups[name.split('.')[0]];delta=state[name]-p.detach()
                g['tensors']+=1;g['trainable']+=int(id(p) in optimizer_ids)
                g['changed']+=int(not torch.equal(state[name],p.detach()))
                g['squared_change']+=float(delta.double().square().sum())
                g['squared_initial']+=float(p.detach().double().square().sum())
            item['arms'][arm]=dict(groups)
        result['seeds'].append(item)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for group in ('belief_filter','variable_mixer','module_mixer','state_encoder','latent_predictor','value_heads'):
        gs=[s['arms']['clean_full'][group] for s in result['seeds']]
        print(group,'changed',[g['changed'] for g in gs],'relative_norm_median',
              float(np.median([np.sqrt(g['squared_change']/max(g['squared_initial'],1e-30)) for g in gs])))


if __name__=='__main__':main()
