"""Reproduce candidate initialization across seeds without training."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.multisource_intervals import sha256


def main():
    out=Path('artifacts/magnesium_initial_state_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    assets=load_release(Path('models'));reference=None;items=[]
    for seed in (7,19,31,43,59,71,83):
        torch.manual_seed(seed);model=deepcopy(assets.model);expand_history_projection(model,6)
        nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
        state=model.state_dict();h=hashlib.sha256()
        for name,value in state.items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            if reference is not None:torch.testing.assert_close(value,reference[name],atol=0,rtol=0)
        if reference is None:reference=deepcopy(state)
        items.append(dict(seed=seed,state_sha256=h.hexdigest(),tensor_count=len(state)))
    report=dict(complete=True,all_initial_tensors_exact=True,seeds=items,runtime_modified=False,
        binding={str(p):sha256(p) for p in (Path(__file__),Path('scripts/train_native_magnesium_matched.py'),Path('whole_body_jepa/history_event_input.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(items,indent=2))


if __name__=='__main__':main()
