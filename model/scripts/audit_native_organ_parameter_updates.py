"""Compare retained initial/final parameters without loading patient records."""
import json
from pathlib import Path
import torch
from whole_body_jepa.multisource_intervals import sha256


def main():
    output=Path('artifacts/native_organ_parameter_updates_20260913.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    report=Path('artifacts/native_organ_replay_trace_20260913.json')
    trace=json.loads(report.read_text())
    for name,h in trace['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    groups={'local_observation':('jepa.token_projection.','jepa.variable_embedding','jepa.module_token_embedding.'),
        'organ_update':('layer.organ_update_gate.',),'organ_dynamics':('layer.organ_dynamics.','layer.organ_message.'),
        'body_update':('layer.body_update_gate.',),'body_dynamics':('layer.body_dynamics.',),
        'value_head':('value_heads.',),'latent_head':('latent_heads.',)}
    results={};paths=[report,Path(__file__)]
    for arm in ('none','sparse'):
        first=root/f'canonical_replay_trace_selected_{arm}_7.pt';last=root/f'canonical_replay_trace_final_{arm}_7.pt'
        a=torch.load(first,map_location='cpu',weights_only=False)['state_dict']
        b=torch.load(last,map_location='cpu',weights_only=False)['state_dict'];paths.extend((first,last))
        assert trace['logs'][arm]['selected_epoch']==0
        results[arm]={}
        for group,prefixes in groups.items():
            keys=[k for k in a if k.startswith(prefixes) and a[k].is_floating_point()]
            assert keys
            initial=torch.cat([a[k].flatten().double() for k in keys]);final=torch.cat([b[k].flatten().double() for k in keys])
            difference=final-initial
            results[arm][group]=dict(tensors=len(keys),changed_tensors=sum(not torch.equal(a[k],b[k]) for k in keys),
                elements=initial.numel(),initial_norm=float(initial.norm()),update_norm=float(difference.norm()),
                relative_update=float(difference.norm()/initial.norm().clamp_min(1e-30)))
    output.write_text(json.dumps(dict(complete=True,results=results,runtime_modified=False,training_performed=False,
        caveat='Parameter movement is not predictive utility; includes weight decay and different parameter scales.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
