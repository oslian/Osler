"""Fixed-weight integration-resolution diagnostic, not step-size tuning."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


@contextmanager
def resolution(layer, hours):
    if not np.isfinite(hours) or hours<=0:raise ValueError('Positive finite step required')
    original=layer.max_step_hours
    try:
        layer.max_step_hours=hours
        yield
    finally:layer.max_step_hours=original


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    rp=Path('artifacts/native_reconstruction_bun_full_history_20260920.json');pp=root/'bun_full_history_20260920.pt'
    cp=root/'bounded_site_training/bounded.pt';mp=root/'site_balance_manifest_20260920.pt'
    mr=Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json');jp=root/'joint_source_manifest.pt'
    out=Path('artifacts/native_reconstruction_bun_integration_resolution_20260920.json');private=root/'bun_integration_resolution_20260920.pt'
    if out.exists() or private.exists():raise ValueError('Existing diagnostic')
    prior,mapping=[json.loads(p.read_text()) for p in (rp,mr)]
    for p,h in prior['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    if sha256(pp)!=prior['private_sha256'] or sha256(mp)!=mapping['private_sha256']:raise ValueError('Private data changed')
    old,manifest,joint,ck=[torch.load(p,weights_only=False) for p in (pp,mp,jp,cp)]
    variables=joint['variables'];col=variables.index('bun');steps=(1.,.5,.25)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    if model.trajectory.layer.max_step_hours!=1.:raise ValueError('Unexpected trained integration step')
    raw,result={},{}
    with torch.inference_mode():
        for group,previous in old.items():
            store=SourceStore([s for s in manifest['shards'] if s['group']==group and s['role']=='encoder_tune'],[col])
            if set(store.labels)!={(r['patient'],col) for r in previous}:raise ValueError('Pair mismatch')
            rows=[]
            for r in previous:
                e=store[r['patient'],col];predictions={}
                for step in steps:
                    with resolution(model.trajectory.layer,step):
                        predictions[str(step)]=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],'bun')[0])
                errors={k:abs(v-e['truth_normalized']) for k,v in predictions.items()}
                if not np.isclose(errors['1.0'],r['errors']['full'],atol=1e-7,rtol=0):raise ValueError('Baseline mismatch')
                rows.append(dict(patient=r['patient'],predictions=predictions,errors=errors))
            raw[group]=rows
            result[group]=dict(patients=len(rows),mae={str(s):float(np.mean([r['errors'][str(s)] for r in rows])) for s in steps},
                versus_original={str(s):interval([r['errors'][str(s)]-r['errors']['1.0'] for r in rows]) for s in steps[1:]},
                mean_absolute_prediction_shift={f'{a}->{b}':float(np.mean([abs(r['predictions'][str(a)]-r['predictions'][str(b)]) for r in rows])) for a,b in zip(steps,steps[1:])})
            print(group,result[group],flush=True)
    torch.save(raw,private)
    out.write_text(json.dumps(dict(groups=result,steps_hours=steps,
        bindings={str(p):sha256(p) for p in (Path(__file__),rp,pp,cp,mp,mr,jp)},private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='Trained dynamics explicitly use substep delta as a feature. Changing max step changes both numerical integration and learned field inputs; not pure solver error or a validated deployment step-size choice.'),indent=2)+'\n')


if __name__=='__main__':main()
