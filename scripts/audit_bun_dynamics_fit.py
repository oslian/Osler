"""Full source-fit BUN body-dynamics comparison under MAE and training loss."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_bun_dynamics_branches import disabled_dynamics
from scripts.audit_continued_selection_pairs import interval
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def smooth_l1_from_error(errors):
    x = np.asarray(errors, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Finite absolute errors required')
    return np.where(x < 1, .5*x*x, x-.5)


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    mp = root/'site_balance_manifest_20260920.pt'
    cp = root/'bounded_site_training/bounded.pt'
    tp = Path('artifacts/native_reconstruction_bounded_site_training_20260920.json')
    jp = root/'joint_source_manifest.pt'
    out = Path('artifacts/native_reconstruction_bun_dynamics_fit_20260920.json')
    private = root/'bun_dynamics_fit_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing audit')
    mapping_report, training = [json.loads(p.read_text()) for p in (rp,tp)]
    if sha256(mp) != mapping_report['private_sha256']:
        raise ValueError('Manifest changed')
    for p,h in training['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Training dependency changed')
    manifest,joint,ck = [torch.load(p,weights_only=False) for p in (mp,jp,cp)]
    if ck['bindings'] != training['bindings']:
        raise ValueError('Checkpoint changed')
    variables=joint['variables']; col=variables.index('bun')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw,result={},{}
    with torch.inference_mode():
        for group in ('selected','broad'):
            rows=[];seen=set()
            for s in manifest['shards']:
                if s['group']!=group or s['role']!='fit':continue
                if s['patient'] in seen:raise ValueError('Duplicate patient')
                seen.add(s['patient'])
                path=Path(s['path'])
                if sha256(path)!=s['sha256']:raise ValueError('Sample changed')
                shard=torch.load(path,weights_only=False)
                if shard['patient']!=s['patient']:raise ValueError('Identity mismatch')
                if col not in shard['samples']:continue
                e=shard['samples'][col];predictions={}
                for name,off in (('full',False),('no_body',True)):
                    with disabled_dynamics(model.trajectory.layer,body=off):
                        predictions[name]=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],'bun')[0])
                rows.append(dict(patient=s['patient'],truth=float(e['truth_normalized']),predictions=predictions))
            raw[group]=rows;metrics={}
            for metric in ('mae','smooth_l1'):
                errors={k:np.array([abs(r['predictions'][k]-r['truth']) for r in rows]) for k in ('full','no_body')}
                if metric=='smooth_l1':errors={k:smooth_l1_from_error(v) for k,v in errors.items()}
                metrics[metric]=dict(full=float(errors['full'].mean()),no_body=float(errors['no_body'].mean()),
                    no_body_minus_full=interval(errors['no_body']-errors['full']))
            result[group]=dict(patients=len(rows),metrics=metrics)
            print(group,result[group],flush=True)
    torch.save(raw,private)
    out.write_text(json.dumps(dict(groups=result,
        bindings={str(p):sha256(p) for p in (Path(__file__),Path('scripts/audit_bun_dynamics_branches.py'),rp,mp,cp,tp,jp)},
        private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='All available source fitting BUN pairs, not exact sampled training exposure or empirical draw-weighted loss. Fixed selected checkpoint, in-sample descriptive bootstrap; no new training or generalization claim.'),indent=2)+'\n')


if __name__=='__main__':main()
