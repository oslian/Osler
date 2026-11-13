"""Fixed-weight body-versus-organ dynamics ablation; never a serving policy."""
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
def disabled_dynamics(layer, body=False, organs=False):
    handles = []
    try:
        for disabled, module in ((body, layer.body_dynamics), (organs, layer.organ_dynamics)):
            if disabled:
                handles.append(module.register_forward_hook(lambda module, args, output: torch.zeros_like(output)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_bun_history_dynamics_20260920.json')
    pp = root/'bun_history_dynamics_20260920.pt'
    mp = root/'site_balance_manifest_20260920.pt'
    mr = Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    cp = root/'bounded_site_training/bounded.pt'
    jp = root/'joint_source_manifest.pt'
    out = Path('artifacts/native_reconstruction_bun_dynamics_branches_20260920.json')
    private = root/'bun_dynamics_branches_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing diagnostic')
    prior, mapping_report = [json.loads(p.read_text()) for p in (rp, mr)]
    for p, h in prior['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed dependency')
    if sha256(pp) != prior['private_sha256'] or sha256(mp) != mapping_report['private_sha256']:
        raise ValueError('Private data changed')
    saved, manifest, joint, ck = [torch.load(p, weights_only=False) for p in (pp, mp, jp, cp)]
    variables = joint['variables']; col = variables.index('bun')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6); assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw, result = {}, {}
    with torch.inference_mode():
        for group, previous in saved.items():
            store = SourceStore([s for s in manifest['shards'] if s['group'] == group and s['role'] == 'encoder_tune'], [col])
            if set(store.labels) != {(r['patient'], col) for r in previous}:
                raise ValueError('Patient mismatch')
            rows = []
            replay_difference = {'full': 0., 'neither': 0.}
            for r in previous:
                e = store[r['patient'], col]
                errors = {}
                for name, body, organs in (('full',False,False),('no_body',True,False),('no_organs',False,True),('neither',True,True)):
                    with disabled_dynamics(model.trajectory.layer, body, organs):
                        prediction = float(model(e['tokens'], e['event_hours'], e['anchor_hour'], 'bun')[0])
                    errors[name] = abs(prediction-e['truth_normalized'])
                for name, old in (('full','full'),('neither','no_dynamics')):
                    difference = abs(errors[name]-r['errors'][old])
                    replay_difference[name] = max(replay_difference[name], difference)
                    # The old no-dynamics loop directly assigns elapsed time;
                    # advance retains float32 substep clock accumulation.
                    tolerance = 1e-7 if name == 'full' else 1e-5
                    if difference > tolerance:
                        raise ValueError(f'Prior replay mismatch: {name}, difference={difference}')
                rows.append(dict(patient=r['patient'], errors=errors))
            raw[group] = rows
            result[group] = dict(patients=len(rows), mae={k:float(np.mean([r['errors'][k] for r in rows])) for k in errors},
                maximum_prior_replay_difference=replay_difference,
                versus_full={k:interval([r['errors'][k]-r['errors']['full'] for r in rows]) for k in errors if k!='full'},
                maximum_no_body_vs_neither_error_difference=max(abs(r['errors']['no_body']-r['errors']['neither']) for r in rows))
            print(group, result[group], flush=True)
    torch.save(raw, private)
    out.write_text(json.dumps(dict(groups=result,
        bindings={str(p):sha256(p) for p in (Path(__file__), rp, pp, mp, mr, cp, jp)}, private_sha256=sha256(private),
        runtime_modified=False, promotion_allowed=False,
        limitation='Fixed trained weights, source-selection BUN, unadjusted paired intervals. Hooks zero dynamics output only, leaving updates, clocks and uncertainty growth intact. Not matched training or physiological causality.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
