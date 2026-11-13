"""Factorial diagnostic of old observation updates and elapsed-time dynamics."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def replay_factors(model, tokens, event_hours, anchor, target):
    tr = model.trajectory
    clean, times = mask_native_target(tokens, event_hours, tr.variables, target)
    body, organs, presence = encode_native_organ_observations(tr.jepa, tr.variables, clean)
    n = len(tr.variables)
    keep = torch.zeros(len(times), dtype=torch.bool, device=times.device)
    for col in range(n):
        indices = clean[:, n+col].nonzero().flatten()
        if len(indices):
            keep[indices[-1]] = True
    relative = (times-times[0]).to(clean.dtype)
    outputs = {}
    for name, old_updates, dynamics in (('full', True, True), ('no_dynamics', True, False),
                                      ('latest_updates_fixed_clock', False, True), ('latest_no_dynamics', False, False)):
        state = tr.layer.initialize(clean.new_zeros((1, tr.jepa.latent)),
            clean.new_zeros((1, len(tr.organ_names), tr.jepa.token_dim)),
            clean.new_zeros((1, len(tr.organ_names))))
        for i in range(len(times)):
            delta = relative[i:i+1]-state.elapsed_hours
            state = tr.layer.advance(state, delta) if dynamics else replace(state, elapsed_hours=relative[i:i+1])
            if old_updates or bool(keep[i]):
                state = tr.layer.update(state, body[i:i+1], organs[i:i+1], presence[i:i+1], delta)
        gap = (torch.as_tensor(anchor, device=times.device, dtype=times.dtype)-times[-1]).to(clean.dtype).reshape(1)
        state = tr.layer.advance(state, gap) if dynamics else replace(state, elapsed_hours=state.elapsed_hours+gap)
        outputs[name] = float(model.readout(state.body)[0, tr.variables.index(target)])
    return outputs


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    prior_path = Path('artifacts/native_reconstruction_bun_full_history_20260920.json')
    rows_path = root/'bun_full_history_20260920.pt'
    mp = root/'site_balance_manifest_20260920.pt'
    mr = Path('artifacts/native_reconstruction_site_balance_preflight_20260920.json')
    cp = root/'bounded_site_training/bounded.pt'
    jp = root/'joint_source_manifest.pt'
    out = Path('artifacts/native_reconstruction_bun_history_dynamics_20260920.json')
    private = root/'bun_history_dynamics_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing diagnostic')
    prior, mapping_report = [json.loads(p.read_text()) for p in (prior_path, mr)]
    if sha256(rows_path) != prior['private_sha256'] or sha256(mp) != mapping_report['private_sha256']:
        raise ValueError('Changed data')
    for p, h in prior['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed dependency')
    saved = torch.load(rows_path, weights_only=False)
    manifest, joint, ck = [torch.load(p, weights_only=False) for p in (mp, jp, cp)]
    variables = joint['variables']; col = variables.index('bun')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6); assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw, result = {}, {}
    with torch.inference_mode():
        for group, old in saved.items():
            store = SourceStore([s for s in manifest['shards'] if s['group'] == group and s['role'] == 'encoder_tune'], [col])
            if set(store.labels) != {(r['patient'], col) for r in old}:
                raise ValueError('Patient mismatch')
            rows = []
            for r in old:
                e = store[r['patient'], col]
                predictions = replay_factors(model, e['tokens'], e['event_hours'], e['anchor_hour'], 'bun')
                errors = {k: abs(v-e['truth_normalized']) for k,v in predictions.items()}
                if not np.isclose(errors['full'], r['errors']['full'], atol=1e-7, rtol=0):
                    raise ValueError('Full patient replay mismatch')
                rows.append(dict(patient=r['patient'], errors=errors))
            raw[group] = rows
            result[group] = dict(patients=len(rows), mae={k: float(np.mean([r['errors'][k] for r in rows])) for k in predictions},
                versus_full={k: interval([r['errors'][k]-r['errors']['full'] for r in rows]) for k in predictions if k != 'full'},
                latest_effect_without_dynamics=interval([r['errors']['latest_no_dynamics']-r['errors']['no_dynamics'] for r in rows]))
            print(group, result[group], flush=True)
    torch.save(raw, private)
    out.write_text(json.dumps(dict(groups=result, bindings={str(p):sha256(p) for p in (Path(__file__), prior_path, rows_path, mp, mr, cp, jp)},
        private_sha256=sha256(private), runtime_modified=False, promotion_allowed=False,
        limitation='Source-only fixed-weight path intervention, original event grid, initialization and update deltas. '
        'Unadjusted post-selection intervals; input/state distribution shift, not matched training or physiological causality.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
