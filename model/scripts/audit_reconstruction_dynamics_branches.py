"""All-trained-target source diagnostic of shared reconstruction dynamics."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_bun_dynamics_branches import disabled_dynamics
from scripts.audit_continued_selection_pairs import interval
from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def summarize(rows, targets):
    keys = [(r['patient'], r['target']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate patient-target pairs')
    result = {}
    for target in targets:
        selected = [r for r in rows if r['target'] == target]
        if not selected:
            raise ValueError('Missing target')
        result[target] = dict(patients=len(selected),
            mae={k: float(np.mean([r['errors'][k] for r in selected])) for k in ('full', 'no_body', 'no_organs')},
            versus_full={k: interval([r['errors'][k]-r['errors']['full'] for r in selected]) for k in ('no_body', 'no_organs')})
    return result


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    rp = Path('artifacts/native_reconstruction_history_suffix_20260920.json')
    pp = root/'history_suffix_20260920.pt'
    cp = root/'bounded_site_training/bounded.pt'
    paths = (root/'joint_source_manifest.pt', root/'full_samples/index.pt',
             Path('data/research_recovery/reconstruction_nested_selection_20260917.pt'))
    out = Path('artifacts/native_reconstruction_dynamics_branches_20260920.json')
    private = root/'dynamics_branches_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing diagnostic')
    prior = json.loads(rp.read_text())
    for p, h in prior['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed dependency')
    if sha256(pp) != prior['private_sha256']:
        raise ValueError('Private rows changed')
    saved = torch.load(pp, weights_only=False)
    joint, full, split = [torch.load(p, weights_only=False) for p in paths]
    ck = torch.load(cp, weights_only=False)
    variables, targets = joint['variables'], joint['targets']
    cols = [variables.index(t) for t in targets]
    shards = source_shards(joint, full, 'expanded', split['roles'])
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6); assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    raw, output = {}, {}
    with torch.inference_mode():
        for group in ('selected','broad'):
            store = SourceStore([s for s in shards if s['group']==group and s['role']=='encoder_tune'], cols)
            old = {(r['patient'], variables.index(r['target'])):r for r in saved[group]}
            if set(old) != set(store.labels):
                raise ValueError('Pair mismatch')
            rows = []
            for key, truth in store.labels.items():
                e = store[key]; target = variables[key[1]]; errors = {}
                for name, body, organs in (('full',False,False),('no_body',True,False),('no_organs',False,True)):
                    with disabled_dynamics(model.trajectory.layer, body, organs):
                        p = float(model(e['tokens'],e['event_hours'],e['anchor_hour'],target)[0])
                    errors[name] = abs(p-truth)
                if not np.isclose(errors['full'],old[key]['errors']['full'],atol=1e-7,rtol=0):
                    raise ValueError('Original patient replay mismatch')
                rows.append(dict(patient=key[0],target=target,errors=errors))
            raw[group], output[group] = rows, summarize(rows,targets)
            print(group,len(rows),'complete',flush=True)
    torch.save(raw,private)
    out.write_text(json.dumps(dict(groups=output,trained_targets=targets,
        bindings={str(p):sha256(p) for p in (Path(__file__),Path('scripts/audit_bun_dynamics_branches.py'),rp,pp,cp,*paths)},
        private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='Original source-selection90+90patients, not full3323source replay. Fixed checkpoint; unadjusted post-selection paired intervals. No learned gate, matched training, future forecast or new external validation.'),indent=2)+'\n')
    for group, per in output.items():
        print(group,{k:float(np.mean([v['mae'][k] for v in per.values()])) for k in ('full','no_body','no_organs')})


if __name__=='__main__':
    main()
