"""Decay-only candidate compared with its frozen accumulated reference."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_decay_evaluation_20260919.json')
    private = root / 'decay_matched_evaluation'
    if out.exists() or private.exists():
        raise ValueError('Existing evaluation')
    tp = Path('artifacts/native_reconstruction_decay_matched_source_20260919.json')
    training = json.loads(tp.read_text())
    for p, h in training['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Training changed')
    rp = Path('artifacts/native_reconstruction_accumulated_evaluation_20260919.json')
    parent = json.loads(rp.read_text())
    for p, h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Evaluation reference changed')
    cp = root / 'decay_matched_training' / (training['source_selected_arm'] + '.pt')
    ck = torch.load(cp, weights_only=False)
    if ck['bindings'] != training['bindings'] or ck['epoch'] != 8:
        raise ValueError('Checkpoint mismatch')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models'))); variables = assets.checkpoint['variables']
    expand_history_projection(assets.model, 6); assets.model.belief_input_mode = 'observed_steps'
    model = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    indices = {'broad': Path('data/research_recovery/expanded_confirmation_20260917/samples/index.pt'),
               'selected': BACKUP / 'map_native_anchor_support/expanded_reconstruction_20260913/index.pt'}
    paths = [Path(__file__), tp, rp, cp, *indices.values()]; private.mkdir(); results = {}
    for group, ip in indices.items():
        refpath = root / 'accumulated_evaluation' / f'{group}.pt'; paths.append(refpath)
        refs = {(r['patient'], r['target']): r for r in torch.load(refpath, weights_only=False)}
        patients = {k[0] for k in refs}; rows = []; seen = set()
        with torch.inference_mode():
            for item in torch.load(ip, weights_only=False)['shards']:
                if item['patient'] not in patients:
                    continue
                p = durable(item['path'])
                if sha256(p) != item['sha256']:
                    raise ValueError('Sample changed')
                for c, e in torch.load(p, weights_only=False)['samples'].items():
                    key = (item['patient'], variables[c])
                    if key not in refs:
                        continue
                    if key in seen or refs[key]['truth'] != e['truth_normalized']:
                        raise ValueError('Pair mismatch')
                    seen.add(key)
                    row = dict(refs[key]); row['original_accumulated'] = row.pop('candidate'); row['candidate'] = float(model(input_tokens(e, 'real', len(variables)),
                        e['event_hours'], e['anchor_hour'], variables[c])[0]); rows.append(row)
        if seen != set(refs):
            raise ValueError('Incomplete pairs')
        path = private / f'{group}.pt'; torch.save(rows, path); paths.append(path)
        per = {}
        for t in parent['groups'][group]:
            rr = [r for r in rows if r['target'] == t]
            if not rr:
                per[t] = dict(status='no_trained_source_head'); continue
            y = np.array([r['truth'] for r in rr]); sites = [r['site'] for r in rr]
            errors = {k: abs(np.array([r[k] for r in rr]) - y) for k in
                      ('old', 'small_real', 'expanded_real', 'expanded_population', 'original_accumulated', 'candidate')}
            checks = {k: evidence(errors['candidate']-v, sites) for k, v in errors.items() if k != 'candidate'}
            per[t] = dict(patients=len(rr), mae={k: float(v.mean()) for k, v in errors.items()},
                          comparisons=checks, descriptive_comparisons_pass=confirmed(checks, len(rr)))
        results[group] = per
        usable = [r for r in per.values() if 'mae' in r]
        print(group, 'macro', {k: float(np.mean([r['mae'][k] for r in usable])) for k in usable[0]['mae']},
              'better expanded', sum(r['mae']['candidate'] < r['mae']['expanded_real'] for r in usable), flush=True)
    out.write_text(json.dumps(dict(groups=results, bindings={str(p): sha256(p) for p in paths},
        runtime_modified=False, promotion_allowed=False,
        limitation='Source-selected real arm only; lacks matched accumulated process/placebo controls. Reused cohorts, current reconstruction only.'), indent=2)+'\n')


if __name__ == '__main__':
    main()

