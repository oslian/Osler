"""Paired frozen fits on identical original source patients; no new selection."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_exposure_retention_fit_20260919.json')
    saved = root / 'exposure_retention_fit.pt'
    if out.exists() or saved.exists():
        raise ValueError('Existing diagnostic')
    tp = Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    training = json.loads(tp.read_text())
    for p, h in training['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Training changed')
    ip = root / 'joint_source_manifest.pt'
    index = torch.load(ip, weights_only=False)
    variables, targets = index['variables'], index['targets']
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path('models')))
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = 'observed_steps'
    template = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ('map',))).eval()
    models, checkpoints = {}, {}
    paths = [Path(__file__), tp, ip]
    for mode in ('small', 'expanded'):
        p = root / 'source_exposure_pair' / mode / 'real.pt'
        ck = torch.load(p, weights_only=False)
        if ck['bindings'] != training['bindings'] or ck['epoch'] != 8:
            raise ValueError('Checkpoint mismatch')
        model = deepcopy(template)
        model.load_state_dict(ck['best_state_dict'])
        models[mode], checkpoints[mode] = model, ck
        paths.append(p)
    rows = []
    with torch.inference_mode():
        for i, item in enumerate(index['shards']):
            p = Path(item['path'])
            if sha256(p) != item['sha256']:
                raise ValueError('Sample changed')
            for c, e in torch.load(p, weights_only=False)['samples'].items():
                t = variables[c]
                if t not in targets:
                    continue
                row = dict(patient=item['patient'], group=item['group'], role=item['role'], target=t,
                           truth=e['truth_normalized'])
                for mode, model in models.items():
                    row[mode] = float(model(input_tokens(e, 'real', len(variables)),
                        e['event_hours'], e['anchor_hour'], t)[0])
                    row[mode + '_prior'] = checkpoints[mode]['medians'][c]
                rows.append(row)
            if (i + 1) % 250 == 0:
                print('Replayed patients', i + 1, flush=True)
    result = {}
    for g in ('selected', 'broad'):
        result[g] = {}
        for t in targets:
            per = {}
            for role in ('fit', 'encoder_tune'):
                rr = [r for r in rows if r['group'] == g and r['target'] == t and r['role'] == role]
                y = np.array([r['truth'] for r in rr])
                scores = {k: float(abs(np.array([r[k] for r in rr]) - y).mean())
                          for k in ('small', 'expanded', 'small_prior', 'expanded_prior')}
                if role == 'encoder_tune':
                    for mode in models:
                        ck = checkpoints[mode]
                        expected = ck['log'][ck['selected_epoch'] - 1]['group_target_scores'][g][t]
                        if not np.isclose(scores[mode], expected, atol=1e-7, rtol=0):
                            raise ValueError('Selection replay mismatch')
                per[role] = dict(patients=len(rr), mae=scores,
                    expanded_minus_small=scores['expanded'] - scores['small'],
                    small_increment=scores['small'] - scores['small_prior'],
                    expanded_increment=scores['expanded'] - scores['expanded_prior'])
            result[g][t] = per
    torch.save(rows, saved)
    paths.append(saved)
    out.write_text(json.dumps(dict(groups=result, bindings={str(p): sha256(p) for p in paths},
        runtime_modified=False, limitation='Source-only paired fit/selection diagnostic; original broad fit subset is not the full expanded training population.'), indent=2) + '\n')
    for g, ts in result.items():
        print(g, 'fit-harm', sum(r['fit']['expanded_minus_small'] > 0 for r in ts.values()),
              'tune-harm', sum(r['encoder_tune']['expanded_minus_small'] > 0 for r in ts.values()))
        for t in ('glucose', 'ast', 'alt', 'troponin_i', 'bun', 'creatinine'):
            print(t, ts[t])


if __name__ == '__main__':
    main()
