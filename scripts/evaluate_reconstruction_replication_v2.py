"""Evaluate completed replication arms using their own patient partition."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_partitions import patient_roles
from whole_body_jepa.runtime import load_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(19,31,43,59,71,83))
    args = parser.parse_args()
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private = root / f'reconstruction_replication_v2_{args.seed}_20260914'
    source = Path(f'artifacts/native_reconstruction_replication_v2_{args.seed}_epoch8_20260914.json')
    out = Path(f'artifacts/native_reconstruction_replication_v2_{args.seed}_evaluation_20260914.json')
    if out.exists() or (private / 'heldout_predictions.pt').exists():
        raise ValueError('existing evaluation evidence')
    trained = json.loads(source.read_text())
    if not trained['training_complete'] or trained['seed'] != args.seed:
        raise ValueError('incomplete or wrong training')
    for p, h in trained['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError(f'changed dependency: {p}')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    targets = [variables.index(v) for v in trained['source_targets']]
    queries = root / 'queries.parquet'
    partition = root / f'partition_{args.seed}.npz'
    roles = patient_roles(pd.read_parquet(queries), dict(np.load(partition)))
    index_path = root / 'expanded_reconstruction_20260913/index.pt'
    index = torch.load(index_path, weights_only=False)
    if set(roles) != {s['patient'] for s in index['shards']}:
        raise ValueError('patient inventory mismatch')
    fit = {c: [] for c in targets}
    heldout = []
    seen = set()
    for shard in index['shards']:
        if sha256(Path(shard['path'])) != shard['sha256']:
            raise ValueError('changed shard')
        sample = torch.load(shard['path'], weights_only=False)
        patient = sample['patient']
        if patient != shard['patient']:
            raise ValueError('patient identity mismatch')
        # Stored shard roles belong to the old seed, not this replication.
        role = roles[patient]
        for c, example in sample['samples'].items():
            if c not in fit:
                continue
            key = (patient, c)
            if key in seen:
                raise ValueError('duplicate patient target')
            seen.add(key)
            if role == 'fit':
                fit[c].append(example['truth_normalized'])
            elif role in ('evaluation', 'hospital'):
                heldout.append((role, patient, c, example))
    medians = {c: float(np.median(v)) for c, v in fit.items()}
    if any(len(v) < 30 for v in fit.values()):
        raise ValueError('source support mismatch')
    jepa = assets.model
    expand_history_projection(jepa, 6)
    jepa.belief_input_mode = 'observed_steps'
    template = NativeBodyReconstruction(NativeOrganStateResidual(jepa, variables, ('map',))).eval()
    models = {}
    paths = []
    for arm in ('real', 'process', 'placebo'):
        path = private / f'{arm}.pt'
        ck = torch.load(path, weights_only=False)
        if (ck['epoch'] != 8 or ck['bindings'] != trained['bindings']
                or ck['targets'] != targets or ck['variables'] != variables
                or ck['selected_epoch'] != trained['arms'][arm]['selected_epoch']):
            raise ValueError('checkpoint does not match completed training')
        model = deepcopy(template)
        model.load_state_dict(ck['best_state_dict'])
        models[arm] = model
        paths.append(path)
    rows = {r: [] for r in ('evaluation', 'hospital')}
    with torch.no_grad():
        for i, (role, patient, c, e) in enumerate(heldout):
            predictions = {arm: float(model(input_tokens(e, arm, len(variables)),
                           e['event_hours'], e['anchor_hour'], variables[c])[0])
                           for arm, model in models.items()}
            rows[role].append(dict(subject_id=patient, target=variables[c],
                                  truth=e['truth_normalized'], population=medians[c], **predictions))
            if (i+1) % 2000 == 0:
                print('evaluated', i+1, '/', len(heldout), flush=True)
    result = {}
    for role, records in rows.items():
        result[role] = {}
        for target, part in pd.DataFrame(records).groupby('target'):
            p = {a: part[a].to_numpy() for a in ('real', 'process', 'placebo', 'population')}
            result[role][target] = dict(patients=len(part),
                normalized_mae={a: float(np.mean(abs(v-part.truth.to_numpy()))) for a,v in p.items()},
                comparisons={ref: paired_model_errors(part, p, ref, seed=20260914)
                             for ref in ('population', 'process', 'placebo')})
    saved = private / 'heldout_predictions.pt'
    torch.save(rows, saved)
    dependencies = [Path(__file__), source, queries, partition, index_path, saved,
                    Path('whole_body_jepa/change_evaluation.py'), *paths]
    out.write_text(json.dumps(dict(complete=True, seed=args.seed, roles=result,
        source_unsupported=trained['source_unsupported'], runtime_modified=False, promotion_allowed=False,
        limitation='Nominal paired intervals; repeated cohort and same hospital holdout. Current reconstruction only.',
        bindings={str(p): sha256(p) for p in dependencies}), indent=2)+'\n')
    print('complete', out, flush=True)


if __name__ == '__main__':
    main()
