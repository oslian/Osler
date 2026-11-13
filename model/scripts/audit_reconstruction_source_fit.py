"""Replay all fit/tune pairs of an unchanged selected reconstruction model."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_partitions import patient_roles
from whole_body_jepa.runtime import load_release


TEMP = Path('/private/tmp/rate_anchored_potassium_20260911')
BACKUP = Path('data/research_recovery/rate_anchored_potassium_20260911')


def durable(path):
    path = Path(path)
    return BACKUP / path.relative_to(TEMP) if path.is_relative_to(TEMP) else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(19, 31, 43, 59, 71, 83))
    args = parser.parse_args()
    output = Path(f'artifacts/native_reconstruction_source_fit_{args.seed}_20260917.json')
    private = Path(f'data/research_recovery/source_fit_{args.seed}_20260917')
    if output.exists() or private.exists():
        raise ValueError('existing source-fit evidence')
    training_path = Path(f'artifacts/native_reconstruction_replication_v2_{args.seed}_epoch8_20260914.json')
    evaluation_path = Path(f'artifacts/native_reconstruction_replication_v2_{args.seed}_evaluation_20260914.json')
    training = json.loads(training_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if not training['training_complete'] or not evaluation['complete']:
        raise ValueError('incomplete original experiment')
    bindings = {str(p): sha256(p) for p in (Path(__file__), training_path, evaluation_path)}
    for filename, digest in training['bindings'].items():
        p = durable(filename)
        if sha256(p) != digest:
            raise ValueError(f'changed training dependency: {filename}')
        bindings[str(p)] = digest
    root = BACKUP / 'map_native_anchor_support'
    index_path = root / 'expanded_reconstruction_20260913/index.pt'
    query_path = root / 'queries.parquet'
    # Use original report hashes even when reading recovered durable copies.
    for name in (index_path, query_path):
        candidates = [h for p, h in evaluation['bindings'].items() if Path(p).name == name.name]
        if len(candidates) != 1 or sha256(name) != candidates[0]:
            raise ValueError('original input binding mismatch')
        bindings[str(name)] = candidates[0]
    roles = patient_roles(pd.read_parquet(query_path), dict(np.load(root / f'partition_{args.seed}.npz')))
    index = torch.load(index_path, weights_only=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = load_release(Path('models'))
    variables = assets.checkpoint['variables']
    targets = [variables.index(t) for t in training['source_targets']]
    samples = []
    seen = set()
    fit_truths = {c: [] for c in targets}
    for shard in index['shards']:
        p = durable(shard['path'])
        if sha256(p) != shard['sha256']:
            raise ValueError('changed source shard')
        if roles[shard['patient']] not in ('fit', 'tune'):
            continue
        s = torch.load(p, weights_only=False)
        if s['patient'] != shard['patient']:
            raise ValueError('patient identity mismatch')
        role = roles[s['patient']]
        for c, e in s['samples'].items():
            if c not in targets:
                continue
            key = (s['patient'], c)
            if key in seen:
                raise ValueError('duplicate source patient-target')
            seen.add(key)
            samples.append((role, s['patient'], c, e))
            if role == 'fit':
                fit_truths[c].append(e['truth_normalized'])
    medians = {c: float(np.median(v)) for c, v in fit_truths.items()}
    if any(len(v) < 30 for v in fit_truths.values()):
        raise ValueError('source support changed')
    jepa = assets.model
    expand_history_projection(jepa, 6)
    jepa.belief_input_mode = 'observed_steps'
    template = NativeBodyReconstruction(NativeOrganStateResidual(jepa, variables, ('map',))).eval()
    models = {}
    for arm in ('real', 'process'):
        p = root / f'reconstruction_replication_v2_{args.seed}_20260914/{arm}.pt'
        expected = [h for f, h in evaluation['bindings'].items() if Path(f).name == p.name]
        if len(expected) != 1 or sha256(p) != expected[0]:
            raise ValueError('checkpoint hash mismatch')
        ck = torch.load(p, weights_only=False)
        if (ck['epoch'] != 8 or ck['bindings'] != training['bindings'] or
                ck['targets'] != targets or ck['variables'] != variables or
                ck['selected_epoch'] != training['arms'][arm]['selected_epoch']):
            raise ValueError('wrong selected checkpoint')
        model = deepcopy(template)
        model.load_state_dict(ck['best_state_dict'])
        models[arm] = model
        bindings[str(p)] = expected[0]
    private.mkdir()
    rows = {r: [] for r in ('fit', 'tune')}
    with torch.no_grad():
        for i, (role, patient, c, e) in enumerate(samples):
            predictions = {arm: float(model(input_tokens(e, arm, len(variables)),
                e['event_hours'], e['anchor_hour'], variables[c])[0]) for arm, model in models.items()}
            rows[role].append(dict(subject_id=patient, target=variables[c],
                truth=e['truth_normalized'], population=medians[c], **predictions))
            if (i+1) % 2000 == 0:
                print('source replay', i+1, '/', len(samples), flush=True)
    saved = private / 'source_predictions.pt'
    torch.save(rows, saved)
    result = {}
    for role, records in rows.items():
        result[role] = {}
        for target, frame in pd.DataFrame(records).groupby('target'):
            errors = {arm: abs(frame[arm].to_numpy()-frame.truth.to_numpy())
                      for arm in ('real', 'process', 'population')}
            result[role][target] = dict(patients=len(frame),
                normalized_mae={a: float(v.mean()) for a, v in errors.items()},
                real_minus_process=float((errors['real']-errors['process']).mean()),
                real_minus_population=float((errors['real']-errors['population']).mean()))
    for arm in ('real', 'process'):
        reproduced = np.mean([r['normalized_mae'][arm] for r in result['tune'].values()])
        if not np.isclose(reproduced, training['arms'][arm]['best_tune'], rtol=0, atol=1e-7):
            raise ValueError('selected tune score did not reproduce')
    bindings[str(saved)] = sha256(saved)
    output.write_text(json.dumps(dict(complete=True, seed=args.seed, roles=result,
        source_unsupported=training['source_unsupported'], bindings=bindings,
        tune_score_reproduced=True, runtime_modified=False, promotion_allowed=False,
        limitation='Descriptive selected-checkpoint fit/tune errors. Tune selected the checkpoints; not independent validation or formal significance. No new model fitting or target selection.'),
        indent=2, allow_nan=False)+'\n')
    print('complete', output, flush=True)


if __name__ == '__main__':
    main()
