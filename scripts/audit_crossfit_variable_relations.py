"""Variable-level support and rank relations on frozen outer examples."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from scripts.encoder_fold_support import supported_example
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.nowcast import related_siblings


def latest_observations(example, scaler):
    n = len(scaler['counts'])
    x = example['tokens'].detach().cpu().numpy()
    t = example['event_hours'].detach().cpu().numpy()
    anchor = float(example['anchor_hour'])
    if not len(t) or not np.isfinite(t).all() or (t > anchor).any() or (np.diff(t) <= 0).any():
        raise ValueError('Invalid history')
    mask = x[:, n:2*n]
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Invalid mask')
    value = np.full(n, np.nan); age = np.full(n, np.nan)
    for c in np.flatnonzero(mask.any(axis=0)):
        row = np.flatnonzero(mask[:, c])[-1]
        value[c] = x[row, c]*scaler['scale'][c]+scaler['location'][c]
        age[c] = float(x[row, 2*n+c])*24+anchor-t[row]
    present = np.isfinite(value)
    if not np.isfinite(age[present]).all() or (age[present] < 0).any():
        raise ValueError('Invalid measurement age')
    return value, age


def relation(values, ages, truth):
    observed = np.isfinite(values)
    x = values[observed]; y = truth[observed]; a = ages[observed]
    n = len(x)
    valid = n >= 30 and np.unique(x).size > 1 and np.unique(y).size > 1
    rho = float(spearmanr(x, y).statistic) if valid else None
    return dict(patients=len(truth), observed=n,
                observed_fraction=float(observed.mean()),
                age_median_hours=float(np.median(a)) if n else None,
                fresh_1h_fraction=float((observed & (ages <= 1)).mean()),
                rho=rho, association_supported=valid)


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_crossfit_variable_relations_20260921.json')
    if out.exists():
        raise ValueError('Existing diagnostic')
    rp = Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    parent = json.loads(rp.read_text())
    for p, h in parent['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed outer dependency')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    sp = root/'encoder_fold_scaling_20260920.pt'
    jp = root/'joint_source_manifest.pt'
    plan = torch.load(fp, weights_only=False)
    scalers = torch.load(sp, weights_only=False)['scalers']
    variables = torch.load(jp, weights_only=False)['variables']
    paths = [root/'encoder_crossfit_outer_20260921'/f'fold{i}.parquet' for i in range(5)]
    for i, p in enumerate(paths):
        if sha256(p) != parent['prediction_hashes'][str(i)]:
            raise ValueError('Changed predictions')
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if frame.duplicated(['patient', 'target']).any():
        raise ValueError('Duplicate outer example')
    grouped = frame.groupby('patient', sort=False)
    values = np.full((len(frame), len(variables)), np.nan)
    ages = np.full_like(values, np.nan)
    processed = np.zeros(len(frame), bool)
    torch.set_num_threads(1)
    for shard in plan['shards']:
        patient = shard['patient']
        if patient not in grouped.indices:
            continue
        p = Path(shard['path'])
        if sha256(p) != shard['sha256']:
            raise ValueError('Changed shard')
        samples = torch.load(p, weights_only=False)['samples']
        for row in grouped.get_group(patient).itertuples():
            if plan['folds'][row.fold][patient] != 'test':
                raise ValueError('Not outer test')
            e = samples[variables.index(row.target)]
            if e['truth_normalized'] != row.truth:
                raise ValueError('Different label')
            try:
                changed = supported_example(e, scalers[row.fold])
            except ValueError as exc:
                if str(exc) != row.status:
                    raise
                continue
            if row.status != 'ok' or processed[row.Index]:
                raise ValueError('Changed status or repeated example')
            v, a = latest_observations(changed, scalers[row.fold])
            hidden = {row.target} | related_siblings(row.target)
            if any(np.isfinite(v[c]) for c, name in enumerate(variables) if name in hidden):
                raise ValueError('Target/sibling input leak')
            values[row.Index] = v; ages[row.Index] = a; processed[row.Index] = True
    if not np.array_equal(processed, frame.status.eq('ok').to_numpy()):
        raise ValueError('Eligible population mismatch')
    result = {}
    for target, part in frame.loc[processed].groupby('target', sort=True):
        report = {}
        scopes = [('all', part)] + [(f'component_{i}', p) for i, (_, p) in
                                     enumerate(part.groupby('component', sort=True))]
        for scope, subset in scopes:
            report[scope] = {}
            for group, batch in subset.groupby('group'):
                idx = batch.index.to_numpy(); y = batch.truth.to_numpy()
                report[scope][group] = {name: relation(values[idx, c], ages[idx, c], y)
                                        for c, name in enumerate(variables)}
        result[target] = report
    out.write_text(json.dumps(dict(targets=result, eligible_pairs=int(processed.sum()),
        bindings={str(p): sha256(p) for p in [Path(__file__), rp, fp, sp, jp, *paths]},
        runtime_modified=False, promotion_allowed=False,
        limitation='Post-evaluation descriptive associations only. Latest available observed value, ages at anchor; original common normalized value units. Correlations conditional on being observed and not adjusted for case mix or multiple comparisons. No features, models, gates or patients selected from these outcomes.'), indent=2)+'\n')
    print(json.dumps(dict(targets=len(result), eligible_pairs=int(processed.sum()), output=str(out))))


if __name__ == '__main__':
    main()
