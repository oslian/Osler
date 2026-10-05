"""Describe exact frozen outer inputs, without fitting or selecting a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_support import supported_example
from whole_body_jepa.multisource_intervals import sha256


def observation_summary(example, n):
    x = example['tokens'].detach().cpu().numpy()
    times = example['event_hours'].detach().cpu().numpy()
    anchor = float(example['anchor_hour'])
    mask = x[:, n:2*n].astype(bool)
    if not len(times) or not np.isfinite(times).all() or (times > anchor).any():
        raise ValueError('Invalid event times')
    if (np.diff(times) <= 0).any():
        raise ValueError('Events must be strictly increasing')
    if not np.isin(x[:, n:2*n], [0, 1]).all():
        raise ValueError('Nonbinary mask')
    # Token ages are days at availability; add elapsed time to the anchor.
    ages = x[:, 2*n:3*n] * 24 + (anchor-times)[:, None]
    latest = np.where(mask, ages, np.inf).min(axis=0)
    valid = np.isfinite(latest)
    if not valid.any() or (latest[valid] < 0).any():
        raise ValueError('Invalid observed support')
    return dict(event_rows=len(times), observed_entries=int(mask.sum()),
                variables_seen=int(valid.sum()),
                variables_fresh_1h=int((latest <= 1).sum()),
                variables_fresh_6h=int((latest <= 6).sum()),
                latest_age_median_hours=float(np.median(latest[valid])),
                newest_observation_age_hours=float(latest[valid].min()),
                last_event_gap_hours=float(anchor-times[-1]),
                history_span_hours=float(anchor-times[0]))


def describe(frame, columns):
    result = {}
    for col in columns:
        a = frame[col].to_numpy(dtype=float)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite diagnostic')
        result[col] = dict(mean=float(a.mean()),
                          quantiles=np.quantile(a, [.1, .5, .9]).tolist())
    return dict(patients=len(frame), metrics=result)


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    parent_path = Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    output = Path('artifacts/native_reconstruction_crossfit_observation_support_20260921.json')
    if output.exists():
        raise ValueError('Existing diagnostic')
    parent = json.loads(parent_path.read_text())
    for name, digest in parent['bindings'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Changed outer dependency')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    sp = root/'encoder_fold_scaling_20260920.pt'
    jp = root/'joint_source_manifest.pt'
    plan = torch.load(fp, weights_only=False)
    scalers = torch.load(sp, weights_only=False)['scalers']
    schema = torch.load(jp, weights_only=False)
    variables = schema['variables']; n = len(variables)
    paths = [root/'encoder_crossfit_outer_20260921'/f'fold{i}.parquet' for i in range(5)]
    for i, path in enumerate(paths):
        if sha256(path) != parent['prediction_hashes'][str(i)]:
            raise ValueError('Changed outer predictions')
    predictions = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if predictions.duplicated(['patient', 'target']).any():
        raise ValueError('Duplicate prediction')
    rows = []; summary_keys = None
    by_patient = predictions.groupby('patient', sort=False)
    torch.set_num_threads(1)
    for shard in plan['shards']:
        patient = shard['patient']
        if patient not in by_patient.indices:
            continue
        pred = by_patient.get_group(patient)
        fold = int(pred.fold.iloc[0])
        if pred.fold.nunique() != 1 or plan['folds'][fold][patient] != 'test':
            raise ValueError('Wrong outer role')
        path = Path(shard['path'])
        if sha256(path) != shard['sha256']:
            raise ValueError('Changed patient shard')
        samples = torch.load(path, weights_only=False)['samples']
        for row in pred.itertuples(index=False):
            e = samples[variables.index(row.target)]
            if float(e['truth_normalized']) != row.truth:
                raise ValueError('Label mismatch')
            try:
                example = supported_example(e, scalers[fold])
            except ValueError as exc:
                if str(exc) != row.status:
                    raise
                continue
            if row.status != 'ok':
                raise ValueError('Abstention changed')
            stats = observation_summary(example, n)
            summary_keys = list(stats)
            rows.append(dict(target=row.target, group=row.group, component=row.component,
                truth=row.truth, population_error=abs(row.truth-row.population),
                process_gain=abs(row.real-row.truth)-abs(row.process-row.truth), **stats))
    frame = pd.DataFrame(rows)
    if len(frame) != parent['summary']['eligible_pairs']:
        raise ValueError('Eligible population changed')
    metrics = summary_keys + ['truth', 'population_error', 'process_gain']
    targets = {}
    for target, part in frame.groupby('target', sort=True):
        groups = {g: describe(p, metrics) for g, p in part.groupby('group')}
        components = []
        for ordinal, (_, comp) in enumerate(part.groupby('component', sort=True)):
            components.append(dict(component_ordinal=ordinal,
                groups={g: describe(p, metrics) for g, p in comp.groupby('group')}))
        targets[target] = dict(groups=groups, components=components)
    bindings = {str(p): sha256(p) for p in [Path(__file__), parent_path, fp, sp, jp, *paths]}
    output.write_text(json.dumps(dict(eligible_pairs=len(frame), targets=targets,
        bindings=bindings, runtime_modified=False, promotion_allowed=False,
        limitation='Descriptive fixed outer-input audit. Observed entries are token entries, not deduplicated clinical events. Input summaries do not establish matched physiology or a causal explanation. Labels used only for diagnosis, never routing. Normalized truths are not clinical units.'), indent=2)+'\n')
    print(json.dumps(dict(eligible_pairs=len(frame), targets=len(targets), output=str(output))))


if __name__ == '__main__':
    main()
