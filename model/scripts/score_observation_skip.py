"""All47, all5 source assessment of a fixed observation skip; no promotion."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_observation_skip import NEW_ARMS, PARAMETERS, PRIVATE, ROOT
from whole_body_jepa.multisource_intervals import sha256


REFERENCES = ('body_probe', 'native', 'raw_probe', 'population', 'process_skip', 'shuffled_skip')
ARMS = ('observation_skip', *REFERENCES)
FAMILY = 47 * len(REFERENCES)


def losses(frame):
    if (frame.empty or frame.target.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired target frame')
    error = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    error['patient'] = frame.patient.to_numpy()
    people = error.groupby('patient', sort=True)[list(ARMS)].mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    people['component'] = links.reindex(people.index)
    return people, people.groupby('component')[list(ARMS)].mean()


def screen(stats, patients, components):
    return bool(patients >= 30 and components >= 3 and all(
        s['mean'] < 0 and s['p_less'] is not None and s['p_less'] < .05 / FAMILY
        and s['ci95'] is not None and s['ci95'][1] < 0 for s in stats.values()))


def assess(frame, seed, repetitions=5000):
    people, groups = losses(frame)
    comparisons = {}
    for i, ref in enumerate(REFERENCES):
        stats = dict(patient=paired_summary(people.observation_skip-people[ref], seed+i, repetitions),
            component=paired_summary(groups.observation_skip-groups[ref], seed+100+i, repetitions))
        comparisons[ref] = dict(paired=stats, screen=screen(stats, len(people), len(groups)),
            component_mean_harms=int((groups.observation_skip > groups[ref]).sum()))
    return dict(query_occurrences=len(frame), unique_patients=len(people), components=len(groups),
        normalized_mae=dict(patient=people[list(ARMS)].mean().to_dict(), component=groups.mean().to_dict()),
        comparisons=comparisons, full_screen=all(r['screen'] for r in comparisons.values()),
        per_component=[dict(component=str(c), patients=int((people.component == c).sum()),
                            normalized_mae=row.to_dict()) for c, row in groups.iterrows()])


def main():
    out = Path('artifacts/native_reconstruction_observation_skip_assessment_20261003.json')
    if out.exists():
        raise ValueError('Existing complete assessment')
    fp = ROOT/'partitions_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds']
    original_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    original = json.loads(original_path.read_text())
    checked = {}
    verify_bound_files(original['bindings'], checked)
    paths = [Path(__file__), Path('scripts/score_additional_reconstruction.py'), fp, original_path]
    frames, folds = [], {}
    targets = sorted(original['targets'])
    if len(targets) != 47:
        raise ValueError('Changed reference family')
    for f in range(5):
        p = Path(f'artifacts/native_reconstruction_observation_skip_fold{f}_20261003.json')
        report = json.loads(p.read_text())
        verify_bound_files(report['bindings'], checked)
        if report['parameters'] != PARAMETERS or report['fold'] != f or sorted(report['targets']) != targets:
            raise ValueError('Wrong trained family')
        dest = PRIVATE/f'observation_skip_20261003/fold{f}'
        cp = dest/'contract.json'
        contract = json.loads(cp.read_text())
        if contract['bindings'] != report['bindings']:
            raise ValueError('Wrong bound contract')
        paths.extend((p, cp))
        folds[str(f)] = {}
        for target in targets:
            meta = report['targets'][target]
            qp, mp = dest/f'{target}.parquet', dest/f'{target}.joblib'
            if (meta['prediction_sha256'] != sha256(qp) or meta['model_sha256'] != sha256(mp)
                    or meta['contract_sha256'] != sha256(cp)):
                raise ValueError('Changed model/prediction')
            frame = pd.read_parquet(qp)
            old = pd.read_parquet(PRIVATE/f'fold{f}_probes/{target}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient) != set(old.patient)
                    or any(roles[f][p] != 'selection' for p in frame.patient)
                    or not (frame.fold == f).all() or not (frame.target == target).all()
                    or len(frame) != meta['selection_patients']):
                raise ValueError('Changed query identity')
            old = old.set_index('patient').loc[frame.patient]
            if not np.array_equal(frame.component, old.component):
                raise ValueError('Changed component')
            for col in ('truth', 'native', 'body_probe', 'raw_probe', 'population'):
                if not np.array_equal(frame[col].to_numpy(), old[col].to_numpy()):
                    raise ValueError('Changed reference column')
            people, groups = losses(frame)
            means = dict(patient=people[list(ARMS)].mean().to_dict(), component=groups.mean().to_dict())
            for level in means:
                for arm in NEW_ARMS:
                    if not np.isclose(means[level][arm], meta['normalized_mae'][level][arm], atol=1e-12, rtol=1e-12):
                        raise ValueError('Changed saved metric')
            folds[str(f)][target] = means
            frames.append(frame)
            paths.extend((qp, mp))
        print('Verified observation skip fold', f, flush=True)
    data = pd.concat(frames, ignore_index=True)
    results = {}
    for i, target in enumerate(targets):
        results[target] = assess(data.loc[data.target == target], 20261003+1000*i)
        for level in ('patient', 'component'):
            for arm in ('native', 'body_probe', 'raw_probe', 'population'):
                if not np.isclose(results[target]['normalized_mae'][level][arm],
                        original['targets'][target]['normalized_mae'][level][arm], atol=1e-12, rtol=1e-12):
                    raise ValueError('Original all-fold assessment not reproduced')
        print('Scored observation skip', i+1, 'of47', flush=True)
    summary = dict(query_occurrences=len(data), unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient', 'target']].drop_duplicates()),
        full_screen=[t for t, r in results.items() if r['full_screen']],
        comparison_screens={ref: [t for t, r in results.items() if r['comparisons'][ref]['screen']]
                            for ref in REFERENCES},
        normalized_target_macro={level: {arm: float(np.mean([r['normalized_mae'][level][arm]
            for r in results.values()])) for arm in ARMS} for level in ('patient', 'component')})
    out.write_text(json.dumps(dict(summary=summary, targets=results, folds=folds,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Source-only reused development; paired normalized losses averaged within patient over overlapping folds; conditional bootstrap without model refits. Current masked reconstruction, not forecast, new lockbox, physiology-only proof or exact effective capacity equivalence.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
