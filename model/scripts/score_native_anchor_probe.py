"""All-target source assessment, native-anchored real-label readout contrast."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import paired_summary
from scripts.score_monitored_continuation_outer import verify_bound_files
from scripts.train_native_anchor_probe import DEST, NEW_ARMS, PARAMETERS, PRIVATE, ROOT
from whole_body_jepa.multisource_intervals import sha256


REFERENCES = ('body_probe', 'native', 'raw_probe', 'population', 'absolute_anchor',
              'anchor_only', 'shuffled_anchor', 'observation_skip')
ARMS = ('anchored_residual', *REFERENCES)
FAMILY = 47*len(REFERENCES)


def losses(frame):
    if (frame.empty or frame.target.nunique() != 1 or frame.duplicated(['patient', 'fold']).any()
            or frame.groupby('patient').component.nunique().max() != 1
            or not np.isfinite(frame[['truth', *ARMS]].to_numpy()).all()):
        raise ValueError('Invalid paired frame')
    errors = frame[list(ARMS)].subtract(frame.truth, axis=0).abs()
    errors['patient'] = frame.patient.to_numpy()
    people = errors.groupby('patient', sort=True)[list(ARMS)].mean()
    links = frame.drop_duplicates('patient').set_index('patient').component
    people['component'] = links.reindex(people.index)
    return people, people.groupby('component')[list(ARMS)].mean()


def assess(frame, seed, repetitions=5000):
    people, groups = losses(frame)
    comparisons = {}
    for i, ref in enumerate(REFERENCES):
        paired = dict(patient=paired_summary(people.anchored_residual-people[ref], seed+i, repetitions),
            component=paired_summary(groups.anchored_residual-groups[ref], seed+100+i, repetitions))
        passed = len(people) >= 30 and len(groups) >= 3 and all(
            s['mean'] < 0 and s['p_less'] is not None and s['p_less'] < .05/FAMILY
            and s['ci95'] is not None and s['ci95'][1] < 0 for s in paired.values())
        comparisons[ref] = dict(paired=paired, screen=bool(passed),
            component_mean_harms=int((groups.anchored_residual > groups[ref]).sum()))
    return dict(query_occurrences=len(frame), unique_patients=len(people), components=len(groups),
        normalized_mae=dict(patient=people[list(ARMS)].mean().to_dict(), component=groups.mean().to_dict()),
        comparisons=comparisons, full_screen=all(c['screen'] for c in comparisons.values()),
        per_component=[dict(component=str(c), patients=int((people.component == c).sum()),
                            normalized_mae=row.to_dict()) for c,row in groups.iterrows()])


def main():
    out = Path('artifacts/native_reconstruction_native_anchor_assessment_20261003.json')
    if out.exists():
        raise ValueError('Existing complete assessment')
    fp = ROOT/'partitions_20260922.pt'
    roles = torch.load(fp, weights_only=False)['folds']
    original_path = Path('artifacts/native_reconstruction_nonlinear_source_assessment_20261001.json')
    skip_path = Path('artifacts/native_reconstruction_observation_skip_assessment_20261003.json')
    original, skip = [json.loads(p.read_text()) for p in (original_path, skip_path)]
    checked = {}
    for report in (original, skip):
        verify_bound_files(report['bindings'], checked)
    targets = sorted(original['targets'])
    if len(targets) != 47 or targets != sorted(skip['targets']):
        raise ValueError('Changed target family')
    paths = [Path(__file__), Path('scripts/score_additional_reconstruction.py'), fp, original_path, skip_path]
    frames, fits = [], {}
    for f in range(5):
        p = Path(f'artifacts/native_reconstruction_native_anchor_fold{f}_20261003.json')
        report = json.loads(p.read_text())
        verify_bound_files(report['bindings'], checked)
        if report['fold'] != f or report['parameters'] != PARAMETERS or sorted(report['targets']) != targets:
            raise ValueError('Wrong completed fold')
        cp = DEST/f'fold{f}/contract.json'
        if json.loads(cp.read_text())['bindings'] != report['bindings']:
            raise ValueError('Changed final contract')
        paths.extend((p, cp)); fits[str(f)] = {}
        for target, meta in report['targets'].items():
            qp, mp = [DEST/f'fold{f}/{target}.{suffix}' for suffix in ('parquet', 'joblib')]
            if (meta['contract_sha256'] != sha256(cp) or meta['prediction_sha256'] != sha256(qp)
                    or meta['model_sha256'] != sha256(mp)):
                raise ValueError('Changed candidate artifact')
            frame = pd.read_parquet(qp)
            old = pd.read_parquet(PRIVATE/f'fold{f}_probes/{target}.parquet')
            if (frame.patient.duplicated().any() or set(frame.patient) != set(old.patient)
                    or any(roles[f][p] != 'selection' for p in frame.patient)
                    or not (frame.fold == f).all() or not (frame.target == target).all()
                    or len(frame) != meta['selection_patients']):
                raise ValueError('Changed query identity')
            old = old.set_index('patient').loc[frame.patient]
            for col in ('truth', 'native', 'body_probe', 'raw_probe', 'population', 'component'):
                if not np.array_equal(frame[col].to_numpy(), old[col].to_numpy()):
                    raise ValueError('Changed reference')
            frames.append(frame); paths.extend((qp, mp)); fits[str(f)][target] = meta['fit_truth_mae']
    data = pd.concat(frames, ignore_index=True)
    results = {}
    for i,target in enumerate(targets):
        r = assess(data.loc[data.target == target], 20261003+i*1000)
        for level in ('patient', 'component'):
            for arm in ('native', 'body_probe', 'raw_probe', 'population', 'observation_skip'):
                parent = skip if arm == 'observation_skip' else original
                if not np.isclose(r['normalized_mae'][level][arm],
                    parent['targets'][target]['normalized_mae'][level][arm], atol=1e-12, rtol=1e-12):
                    raise ValueError('All-fold reference mismatch')
        results[target] = r
        print('Scored native anchor', i+1, 'of47', flush=True)
    summary = dict(query_occurrences=len(data), unique_patients=int(data.patient.nunique()),
        unique_patient_target_pairs=len(data[['patient', 'target']].drop_duplicates()),
        full_screen=[t for t,r in results.items() if r['full_screen']],
        comparison_screens={ref: [t for t,r in results.items() if r['comparisons'][ref]['screen']]
                            for ref in REFERENCES},
        normalized_target_macro={level: {arm: float(np.mean([r['normalized_mae'][level][arm]
            for r in results.values()])) for arm in ARMS} for level in ('patient', 'component')})
    if len(data) != original['summary']['query_occurrences']:
        raise ValueError('Incomplete source grid')
    out.write_text(json.dumps(dict(summary=summary, targets=results, fit_truth=fits,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation='Source-only reused development; same encoder saw fit patients, residual labels not OOF. Conditional patient/component bootstrap, overlapping trained fits, no external/runtime/interval or forecast claim.'), indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
