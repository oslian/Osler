"""Source-only measured auxiliary-label support at fixed glucose anchors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_event_aware_patient_state import patient_partition
from whole_body_jepa.availability_supervision import future_pairs
from whole_body_jepa.multisource_intervals import SEEDS, sha256


def main():
    directory = Path('/private/tmp/encoder_source_expansion_20260910')
    paths = [Path('/private/tmp/availability_glucose_development.parquet'),
             Path('/private/tmp/availability_jepa_eicu_labs.parquet'),
             directory/'queries.parquet', directory/'labs.parquet']
    preparation = json.loads(Path('artifacts/encoder_source_expansion_20260910.json').read_text())
    for path in paths[2:]:
        if sha256(path) != preparation['binding'][str(path)]:
            raise ValueError('source expansion changed')
    original = pd.read_parquet(paths[0])
    extra = pd.read_parquet(paths[2])
    eligible = set(extra.loc[extra.fresh & extra.truth.notna(), 'subject_id'])
    selected = set(sorted(eligible, key=lambda s: hashlib.sha256(
        ('learning-curve-v1:'+s).encode()).hexdigest())[:2000])
    if len(selected) != 2000 or selected & set(original.subject_id):
        raise ValueError('invalid source expansion')
    extra = extra.loc[extra.subject_id.isin(selected)]
    frame = pd.concat([original, extra], ignore_index=True)
    frame = frame.loc[frame.fresh & frame.truth.notna()].reset_index(drop=True)
    raw = pd.concat([pd.read_parquet(paths[1]), pd.read_parquet(paths[3])], ignore_index=True)
    raw = raw.loc[raw.stay_id.isin(frame.stay_id)]
    checkpoint = torch.load('models/whole_body_joint_jepa.pt', map_location='cpu', weights_only=False)
    variables = checkpoint['variables']
    arrays = future_pairs(frame, raw, variables)
    glucose = variables.index('glucose')
    np.testing.assert_allclose(arrays['current'][:, glucose], frame.current, rtol=0, atol=1e-8)
    np.testing.assert_allclose(arrays['future'][:, glucose], frame.truth, rtol=0, atol=1e-8)
    np.testing.assert_allclose(arrays['actual_delta_t'][:, glucose], frame.actual_delta_t, rtol=0, atol=1e-8)
    partitions = {}
    old = ~frame.subject_id.isin(selected).to_numpy()
    for seed in SEEDS:
        fit_old, _ = patient_partition(frame.loc[old, 'subject_id'], seed)
        fit = np.ones(len(frame), dtype=bool)
        fit[old] = fit_old
        partitions[int(seed)] = fit
        if frame.loc[fit, 'subject_id'].nunique() != 2250:
            raise ValueError('fit cohort differs from existing experiment')
    support = []
    for index, variable in enumerate(variables):
        pair = arrays['paired'][:, index]
        splits = []
        for seed, fit in partitions.items():
            valid = pair & fit
            ids = np.flatnonzero(valid)
            draws = pd.DataFrame({'stay': frame.loc[valid, 'stay_id'].to_numpy(),
                'time': frame.loc[valid, 't'].to_numpy() + pd.to_timedelta(
                    arrays['actual_delta_t'][ids, index], unit='h')})
            splits.append(dict(seed=seed, patients=int(frame.loc[valid, 'subject_id'].nunique()),
                               pairs=int(valid.sum()), distinct_draw_times=len(draws.drop_duplicates())))
        support.append(dict(variable=variable, splits=splits,
            eligible=variable != 'glucose' and all(s['patients'] >= 250 and s['pairs'] >= 1000 for s in splits)))
    destination = Path('/private/tmp/auxiliary_supervision_20260910.npz')
    if destination.exists():
        raise ValueError('output already exists')
    np.savez_compressed(destination, **arrays, variables=np.asarray(variables),
                        query_time=frame.t.to_numpy(), stay_id=frame.stay_id.to_numpy(dtype=str),
                        subject_id=frame.subject_id.to_numpy(dtype=str))
    report = dict(status='support_audit_only', runtime_modified=False, external_data_read=False,
        glucose_labels_reproduced=True, pairs=len(frame), patients=int(frame.subject_id.nunique()),
        eligible_auxiliary_variables=[s['variable'] for s in support if s['eligible']], support=support,
        binding={str(p):sha256(p) for p in paths + [Path(__file__),
            Path('whole_body_jepa/availability_supervision.py'),
            Path('AUXILIARY_SUPERVISION_PROTOCOL_20260910.md'), destination]})
    Path('artifacts/auxiliary_supervision_20260910.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('binding', 'support')}, indent=2))
    for s in support:
        if s['eligible']:
            print(s['variable'], min(x['patients'] for x in s['splits']), min(x['pairs'] for x in s['splits']))


if __name__ == '__main__':
    main()
