"""Locate incremental glucose failure using fixed, hash-bound predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    reference = Path('artifacts/rate_glucose_jepa_20260911.json')
    output = Path('artifacts/glucose_native_fit_gap_20260911.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(reference.read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'changed reference input: {name}')
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    labels = np.load(root / 'multitarget_labels.npz')
    target = list(labels['variables']).index('glucose')
    frame = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    frame = frame.rename(columns={'uniquepid': 'subject_id'})
    frame['truth'] = labels['future'][:, target]
    results = []
    binding = {str(reference): sha256(reference), str(Path(__file__)): sha256(Path(__file__))}
    for entry in ref['seeds']:
        seed = entry['seed']
        paths = [root / f'glucose_native_predictions_{seed}.npz', root / f'partition_{seed}.npz']
        binding.update({str(p): sha256(p) for p in paths})
        points, partitions = (np.load(p) for p in paths)
        roles = {role: partitions[role][valid] & labels['paired'][:, target]
                 for role in ('fit', 'tune', 'evaluation', 'hospital')}
        patients = {role: set(frame.loc[mask, 'subject_id']) for role, mask in roles.items()}
        for left in patients:
            for right in patients:
                if left != right and patients[left] & patients[right]:
                    raise ValueError('patient overlap')
        result = {'seed': seed, 'roles': {}}
        for role, mask in roles.items():
            selected = frame.loc[mask]
            mae = {arm: patient_equal_mae(selected, selected.truth, points[arm][mask])
                   for arm in ('own', 'real', 'support')}
            if role in ('evaluation', 'hospital'):
                for arm in mae:
                    np.testing.assert_allclose(mae[arm], entry['roles'][role]['mae'][arm], atol=1e-7, rtol=0)
            if role == 'tune':
                for arm in mae:
                    np.testing.assert_allclose(mae[arm], entry['arms'][arm]['tune_mae'], atol=1e-7, rtol=0)
            comparisons = {f'real_vs_{arm}': paired(selected, points['real'][mask], points[arm][mask],
                           f'glucose-native-fit-gap:{seed}:{role}:{arm}', 2000)
                           for arm in ('own', 'support')}
            result['roles'][role] = {'mae': mae, 'comparisons': comparisons}
        results.append(result)
    report = {'complete': True, 'fitting_performed': False, 'runtime_modified': False,
              'status': 'diagnostic_only', 'pretraining_overlap_not_excluded': True,
              'binding': binding, 'seeds': results}
    output.write_text(json.dumps(report, indent=2) + '\n')
    for role in roles:
        group = [r['roles'][role] for r in results]
        summary = {arm: float(np.median([g['mae'][arm] for g in group])) for arm in mae}
        for arm in ('own', 'support'):
            ci = [g['comparisons'][f'real_vs_{arm}']['ci95'] for g in group]
            summary[f'vs_{arm}_gain_harm'] = [sum(c[1] < 0 for c in ci), sum(c[0] > 0 for c in ci)]
        print(role, summary)


if __name__ == '__main__':
    main()
