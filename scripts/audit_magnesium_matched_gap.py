"""Locate the matched magnesium input gap without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    path = Path('artifacts/native_magnesium_matched_20260911.json')
    ref = json.loads(path.read_text())
    assert ref['complete']
    for name, digest in ref['binding'].items():
        assert sha256(Path(name)) == digest, name
    out = Path('artifacts/magnesium_matched_gap_20260911.json')
    if out.exists():
        raise ValueError('fresh output required')
    labels = dict(np.load(root / 'multitarget_labels.npz'))
    variables = list(labels['variables'])
    col = variables.index('magnesium')
    eligible = labels['paired'][:, col] & labels['paired'][:, variables.index('glucose')]
    valid = np.load(root / 'labels.npz')['paired'][:, 0]
    q = pd.read_parquet(root / 'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid': 'subject_id'})
    q['truth'] = labels['future'][:, col]
    items = []
    for entry in ref['seeds']:
        seed = entry['seed']
        points = dict(np.load(root / f'magnesium_matched_predictions_{seed}.npz'))
        roles = {k: v[valid] & eligible for k, v in dict(np.load(root / f'partition_{seed}.npz')).items()}
        ids = {r: set(q.loc[m, 'subject_id']) for r, m in roles.items()}
        names = list(ids)
        for i, name in enumerate(names):
            for other in names[i+1:]:
                assert not ids[name] & ids[other], (name, other)
        item = dict(seed=seed, roles={})
        for role in ('fit', 'tune', 'evaluation', 'hospital'):
            m = roles[role]
            frame = q.loc[m]
            mae = {a: patient_equal_mae(frame, frame.truth, p[m]) for a, p in points.items()}
            if role in entry['roles']:
                for a in mae:
                    np.testing.assert_allclose(mae[a], entry['roles'][role]['mae'][a], rtol=0, atol=1e-12)
            item['roles'][role] = dict(patients=len(ids[role]), mae=mae,
                comparison=paired(frame, points['real'][m], points['own'][m], f'mg-gap-{seed}-{role}', 1000))
        np.testing.assert_allclose(item['roles']['tune']['mae']['real'], entry['arms']['real']['tune_mae'], rtol=0, atol=1e-12)
        items.append(item)
    report = dict(complete=True, fitting_performed=False, runtime_modified=False, seeds=items,
                  binding={str(p): sha256(p) for p in (path, Path(__file__))})
    out.write_text(json.dumps(report, indent=2) + '\n')
    for role in ('fit', 'tune', 'evaluation', 'hospital'):
        rows = [s['roles'][role] for s in items]
        print(role, {a: float(np.median([r['mae'][a] for r in rows])) for a in points},
              'gain', sum(r['comparison']['ci95'][1] < 0 for r in rows),
              'harm', sum(r['comparison']['ci95'][0] > 0 for r in rows))


if __name__ == '__main__':
    main()
