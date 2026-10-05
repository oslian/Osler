"""Locate raw-history increment loss across fitting and new MAP patients."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_anchor_tree_20260912.json')
    out = Path('artifacts/map_native_tree_fit_gap_20260912.json')
    if out.exists():
        raise ValueError('fresh output required')
    original = json.loads(parent.read_text())
    for path, digest in original['bindings'].items():
        assert sha256(Path(path)) == digest, path
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:, 0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][valid, 0]
    results = []
    for entry in original['seeds']:
        seed = entry['seed']
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        points = dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        scores = {}
        for role, mask in roles.items():
            frame = q.loc[mask]
            predictions = {k:v[mask] for k,v in points.items()}
            scores[role] = dict(rows=int(mask.sum()), patients=int(frame.subject_id.nunique()),
                mae={k:patient_equal_mae(frame, frame.truth, v) for k,v in predictions.items()},
                versus_own=paired_model_errors(frame, predictions, 'own', seed=20260912),
                versus_support=paired_model_errors(frame, predictions, 'support', seed=20260912))
            if role in entry['roles']:
                assert scores[role]['mae'] == entry['roles'][role]['mae']
                assert scores[role]['versus_own'] == entry['roles'][role]['versus_own']
                assert scores[role]['versus_support'] == entry['roles'][role]['versus_support']
        results.append(dict(seed=seed, roles=scores))
    report = dict(complete=True, training_performed=False, runtime_modified=False,
                  original_scores_exact=True, seeds=results,
                  caveat='Fit scores are resubstitution, not independent efficacy; tune unused by fixed configuration.',
                  bindings={str(p):sha256(p) for p in (parent, Path(__file__),
                    Path('whole_body_jepa/change_evaluation.py'),
                    Path('scripts/audit_recurrent_event_state_loop.py'))})
    out.write_text(json.dumps(report, indent=2)+'\n')
    for role in roles:
        print(role, {k:float(np.median([r['roles'][role]['mae'][k] for r in results])) for k in points})
        for ref in ('own','support'):
            c = [r['roles'][role]['versus_'+ref]['full'] for r in results]
            print('full vs', ref, 'gain/harm', sum(x['ci95'][1]<0 for x in c), sum(x['ci95'][0]>0 for x in c))


if __name__ == '__main__':
    main()
