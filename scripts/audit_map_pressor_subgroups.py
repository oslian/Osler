"""Recorded pre-anchor support strata, using fixed predictions only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/map_recorded_pressor_training_20260912.json')
    source = json.loads(parent.read_text())
    for path, digest in source['bindings'].items():
        assert sha256(Path(path)) == digest
    q = pd.read_parquet(root / 'queries.parquet').rename(columns={'uniquepid': 'subject_id'})
    labels = np.load(root / 'map_all_query_labels.npz'); q['truth'] = labels['future'][:, 0]
    context = np.load(root / 'map_recorded_pressors.npz')
    supported = context['known'].any(axis=(1, 2))
    results = []
    for original in source['seeds']:
        seed = original['seed']; roles = np.load(root / f'partition_{seed}.npz')
        predictions = dict(np.load(root / f'map_recorded_pressor_predictions_{seed}.npz'))
        item = dict(seed=seed, roles={})
        for role in ('evaluation', 'hospital'):
            item['roles'][role] = {}
            for name, group in (('recorded_rate', supported), ('no_compatible_rate', ~supported)):
                mask = roles[role] & labels['paired'][:, 0] & group
                f = q.loc[mask]; p = {k: a[mask] for k, a in predictions.items()}
                item['roles'][role][name] = dict(rows=int(mask.sum()), patients=int(f.subject_id.nunique()),
                    mae={k: patient_equal_mae(f, f.truth, a) for k, a in p.items()},
                    rate_comparisons={r: paired_model_errors(f, p, r, seed=20260912, repetitions=20000)['rate']
                                      for r in ('baseline', 'support')})
        results.append(item)
    paths = [Path(__file__), parent, root/'map_recorded_pressors.npz', root/'queries.parquet',
             root/'map_all_query_labels.npz', Path('whole_body_jepa/change_evaluation.py')]
    Path('artifacts/map_pressor_subgroups_20260912.json').write_text(json.dumps(dict(
        results=results, training_performed=False, runtime_modified=False, online_eligible=False,
        limitations=['Strata use recorded timing, not independent availability.',
                     'No compatible rate does not mean no treatment.',
                     'Patients can contribute to both strata.',
                     'Descriptive subgroup audit; no trained routing or causal claim.'],
        bindings={str(p): sha256(p) for p in paths}), indent=2) + '\n')
    for role in ('evaluation', 'hospital'):
        for group in ('recorded_rate', 'no_compatible_rate'):
            ss = [x['roles'][role][group] for x in results]
            print(role, group, 'patients', [s['patients'] for s in ss],
                'mae', {k:float(np.median([s['mae'][k] for s in ss])) for k in predictions},
                {r:(sum(s['rate_comparisons'][r]['ci95'][1]<0 for s in ss),
                    sum(s['rate_comparisons'][r]['ci95'][0]>0 for s in ss)) for r in ('baseline','support')})


if __name__ == '__main__': main()
