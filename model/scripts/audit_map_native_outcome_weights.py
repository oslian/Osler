"""Outcome-equal sensitivity and observation-support decomposition, no fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.change_evaluation import estimate
from whole_body_jepa.multisource_intervals import sha256


def summarize(frame, predictions, event_equal):
    errors = frame[['subject_id', 'stay_id', 'future_id', 'truth']].copy()
    for name, values in predictions.items():
        errors[name] = np.abs(np.asarray(values)-errors.truth.to_numpy())
    grouping = errors.groupby(['subject_id','stay_id','future_id'])
    if (grouping.truth.nunique() > 1).any():
        raise ValueError('same future event has inconsistent truth')
    events = grouping[list(predictions)].mean().reset_index()
    scored = events if event_equal else errors
    patient = scored.groupby('subject_id')[list(predictions)].mean().reset_index()
    comparisons = {}
    for a,b in (('full','own'),('support','own'),('full','support'),('full','persistence')):
        comparisons[a+'_vs_'+b] = estimate(patient, (patient[a]-patient[b]).to_numpy(),
                                          seed=20260912, repetitions=20000)
    return dict(rows=len(frame), events=len(events), patients=len(patient),
                mae=patient[list(predictions)].mean().to_dict(), comparisons=comparisons)


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_anchor_tree_20260912.json')
    output = Path('artifacts/map_native_outcome_weights_20260912.json')
    if output.exists():
        raise ValueError('fresh outputs required')
    ref = json.loads(parent.read_text())
    for p,h in ref['bindings'].items():
        assert sha256(Path(p)) == h, p
    labels = dict(np.load(root/'labels.npz'))
    valid = labels['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q = q.rename(columns={'uniquepid':'subject_id'})
    q['truth'] = labels['future'][valid,0]
    q['future_id'] = labels['future_source_id'][valid,0]
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']
        roles = {k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        points = dict(np.load(root/f'tree_trial/predictions_{seed}.npz'))
        scores = {}
        for role, mask in roles.items():
            frame = q.loc[mask]
            pred = {k:v[mask] for k,v in points.items()}
            scores[role] = {name:summarize(frame,pred,equal) for name,equal in [('row',False),('event',True)]}
            if role in entry['roles']:
                for name,mae in entry['roles'][role]['mae'].items():
                    np.testing.assert_allclose(scores[role]['row']['mae'][name],mae,rtol=1e-12)
        results.append(dict(seed=seed,roles=scores))
    output.write_text(json.dumps(dict(complete=True, runtime_modified=False, training_performed=False,
        promotion_gate_replaced=False, seeds=results,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('whole_body_jepa/change_evaluation.py'))}),indent=2)+'\n')
    for role in roles:
        for mode in ('row','event'):
            rs=[r['roles'][role][mode] for r in results]
            print(role,mode,'mae',{k:float(np.median([r['mae'][k] for r in rs])) for k in points})
            print({key:(sum(r['comparisons'][key]['ci95'][1]<0 for r in rs),
                        sum(r['comparisons'][key]['ci95'][0]>0 for r in rs)) for key in rs[0]['comparisons']})


if __name__ == '__main__':
    main()
