"""Locate repair effects on affected versus unaffected held-out histories."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/native_conflict_repair_training_20260911.json')
    output = Path('artifacts/native_conflict_subgroups_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    reference = json.loads(parent.read_text())
    for path, digest in reference['binding'].items():
        if sha256(Path(path)) != digest: raise ValueError('training dependency changed')
    old = dict(np.load(root/'native_schedule.npz'))
    new = dict(np.load(root/'native_schedule_conflict_replay.npz'))
    changed = []
    for i in range(len(old['query_ptr'])-1):
        lists = []
        for schedule in (old, new):
            a,b = schedule['query_ptr'][i:i+2]
            lists.append(schedule['event_indices'][schedule['group_ptr'][a]:schedule['group_ptr'][b]])
        changed.append(not np.array_equal(*lists))
    changed = np.asarray(changed)
    oldbase = dict(np.load(root/'body_arrays.npz'))
    newbase = dict(np.load(root/'body_arrays_conflict_checked.npz'))
    rows = oldbase['query_rows']; np.testing.assert_array_equal(rows,newbase['query_rows'])
    changed |= np.any(oldbase['input'][rows] != newbase['input'][rows],axis=1)
    valid = np.load(root/'labels.npz')['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    labels = dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'))
    col = labels['variables'].tolist().index('glucose'); eligible = labels['paired'][:,col]
    q['truth'] = labels['future'][:,col]
    result = dict(complete=True,runtime_modified=False,training_performed=False,
        scope='descriptive exposure stratification; not a new selection or promotion gate',seeds=[],
        changed_queries=int(changed.sum()),binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for seed in (7,19,31,43,59,71,83):
        points = dict(np.load(root/f'conflict_repair_predictions_{seed}.npz'))
        roles = {a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            rolemask=roles[role]
            affected_subjects=set(q.loc[rolemask&changed,'subject_id'])
            groups={'changed_rows':rolemask&changed,'unchanged_rows':rolemask&~changed,
                    'unaffected_patients':rolemask&~q.subject_id.isin(affected_subjects).to_numpy()}
            item['roles'][role]={}
            for name,m in groups.items():
                f=q.loc[m]
                item['roles'][role][name]=dict(rows=int(m.sum()),patients=int(f.subject_id.nunique()),
                    mae={a:patient_equal_mae(f,f.truth,p[m]) for a,p in points.items()},
                    comparisons={f'{a}_vs_{b}':paired(f,points[a][m],points[b][m],f'{seed}-{role}-{name}-{a}-{b}',1000)
                        for a,b in [('clean_full','old_full'),('clean_own','old_own'),('clean_full','clean_own')]})
        result['seeds'].append(item)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in ('changed_rows','unchanged_rows','unaffected_patients'):
            print(role,group, 'patients', [s['roles'][role][group]['patients'] for s in result['seeds']])
            for contrast in ('clean_full_vs_old_full','clean_own_vs_old_own','clean_full_vs_clean_own'):
                cs=[s['roles'][role][group]['comparisons'][contrast] for s in result['seeds']]
                print(contrast,'gain',sum(c.get('supported',False) and c['ci95'][1]<0 for c in cs),
                      'harm',sum(c.get('supported',False) and c['ci95'][0]>0 for c in cs),
                      'median_effect',float(np.median([c['mean'] for c in cs])))


if __name__=='__main__':main()
