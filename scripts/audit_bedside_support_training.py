"""Compare real bedside numbers with a same-cadence trained support control."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_bedside_support_comparison_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    report=json.loads(Path('artifacts/native_bedside_support_training_20260911.json').read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('current run dependency changed')
    previous=json.loads(Path('artifacts/native_bedside_rescale_training_20260911.json').read_text())
    valid=np.load(root/'labels.npz')['paired'][:,0]
    labels=dict(np.load(root/'multitarget_labels_joint_conflict_checked.npz'));col=labels['variables'].tolist().index('glucose')
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=labels['future'][:,col]
    result=dict(complete=True,training_performed=False,runtime_modified=False,seeds=[])
    for item in report['seeds']:
        seed=item['seed'];p=root/f'bedside_rescale_predictions_{seed}.npz'
        if sha256(p)!=previous['bindings'][str(p)]:raise ValueError('historical predictions changed')
        real=np.load(p);support=np.load(root/f'bedside_support_predictions_{seed}.npz')
        np.testing.assert_array_equal(real['lab'],support['lab'])
        roles={a:b[valid]&labels['paired'][:,col] for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        result['seeds'].append(dict(seed=seed,lab_control_exact=True,roles={role:paired(q.loc[m],
            real['bedside'][m],support['bedside'][m],f'bedside-real-support-{seed}-{role}',1000)
            for role,m in roles.items() if role in ('evaluation','hospital')}))
    output.write_text(json.dumps(result,indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,'support MAE',float(np.median([s['roles'][role]['mae']['bedside'] for s in report['seeds']])))
        for name,cs in [('support_lab',[s['roles'][role]['comparisons']['bedside_vs_lab'] for s in report['seeds']]),
                        ('real_support',[s['roles'][role] for s in result['seeds']])]:
            print(name,'gain',sum(c['ci95'][1]<0 for c in cs),'harm',sum(c['ci95'][0]>0 for c in cs),
                'median_effect',float(np.median([c['mean'] for c in cs])))


if __name__=='__main__':main()
