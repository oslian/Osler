"""Test whether checkpoint selection alone could satisfy numerical retention."""
import json
from pathlib import Path
from whole_body_jepa.multisource_intervals import sha256


def classify(values,selected):
    if len(values)<2 or not 0<=selected<len(values):raise ValueError('Invalid trajectory')
    initial=values[0]
    if min(values[1:])>=initial:return 'never_improved_in_observed_epochs'
    if values[selected]>initial:return 'improved_then_not_retained'
    return 'selected_not_worse_than_initial'


def main():
    src=Path('artifacts/native_reconstruction_joint_selection_tradeoff_20260918.json')
    train=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    out=Path('artifacts/native_reconstruction_joint_retention_feasibility_20260918.json')
    if out.exists():raise ValueError('Existing diagnostic')
    d=json.loads(src.read_text());training=json.loads(train.read_text())
    for p,h in d['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed evidence')
    groups={}
    for group,targets in d['targets'].items():
        rows={}
        for target,r in targets.items():
            support=training['support'][group]['encoder_tune'][target]
            rows[target]=dict(category=classify(r['epoch_scores'],d['selected_epoch']),
                source_selection_patients=support,low_support=support<30,
                best_trained_minus_initial=min(r['epoch_scores'][1:])-r['initial_mae'])
        groups[group]=rows
    feasible=[r['epoch']for r in d['epochs']if all(not v['targets_worse_than_initial']for v in r['groups'].values())]
    out.write_text(json.dumps(dict(groups=groups,numerically_nonharming_common_epochs=feasible,
        bindings={str(p):sha256(p)for p in (Path(__file__),src,train)},runtime_modified=False,
        limitation='Necessary empirical condition only; numerical zero harm is not a statistical noninferiority gate. Eight observed epochs cannot prove impossibility of learning.'),indent=2)+'\n')
    print('Feasible observed epochs',feasible)
    for group,rows in groups.items():
        for category in sorted({r['category']for r in rows.values()}):
            print(group,category,[t for t,r in rows.items()if r['category']==category])


if __name__=='__main__':main()
