"""Source-selection-only trajectory diagnosis, not a new checkpoint selector."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def main():
    src=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    out=Path('artifacts/native_reconstruction_joint_selection_tradeoff_20260918.json')
    if out.exists():raise ValueError('Existing diagnostic')
    d=json.loads(src.read_text());a=d['arms']['real'];initial=a['initial_scores']
    for p,h in d['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training bindings changed')
    logs=[dict(epoch=0,group_target_scores=initial)]+a['log']
    result={};epochs=[]
    for g,targets in initial.items():
        result[g]={}
        for t,base in targets.items():
            values=[r['group_target_scores'][g][t]for r in logs]
            chosen=values[a['selected_epoch']]
            result[g][t]=dict(initial_mae=base,selected_mae=chosen,
                selected_minus_initial=chosen-base,epoch_scores=values,
                descriptive_minimum_epoch=int(np.argmin(values)),
                selected_minus_descriptive_minimum=chosen-min(values))
    for r in logs:
        per={}
        for g,ts in initial.items():
            values=r['group_target_scores'][g]
            per[g]=dict(macro_mae=float(np.mean(list(values.values()))),
                targets_worse_than_initial=[t for t in ts if values[t]>ts[t]],
                improvements_sum=float(sum(max(ts[t]-values[t],0)for t in ts)),
                harms_sum=float(sum(max(values[t]-ts[t],0)for t in ts)))
        score=float(np.mean([x['macro_mae']for x in per.values()]))
        if r['epoch'] and not np.isclose(score,r['tune_mae'],atol=1e-12):raise ValueError('Selection replay mismatch')
        epochs.append(dict(epoch=r['epoch'],macro_mae=score,groups=per))
    if int(np.argmin([r['macro_mae']for r in epochs]))!=a['selected_epoch']:raise ValueError('Selected epoch mismatch')
    out.write_text(json.dumps(dict(selected_epoch=a['selected_epoch'],epochs=epochs,targets=result,
        bindings={str(p):sha256(p)for p in (Path(__file__),src)},runtime_modified=False,promotion_allowed=False,
        limitation='Descriptive on data already used for epoch selection. Per-target minima are optimistic and cannot replace selected weights.'),indent=2)+'\n')
    print(json.dumps(epochs[a['selected_epoch']],indent=2))
    print('Epoch group scores',[(r['epoch'],{g:v['macro_mae']for g,v in r['groups'].items()})for r in epochs])


if __name__=='__main__':main()
