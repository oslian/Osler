"""Locate source/evaluation sign reversals without changing selection or gates."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def category(source_delta, evaluation_delta):
    if not np.isfinite([source_delta, evaluation_delta]).all():
        raise ValueError('Nonfinite change')
    return ('source_improves' if source_delta < 0 else 'source_not_improved') + '/' + (
        'evaluation_improves' if evaluation_delta < 0 else 'evaluation_not_improved')


def main():
    oldp=Path('artifacts/native_reconstruction_source_exposure_pair_20260918.json')
    newp=Path('artifacts/native_reconstruction_coverage_continuation_20260919.json')
    evalp=Path('artifacts/native_reconstruction_coverage_evaluation_20260920.json')
    out=Path('artifacts/native_reconstruction_continuation_transfer_20260920.json')
    if out.exists(): raise ValueError('Existing diagnostic')
    old,new,evaluation=[json.loads(p.read_text()) for p in (oldp,newp,evalp)]
    for r in (old,new,evaluation):
        for p,h in r['bindings'].items():
            if sha256(Path(p))!=h: raise ValueError(f'Binding changed: {p}')
    previous=old['modes']['expanded']['real'];current=new['arms']['real']
    a,b=previous['selected_epoch'],current['selected_epoch']
    before=current['log'][a-1]['group_target_scores']
    after=current['log'][b-1]['group_target_scores']
    if current['log'][:8]!=previous['log']: raise ValueError('Historical logs changed')
    result={}
    for group,targets in evaluation['groups'].items():
        per={}
        for t,r in targets.items():
            if 'mae' not in r:
                per[t]=dict(status='untrained');continue
            sd=after[group][t]-before[group][t]
            ed=r['mae']['continued_real']-r['mae']['expanded_real']
            per[t]=dict(source_before=before[group][t],source_after=after[group][t],
                        source_delta=sd,evaluation_delta=ed,category=category(sd,ed))
        result[group]=per
    out.write_text(json.dumps(dict(groups=result,old_selected_epoch=a,new_selected_epoch=b,
        bindings={str(p):sha256(p) for p in (oldp,newp,evalp,Path(__file__))},
        runtime_modified=False,limitation='Descriptive mean signs on existing source selection and reused evaluation. Not significant harm, causal attribution or per-target checkpoint selection.'),indent=2)+'\n')
    for g,ts in result.items():
        for c in sorted({r['category'] for r in ts.values() if 'category' in r}):
            print(g,c,[t for t,r in ts.items() if r.get('category')==c])


if __name__=='__main__':main()
