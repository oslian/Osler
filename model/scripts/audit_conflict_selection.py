"""Distinguish selected-epoch changes from matched-epoch repair regressions."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def main():
    parent=Path('artifacts/native_conflict_repair_training_20260911.json')
    output=Path('artifacts/native_conflict_selection_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    report=json.loads(parent.read_text())
    for path,digest in report['binding'].items():
        if sha256(Path(path))!=digest:raise ValueError('training evidence changed')
    result=dict(complete=True,runtime_modified=False,training_performed=False,seeds=[],
        binding={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))})
    for entry in report['seeds']:
        old,clean=[entry['arms'][key] for key in ('old_full','clean_full')]
        for arm in (old,clean):
            assert [x['epoch'] for x in arm['log']]==list(range(1,33))
            if arm['selected_epoch']:
                best=min(arm['log'],key=lambda x:x['tune_mae'])
                assert best['epoch']==arm['selected_epoch']
                assert best['tune_mae']==arm['tune_mae']
        differences=np.array([b['tune_mae']-a['tune_mae'] for a,b in zip(old['log'],clean['log'])])
        result['seeds'].append(dict(seed=entry['seed'],old_epoch=old['selected_epoch'],
            clean_epoch=clean['selected_epoch'],same_epoch=old['selected_epoch']==clean['selected_epoch'],
            first_logged_difference=int(np.flatnonzero(differences!=0)[0]+1) if np.any(differences!=0) else None,
            same_epoch_tune_differences=differences.tolist(),
            hospital=entry['roles']['hospital']['comparisons']['clean_full_vs_old_full']))
    same=[s for s in result['seeds'] if s['same_epoch']]
    result['same_epoch_summary']=dict(splits=len(same),significant_harm=sum(s['hospital']['ci95'][0]>0 for s in same),
                                     significant_gain=sum(s['hospital']['ci95'][1]<0 for s in same))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
