"""Replay magnesium source sampling at selected and final epochs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_encoder_exposure import replay
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    path=Path('artifacts/native_magnesium_matched_20260911.json')
    ref=json.loads(path.read_text())
    for name,digest in ref['binding'].items():
        assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_exposure_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    own=json.loads(Path('artifacts/native_magnesium_own_20260911.json').read_text())
    labels=dict(np.load(root/'multitarget_labels.npz'))
    variables=list(labels['variables']);col=variables.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0]
    eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    q['future_id']=labels['future_source_id'][:,col]
    items=[]
    for entry,old in zip(ref['seeds'],own['seeds']):
        seed=entry['seed'];assert seed==old['seed']
        fit=np.load(root/f'partition_{seed}.npz')['fit'][valid]&eligible
        pools=[np.flatnonzero(fit&q.uniquepid.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'uniquepid']))]
        item=dict(seed=seed,fit_patients=len(pools),fit_rows=int(fit.sum()),
                  rows_per_patient=np.quantile([len(p) for p in pools],[0,.25,.5,.75,1]).tolist(),stages={})
        for stage,epoch in [('own_selected',old['arms']['own']['selected_epoch']),('whole_selected',entry['arms']['real']['selected_epoch']),('final',32)]:
            hits=replay(pools,seed,epochs=epoch)
            hits=np.pad(hits,(0,len(q)-len(hits)));seen=hits>0
            assert hits.sum()==epoch*16*32 and not (seen&~fit).any()
            draws=[int(hits[p].sum()) for p in pools]
            key=['stay_id','future_id']
            item['stages'][stage]=dict(epoch=epoch,row_coverage=float(seen.sum()/fit.sum()),
                patient_coverage=float(np.mean(np.array(draws)>0)),draws_per_patient=np.quantile(draws,[0,.25,.5,.75,1]).tolist(),
                future_event_coverage=len(q.loc[seen,key].drop_duplicates())/len(q.loc[fit,key].drop_duplicates()))
        items.append(item)
    report=dict(complete=True,runtime_modified=False,fitting_performed=False,seeds=items,
                binding={str(p):sha256(p) for p in (path,Path(__file__),Path('scripts/audit_encoder_exposure.py'))})
    out.write_text(json.dumps(report,indent=2)+'\n')
    for stage in items[0]['stages']:
        print(stage,{k:[min(s['stages'][stage][k] for s in items),max(s['stages'][stage][k] for s in items)] for k in ('row_coverage','patient_coverage','future_event_coverage')})
    print('patients',[s['fit_patients'] for s in items])


if __name__=='__main__':main()
