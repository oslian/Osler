"""Tune-only L1 versus Smooth-L1 offsets on identical frozen predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_magnesium_matched_20260911.json');ref=json.loads(parent.read_text())
    for name,digest in ref['binding'].items():assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_loss_calibration_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    lab=dict(np.load(root/'multitarget_labels.npz'));v=list(lab['variables']);k=v.index('magnesium')
    valid=np.load(root/'labels.npz')['paired'][:,0];eligible=lab['paired'][:,k]&lab['paired'][:,v.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=lab['future'][:,k]
    scale=float(load_release(Path('models')).scaler.scales[k]);items=[]
    for entry in ref['seeds']:
        seed=entry['seed'];roles={a:b[valid]&eligible for a,b in dict(np.load(root/f'partition_{seed}.npz')).items()}
        tune=roles['tune'];p=dict(np.load(root/f'magnesium_matched_predictions_{seed}.npz'))
        subjects=q.loc[tune,'subject_id'];w=1/subjects.groupby(subjects).transform('size').to_numpy(float);w/=w.sum()
        item=dict(seed=seed,offsets={},roles={})
        for arm in ('own','real'):
            residual=q.loc[tune,'truth'].to_numpy()-p[arm][tune]
            order=np.argsort(residual);median=float(residual[order][np.searchsorted(np.cumsum(w[order]),.5)])
            huber=float(brentq(lambda b:np.dot(w,np.clip((b-residual)/scale,-1,1)),residual.min()-scale,residual.max()+scale))
            item['offsets'][arm]=dict(l1=median,smooth_l1=huber)
            p[arm+'_l1']=p[arm]+median;p[arm+'_huber']=p[arm]+huber
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            comparisons=[('real_l1','real'),('own_l1','own'),('real_l1','real_huber'),('real_l1','own_l1')]
            item['roles'][role]=dict(mae={a:patient_equal_mae(f,f.truth,b[m]) for a,b in p.items()},
                comparisons={a+'_vs_'+b:paired(f,p[a][m],p[b][m],f'mg-loss-{seed}-{role}-{a}-{b}',2000) for a,b in comparisons})
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,neural_training=False,calibration_fit_role='source_tune',
        smooth_l1_beta_native_units=scale,seeds=items,binding={str(p):sha256(p) for p in [parent,Path(__file__)]}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:float(np.median([s['roles'][role]['mae'][a] for s in items])) for a in items[0]['roles'][role]['mae']})
        print({a:(sum(s['roles'][role]['comparisons'][a]['ci95'][1]<0 for s in items),sum(s['roles'][role]['comparisons'][a]['ci95'][0]>0 for s in items)) for a in items[0]['roles'][role]['comparisons']})


if __name__=='__main__':main()
