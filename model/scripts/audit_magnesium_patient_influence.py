"""Patient influence diagnostic; no patient exclusion or model fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    parent=Path('artifacts/native_magnesium_matched_20260911.json')
    selection_path=Path('artifacts/magnesium_increment_shrink_20260911.json')
    ref=json.loads(parent.read_text());selection=json.loads(selection_path.read_text())
    for report in (ref,selection):
        for name,digest in report['binding'].items():
            assert sha256(Path(name))==digest,name
    out=Path('artifacts/magnesium_patient_influence_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    labels=dict(np.load(root/'multitarget_labels.npz'));variables=list(labels['variables'])
    col=variables.index('magnesium');valid=np.load(root/'labels.npz')['paired'][:,0]
    eligible=labels['paired'][:,col]&labels['paired'][:,variables.index('glucose')]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    items=[]
    for entry in selection['seeds']:
        seed=entry['seed'];p=dict(np.load(root/f'magnesium_matched_predictions_{seed}.npz'))
        p['selected']=p['own']+entry['alpha']*(p['real']-p['own'])
        roles={k:v[valid]&eligible for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        item=dict(seed=seed,roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];truth=labels['future'][m,col];item['roles'][role]={}
            for arm in ('real','selected'):
                delta=np.abs(p[arm][m]-truth)-np.abs(p['own'][m]-truth)
                by=pd.Series(delta,index=q.loc[m,'uniquepid'].to_numpy()).groupby(level=0).mean().to_numpy()
                loo=(by.sum()-by)/(len(by)-1)
                item['roles'][role][arm]=dict(patients=len(by),mean=float(by.mean()),
                    benefit_fraction=float(np.mean(by<0)),quantiles=np.quantile(by,[0,.1,.25,.5,.75,.9,1]).tolist(),
                    leave_one_out_range=[float(loo.min()),float(loo.max())],
                    mean_sign_changes=bool(loo.min()<0 and loo.max()>0))
        items.append(item)
    out.write_text(json.dumps(dict(complete=True,runtime_modified=False,fitting_performed=False,seeds=items,
        binding={str(p):sha256(p) for p in (parent,selection_path,Path(__file__))}),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        print(role,{a:dict(benefit_fraction=[round(s['roles'][role][a]['benefit_fraction'],3) for s in items],
            sign_changes=sum(s['roles'][role][a]['mean_sign_changes'] for s in items)) for a in ('real','selected')})


if __name__=='__main__':main()
