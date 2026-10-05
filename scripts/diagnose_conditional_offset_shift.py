"""Prediction-state composition vs within-state residual shift, diagnostic only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def boundaries(source):
    return np.unique(np.quantile(source.real.to_numpy(),[1/3,2/3])) if len(source) else np.array([])


def profile(frame, cuts):
    if frame.empty:return None
    sizes=frame.groupby('component').size()
    weight=(1/frame.component.map(sizes)).to_numpy();weight=weight/weight.sum()
    bins=np.searchsorted(cuts,frame.real.to_numpy(),side='right')
    residual=(frame.truth-frame.real).to_numpy()
    result=[]
    for b in range(len(cuts)+1):
        select=bins==b;mass=float(weight[select].sum());part=frame.loc[select]
        result.append(dict(bin=b,patients=len(part),components=int(part.component.nunique()),
            supported_components=int((part.groupby('component').size()>=30).sum()),mass=mass,
            mean_residual=float(np.average(residual[select],weights=weight[select])) if mass else None))
    return dict(mean_residual=float(np.average(residual,weights=weight)),bins=result)


def diagnose(source, outer):
    cuts=boundaries(source);a=profile(source,cuts);b=profile(outer,cuts)
    result=dict(cuts=cuts.tolist(),source=a,outer=b,decomposition=None)
    if a is None or b is None or any(x['mass']==0 or y['mass']==0 for x,y in zip(a['bins'],b['bins'])):return result
    composition=sum((y['mass']-x['mass'])*(x['mean_residual']+y['mean_residual'])/2 for x,y in zip(a['bins'],b['bins']))
    within=sum((y['mean_residual']-x['mean_residual'])*(x['mass']+y['mass'])/2 for x,y in zip(a['bins'],b['bins']))
    shift=b['mean_residual']-a['mean_residual']
    if not np.isclose(composition+within,shift,rtol=1e-8,atol=1e-8):raise ValueError('Decomposition mismatch')
    supported=all(x['supported_components']>=3 and y['supported_components']>=3 for x,y in zip(a['bins'],b['bins']))
    result['decomposition']=dict(mean_residual_shift=shift,composition=composition,within_prediction_state=within,
        each_bin_has_three_supported_components=bool(supported))
    return result


def main():
    parent=Path('artifacts/native_reconstruction_additional_calibration_20260925.json')
    r=json.loads(parent.read_text())
    for p,h in r['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    root=Path('data/research_recovery/additional_development_domains_20260921');results={}
    for fold in range(5):
        source=pd.read_parquet(root/f'calibration_20260924/fold{fold}.parquet')
        outer=pd.read_parquet(root/f'outer_20260924/fold{fold}.parquet')
        results[str(fold)]={t:diagnose(source.loc[source.target==t],outer.loc[outer.target==t]) for t in r['targets']}
    full=[x['decomposition'] for f in results.values() for x in f.values() if x['decomposition'] is not None]
    supported=[x for x in full if x['each_bin_has_three_supported_components']]
    summary=dict(target_folds=235,decomposable=len(full),supported=len(supported),
        supported_within_absolute_larger=sum(abs(x['within_prediction_state'])>abs(x['composition']) for x in supported))
    out=Path('artifacts/native_reconstruction_conditional_offset_shift_20260925.json')
    if out.exists():raise ValueError('Existing diagnosis')
    out.write_text(json.dumps(dict(summary=summary,folds=results,bindings={str(parent):sha256(parent),str(Path(__file__)):sha256(Path(__file__))},
        runtime_modified=False,promotion_allowed=False,
        limitation='Posthoc mean-residual decomposition, not median/MAE decomposition or causal mechanism. Bins use source predictions only, not true state. No correction fitted or enabled.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
