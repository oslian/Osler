"""Select readouts using source tune only, never evaluated target winners."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.multisource_intervals import sha256


def select_source(scores):
    if not scores or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('finite nonempty source scores required')
    return min(scores, key=scores.get)


def main():
    output=Path('artifacts/native_reconstruction_readout_selection_20260917.json')
    decisions=Path('artifacts/native_reconstruction_readout_choices_20260917.json')
    if output.exists() or decisions.exists():raise ValueError('existing selection')
    rp=Path('artifacts/native_reconstruction_raw_state_20260917.json')
    report=json.loads(rp.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed bound input')
    cp=Path('data/research_recovery/site_raw_state_features_20260917.pt')
    data=torch.load(cp,weights_only=False)
    catalog=report['targets'];models={};choices={}
    for target in catalog:
        groups={r:[d for d in data if d['target']==target and d['role']==r] for r in ('fit','tune','evaluation')}
        sets=[{d['subject_id'] for d in g} for g in groups.values()]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('overlap')
        fit,tune=groups['fit'],groups['tune']
        x=np.stack([d['body'] for d in fit]);y=np.array([d['truth'] for d in fit])
        tx=np.stack([d['body'] for d in tune]);ty=np.array([d['truth'] for d in tune])
        median=float(np.median(y))
        candidates={'original':None,'population':median}
        scores={'original':float(np.mean(abs(np.array([d['original'] for d in tune])-ty))),
                'population':float(np.mean(abs(median-ty)))}
        for alpha in (1.,10.,100.,1000.,10000.):
            name=f'ridge_{alpha:g}'
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd')).fit(x,y)
            scores[name]=float(np.mean(abs(model.predict(tx)-ty)));candidates[name]=model
        choices[target]=dict(selected=select_source(scores),source_tune_scores=scores)
        models[target]=(candidates,groups['evaluation'])
    dependencies=[Path(__file__),rp,cp,Path('RECONSTRUCTION_READOUT_SELECTION_PROTOCOL_20260917.md')]
    bindings={str(p):sha256(p) for p in dependencies}
    decisions.write_text(json.dumps(dict(choices=choices,bindings=bindings,promotion_allowed=False),indent=2)+'\n')
    results={}
    for target,(candidates,rows) in models.items():
        frame=pd.DataFrame(dict(subject_id=[d['subject_id'] for d in rows],truth=[d['truth'] for d in rows]))
        original=np.array([d['original'] for d in rows]);x=np.stack([d['body'] for d in rows])
        def predict(name):
            if name=='original':return original.copy()
            if name=='population':return np.full(len(rows),candidates[name])
            return candidates[name].predict(x)
        prior=report['targets'][target]['choices']['body']['alpha']
        probe_name='population' if prior is None else f'ridge_{prior:g}'
        pred={'selected':predict(choices[target]['selected']),'original':original,
              'population':predict('population'),'all_probe':predict(probe_name)}
        mae={a:float(np.mean(abs(p-frame.truth.to_numpy()))) for a,p in pred.items()}
        if not np.isclose(mae['all_probe'],report['targets'][target]['mae']['body'],atol=1e-9,rtol=0):raise ValueError('probe reproduction failed')
        if not np.isclose(mae['original'],report['targets'][target]['mae']['original'],atol=1e-9,rtol=0):raise ValueError('original reproduction failed')
        results[target]=dict(patients=len(rows),choice=choices[target]['selected'],mae=mae,
            comparisons={ref:paired_model_errors(frame,pred,ref,seed=20260917,repetitions=2000) for ref in ('original','population','all_probe')})
    bindings[str(decisions)]=sha256(decisions)
    output.write_text(json.dumps(dict(complete=True,targets=results,unsupported=report['unsupported'],bindings=bindings,
        macro_mae={a:float(np.mean([r['mae'][a] for r in results.values()])) for a in ('selected','original','population','all_probe')},
        runtime_modified=False,promotion_allowed=False,
        limitation='Reused development evaluation; source tune selects checkpoint and readout. Unadjusted nominal CIs, no external/forecast/conformal promotion.'),indent=2)+'\n')
    print('completed',output)


if __name__=='__main__':main()
