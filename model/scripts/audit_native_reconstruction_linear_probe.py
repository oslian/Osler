"""Frozen trained reconstruction: patient-specific versus constant corrections."""
import json
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from whole_body_jepa.change_evaluation import paired_model_errors
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.multisource_intervals import sha256


def main():
    parent=Path('artifacts/native_reconstruction_sampling_20260913.json')
    original=json.loads(parent.read_text())
    for name,h in original['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed training dependency')
    out=Path('artifacts/native_reconstruction_linear_probe_20260913.json')
    if out.exists():raise ValueError('fresh output required')
    private=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support/multitarget_sampling_7')
    saved=torch.load(private/'samples.pt',weights_only=False);targets=saved['targets']
    data={r:{k:e for k,e in d.items() if k[1] in targets} for r,d in saved['data'].items()}
    medians={c:float(np.median([e['truth_normalized'] for k,e in data['fit'].items() if k[1]==c])) for c in targets}
    torch.set_num_threads(4);assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    reports={}
    for arm in ('event_weighted','balanced'):
        checkpoint=torch.load(private/f'{arm}.pt',weights_only=False);model.load_state_dict(checkpoint['state_dict'])
        predictions={};features={};captured=[]
        handle=model.readout.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().copy()[0]))
        with torch.no_grad():
            for role,d in data.items():
                predictions[role]={};features[role]={}
                for k,e in d.items():
                    captured.clear()
                    predictions[role][k]=float(model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                    if len(captured)!=1:raise ValueError('one body readout required')
                    features[role][k]=captured[0]
        handle.remove()
        results={};selection={}
        for c in targets:
            keys={r:[k for k in d if k[1]==c] for r,d in data.items()}
            x={r:np.asarray([features[r][k] for k in kk]) for r,kk in keys.items()}
            y={r:np.array([data[r][k]['truth_normalized'] for k in kk]) for r,kk in keys.items()}
            fitted={}
            selection[variables[c]]={}
            for probe in ('real','placebo'):
                inputs={r:np.roll(v,1,axis=0) if probe=='placebo' else v for r,v in x.items()}
                best=(float(np.mean(abs(medians[c]-y['tune']))),None)
                for alpha in (1.,10.,100.,1000.,10000.):
                    fitmodel=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                    fitmodel.fit(inputs['fit'],y['fit'])
                    score=float(np.mean(abs(fitmodel.predict(inputs['tune'])-y['tune'])))
                    if score<best[0]:best=(score,fitmodel)
                selection[variables[c]][probe]=dict(tune_mae=best[0],alpha=None if best[1] is None else best[1][-1].alpha)
                fitted[probe]={r:np.full(len(keys[r]),medians[c]) if best[1] is None or len(keys[r])==0 else best[1].predict(inputs[r]) for r in ('evaluation','hospital')}
            result={}
            for role in ('evaluation','hospital'):
                if len(keys[role])<2:continue
                p={probe:fitted[probe][role] for probe in fitted}
                p['original']=np.array([predictions[role][k] for k in keys[role]])
                p['population']=np.full(len(keys[role]),medians[c])
                expected=original['roles'][role][variables[c]]['normalized_mae'][arm]
                if abs(np.mean(abs(p['original']-y[role]))-expected)>1e-7:raise ValueError('original mismatch')
                frame=pd.DataFrame(dict(subject_id=[k[0] for k in keys[role]],truth=y[role]))
                result[role]=dict(normalized_mae={k:float(np.mean(abs(v-y[role]))) for k,v in p.items()},
                    comparisons={ref:paired_model_errors(frame,p,ref,seed=20260913,repetitions=20000) for ref in ('population','original','placebo')})
            results[variables[c]]=result
        reports[arm]=dict(selection=selection,results=results)
    summary={a:{role:{ref:[target for target,r in report['results'].items() if role in r and
        (r[role]['comparisons'][ref]['real']['ci95'] or [0,0])[1]<0] for ref in ('population','original','placebo')}
        for role in ('evaluation','hospital')} for a,report in reports.items()}
    report=dict(complete=True,roles=reports,summary=summary,accuracy_reproduced=True,
        scope='frozen encoder, source-selected ridge probes; small reused validation, no promotion',
        runtime_modified=False,encoder_trained=False,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('NATIVE_RECONSTRUCTION_LINEAR_PROTOCOL_20260913.md'))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

