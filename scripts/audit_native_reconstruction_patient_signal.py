"""Frozen trained reconstruction: patient-specific versus constant corrections."""
import json
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
    out=Path('artifacts/native_reconstruction_patient_signal_20260913.json')
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
        predictions={}
        with torch.no_grad():
            for role,d in data.items():
                predictions[role]={k:float(model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[k[1]])[0]) for k,e in d.items()}
        constants={c:float(np.mean([v for k,v in predictions['fit'].items() if k[1]==c])) for c in targets}
        roles={}
        for role,d in data.items():
            rows={}
            for c in targets:
                keys=[k for k in d if k[1]==c]
                if len(keys)<2:continue
                p=np.array([predictions[role][k] for k in keys]);y=np.array([d[k]['truth_normalized'] for k in keys])
                perm=np.roll(p,1)
                mae=float(np.mean(abs(p-y)))
                if role in original['roles']:
                    expected=original['roles'][role][variables[c]]['normalized_mae'][arm]
                    if abs(mae-expected)>1e-7:raise ValueError('prediction reproduction mismatch')
                rows[variables[c]]=dict(patients=len(keys),prediction_std=float(p.std()),
                    mean_abs_correction=float(np.mean(abs(p-medians[c]))),
                    patient_permutation_mean_abs_shift=float(np.mean(abs(p-perm))),
                    model_mae=mae,permuted_mae=float(np.mean(abs(perm-y))),
                    fit_constant_mae=float(np.mean(abs(constants[c]-y))),
                    population_mae=float(np.mean(abs(medians[c]-y))))
            roles[role]=rows
        reports[arm]=roles
    summary={a:{r:{k:float(np.median([v[k] for v in t.values()])) for k in
        ('prediction_std','mean_abs_correction','patient_permutation_mean_abs_shift')}
        for r,t in roles.items()} for a,roles in reports.items()}
    report=dict(complete=True,roles=reports,summary=summary,accuracy_reproduced=True,
        scope='frozen-model descriptive diagnostic; fit-mean predictor not independently trained control',
        runtime_modified=False,training_performed=False,
        bindings={str(p):sha256(p) for p in (parent,Path(__file__))})
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
