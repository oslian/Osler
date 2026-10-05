"""Evaluate frozen selected site-partition models, without new selection."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.audit_reconstruction_source_fit import durable
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    training_path=Path('artifacts/native_reconstruction_site_training_epoch8_20260917.json')
    source_path=Path('artifacts/native_reconstruction_site_partition_20260917.json')
    out=Path('artifacts/native_reconstruction_site_evaluation_20260917.json')
    private=Path('data/research_recovery/site_reconstruction_training_20260917')
    saved=private/'evaluation_predictions.pt'
    if out.exists() or saved.exists():raise ValueError('existing evaluation')
    trained=json.loads(training_path.read_text());source=json.loads(source_path.read_text())
    if not trained['training_complete']:raise ValueError('incomplete training')
    for report in (trained,source):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('changed binding')
    roles=torch.load('data/research_recovery/reconstruction_site_partition_20260917.pt',weights_only=False)['roles']
    index=torch.load('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt',weights_only=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    targets=[variables.index(t) for t in trained['source_targets']]
    fit={c:[] for c in targets};examples=[]
    for shard in index['shards']:
        role=roles[shard['patient']]
        if role not in ('fit','evaluation'):continue
        path=durable(shard['path'])
        if sha256(path)!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(path,weights_only=False)
        if s['patient']!=shard['patient']:raise ValueError('identity mismatch')
        for c,e in s['samples'].items():
            if c not in targets:continue
            if role=='fit':fit[c].append(e['truth_normalized'])
            else:examples.append((s['patient'],c,e))
    medians={c:float(np.median(v)) for c,v in fit.items()}
    jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    models={};paths=[]
    for arm in ('real','process','placebo'):
        path=private/f'{arm}.pt';ck=torch.load(path,weights_only=False)
        if (ck['epoch']!=8 or ck['bindings']!=trained['bindings'] or ck['targets']!=targets
                or ck['selected_epoch']!=trained['arms'][arm]['selected_epoch']):raise ValueError('wrong model')
        model=deepcopy(template);model.load_state_dict(ck['best_state_dict']);models[arm]=model;paths.append(path)
    rows=[]
    with torch.no_grad():
        for i,(patient,c,e) in enumerate(examples):
            pred={a:float(m(input_tokens(e,a,len(variables)),e['event_hours'],e['anchor_hour'],variables[c])[0]) for a,m in models.items()}
            rows.append(dict(subject_id=patient,target=variables[c],truth=e['truth_normalized'],population=medians[c],**pred))
            if (i+1)%1000==0:print('evaluated',i+1,'/',len(examples),flush=True)
    frame=pd.DataFrame(rows)
    if frame.duplicated(['subject_id','target']).any():raise ValueError('duplicate patient-target')
    results={}
    for target,part in frame.groupby('target'):
        if len(part)!=source['target_counts'][target]['evaluation']:raise ValueError('support mismatch')
        p={a:part[a].to_numpy() for a in ('real','process','placebo','population')}
        comparisons={ref:paired_model_errors(part,p,ref,seed=20260917) for ref in ('process','placebo','population')}
        results[target]=dict(patients=len(part),normalized_mae={a:float(np.mean(abs(v-part.truth.to_numpy()))) for a,v in p.items()},comparisons=comparisons,
            nominal_all_gain=all(v['real']['supported'] and v['real']['ci95'][1]<0 for v in comparisons.values()))
    torch.save(rows,saved)
    paths.extend([Path(__file__),training_path,source_path,saved,Path('whole_body_jepa/change_evaluation.py')])
    out.write_text(json.dumps(dict(complete=True,targets=results,unsupported=trained['source_unsupported'],
        nominal_all_gain=[t for t,v in results.items() if v['nominal_all_gain']],
        runtime_modified=False,promotion_allowed=False,
        limitation='One fixed development site split, nominal patient intervals without multiplicity correction; pretraining independence uncertified. No future forecast, conformal or external promotion.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print('complete; nominal gains',[t for t,v in results.items() if v['nominal_all_gain']],flush=True)


if __name__=='__main__':main()
