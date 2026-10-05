"""Decompose selected source-tune score without treating MAE as gradient mass."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    out=Path('artifacts/native_reconstruction_site_objective_20260917.json')
    if out.exists():raise ValueError('existing diagnostic')
    path=Path('artifacts/native_reconstruction_site_training_epoch8_20260917.json')
    report=json.loads(path.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed training dependency')
    roles=torch.load('data/research_recovery/reconstruction_site_partition_20260917.pt',weights_only=False)['roles']
    index=torch.load('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt',weights_only=False)
    variables=index['variables'];targets=[variables.index(t) for t in report['source_targets']]
    fit={c:[] for c in targets};tune={c:[] for c in targets}
    for shard in index['shards']:
        role=roles[shard['patient']]
        if role not in ('fit','tune'):continue
        p=durable(shard['path'])
        if sha256(p)!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(p,weights_only=False)
        for c,e in s['samples'].items():
            if c not in targets:continue
            if role=='fit':fit[c].append(e['truth_normalized'])
            else:tune[c].append(e)
    medians={c:float(np.median(v)) for c,v in fit.items()}
    initial={c:float(np.mean([abs(e['truth_normalized']-medians[c]) for e in tune[c]])) for c in targets}
    if not np.isclose(np.mean(list(initial.values())),report['arms']['real']['initial_tune'],rtol=0,atol=1e-7):raise ValueError('initial score mismatch')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=load_release(Path('models'));jepa=assets.model
    expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    template=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    results={};dependencies=[Path(__file__),path]
    for arm in ('real','process'):
        cp=Path(f'data/research_recovery/site_reconstruction_training_20260917/{arm}.pt')
        ck=torch.load(cp,weights_only=False)
        if ck['bindings']!=report['bindings'] or ck['epoch']!=8:raise ValueError('checkpoint mismatch')
        model=deepcopy(template);model.load_state_dict(ck['best_state_dict'])
        scores={}
        with torch.no_grad():
            for c in targets:
                errors=[]
                for e in tune[c]:
                    p=float(model(input_tokens(e,arm,len(variables)),e['event_hours'],e['anchor_hour'],variables[c])[0])
                    errors.append(abs(p-e['truth_normalized']))
                scores[c]=float(np.mean(errors))
        if not np.isclose(np.mean(list(scores.values())),report['arms'][arm]['best_tune'],rtol=0,atol=1e-7):raise ValueError('selected score mismatch')
        results[arm]=scores;dependencies.append(cp)
        print('reproduced',arm,flush=True)
    total=sum(initial.values());n=len(targets)
    rows={variables[c]:dict(fit_patients=len(fit[c]),tune_patients=len(tune[c]),
        initial_tune_mae=initial[c],initial_score_fraction=initial[c]/total,
        fit_median_absolute_deviation=float(np.median(abs(np.asarray(fit[c])-medians[c]))),
        selected_tune_mae={a:results[a][c] for a in results},
        score_change_contribution={a:(results[a][c]-initial[c])/n for a in results}) for c in targets}
    out.write_text(json.dumps(dict(complete=True,targets=rows,bindings={str(p):sha256(p) for p in dependencies},
        runtime_modified=False,promotion_allowed=False,
        limitation='Source-tune selected endpoints only, not per-epoch optimum or independent evaluation. Absolute loss share is not gradient share or selection-change share; no counterfactual checkpoint-selection or new normalization was tested.'),indent=2)+'\n')
    for t,v in sorted(rows.items(),key=lambda x:-x[1]['initial_score_fraction'])[:8]:
        print(t,v,flush=True)


if __name__=='__main__':main()
