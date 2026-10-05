"""Evaluate completed, source-selected expanded shared-body checkpoints."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.runtime import load_release
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors


def main():
    source=Path('artifacts/native_reconstruction_expanded_training_epoch8_20260913.json')
    output=Path('artifacts/native_reconstruction_expanded_evaluation_20260913.json')
    if output.exists():raise ValueError('existing evaluation must not be overwritten')
    trained=json.loads(source.read_text())
    if not trained['training_complete'] or any(a['epoch']!=8 for a in trained['arms'].values()):
        raise ValueError('finish both eight-epoch arms before inspecting heldout errors')
    for p,h in trained['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('training contract changed')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    index_path=root/'expanded_reconstruction_20260913/index.pt'
    cache_report=json.loads(Path('artifacts/native_reconstruction_expanded_20260913.json').read_text())
    for p,h in cache_report['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('cache contract changed')
    index=torch.load(index_path,weights_only=False)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    targets=[variables.index(v) for v in trained['source_targets']]
    data={role:{} for role in ('fit','evaluation','hospital')}
    for shard in index['shards']:
        role=shard['role']
        if role not in data:continue
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('modified sample shard')
        s=torch.load(shard['path'],weights_only=False)
        if s['role']!=role or s['patient'] not in index['roles'][role]:raise ValueError('patient role mismatch')
        for c,e in s['samples'].items():
            if c in targets:data[role][(s['patient'],c)]=e
    medians={c:float(np.median([e['truth_normalized'] for k,e in data['fit'].items() if k[1]==c])) for c in targets}
    del data['fit']
    torch.set_num_threads(4);jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    predictions={};weights=[]
    for arm in ('real','placebo'):
        path=root/'expanded_training_20260913'/f'{arm}.pt';weights.append(path)
        ck=torch.load(path,weights_only=False)
        if ck['epoch']!=8 or ck['bindings']!=trained['bindings'] or ck['selected_epoch']!=trained['arms'][arm]['selected_epoch']:
            raise ValueError('checkpoint selection mismatch')
        model.load_state_dict(ck['best_state_dict']);predictions[arm]={}
        with torch.no_grad():
            for role,examples in data.items():
                predictions[arm][role]={k:float(model(e['tokens'],e['event_hours'],e['anchor_hour'],variables[k[1]])[0])
                                         for k,e in examples.items()}
                print('predicted',arm,role,len(examples),flush=True)
    results={}
    for role,examples in data.items():
        results[role]={}
        for c in targets:
            keys=[k for k in examples if k[1]==c]
            if not keys:
                results[role][variables[c]]=dict(patients=0,supported=False);continue
            frame=pd.DataFrame(dict(subject_id=[k[0] for k in keys],truth=[examples[k]['truth_normalized'] for k in keys]))
            pred={arm:np.array([predictions[arm][role][k] for k in keys]) for arm in predictions}
            pred['population']=np.full(len(keys),medians[c])
            comparisons={ref:paired_model_errors(frame,pred,ref,seed=20260913,repetitions=20000)
                         for ref in ('population','placebo')}
            supported=len(keys)>=30
            gain=supported and all(comparisons[ref]['real']['ci95'][1]<0 for ref in comparisons)
            results[role][variables[c]]=dict(patients=len(keys),supported=supported,
                normalized_mae={a:float(np.mean(abs(v-frame.truth.to_numpy()))) for a,v in pred.items()},
                comparisons=comparisons,nominal_pairwise_gain=bool(gain))
    private=root/'expanded_training_20260913'/'heldout_predictions.pt'
    if private.exists():raise ValueError('private predictions already exist')
    torch.save(dict(predictions=predictions,medians=medians),private)
    paths=[Path(__file__),source,index_path,*weights,private,Path('whole_body_jepa/change_evaluation.py')]
    report=dict(complete=True,roles=results,
                nominal_both_role_gains=[variables[c] for c in targets if all(results[r][variables[c]].get('nominal_pairwise_gain',False) for r in results)],
                limitation='Single reused source split; nominal unadjusted paired CIs, not full promotion or future forecasting.',
                runtime_modified=False,promotion_allowed=False,bindings={str(p):sha256(p) for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n');print('complete',flush=True)


if __name__=='__main__':main()
