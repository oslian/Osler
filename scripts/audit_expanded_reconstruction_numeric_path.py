"""Frozen numeric neutralization preserving native observation support and time."""
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


def neutralize_numbers(tokens, variable_count):
    if tokens.ndim!=2 or variable_count<1 or tokens.shape[1]<3*variable_count:
        raise ValueError('native variable schema required')
    result=tokens.clone();result[:,:variable_count]=0
    return result


def main():
    source=Path('artifacts/native_reconstruction_expanded_evaluation_20260913.json')
    parent=json.loads(source.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('changed parent dependency')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private=root/'expanded_training_20260913'
    output=Path('artifacts/native_reconstruction_numeric_path_20260914.json')
    if output.exists():raise ValueError('existing evidence')
    saved=torch.load(private/'heldout_predictions.pt',weights_only=False)
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
    ck=torch.load(private/'real.pt',weights_only=False)
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    torch.set_num_threads(4);jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval()
    model.load_state_dict(ck['best_state_dict'])
    rows={role:[] for role in parent['roles']}
    with torch.no_grad():
        for shard in index['shards']:
            role=shard['role']
            if role not in rows:continue
            if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('changed shard')
            s=torch.load(shard['path'],weights_only=False)
            for c,e in s['samples'].items():
                key=(s['patient'],c)
                if key not in saved['predictions']['real'][role]:continue
                def predict(x):return float(model(x,e['event_hours'],e['anchor_hour'],variables[c])[0])
                original=predict(e['tokens'])
                if not np.isclose(original,saved['predictions']['real'][role][key],rtol=0,atol=1e-7):
                    raise ValueError('original prediction mismatch')
                neutral=predict(neutralize_numbers(e['tokens'],len(variables)))
                rows[role].append(dict(subject_id=s['patient'],target=variables[c],truth=e['truth_normalized'],original=original,neutral=neutral))
    results={}
    for role,data in rows.items():
        frame=pd.DataFrame(data);results[role]={}
        for target,part in frame.groupby('target'):
            pred={a:part[a].to_numpy() for a in ('original','neutral')}
            results[role][target]=dict(patients=len(part),normalized_mae={a:float(np.mean(abs(p-part.truth.to_numpy()))) for a,p in pred.items()},
                neutral_minus_original=paired_model_errors(part,pred,'original',seed=20260914)['neutral'],
                mean_absolute_prediction_shift=float(np.mean(abs(pred['neutral']-pred['original']))))
    paths=[Path(__file__),source,private/'real.pt']
    output.write_text(json.dumps(dict(complete=True,roles=results,runtime_modified=False,promotion_allowed=False,
        limitation='Inference-only off-distribution input intervention. Not a retrained process-only control, causal physiology proof or deployable correction.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print('complete',flush=True)


if __name__=='__main__':main()
