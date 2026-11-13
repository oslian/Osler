"""Compare trained numeric-free control with frozen expanded real reconstruction."""
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
from scripts.audit_expanded_reconstruction_numeric_path import neutralize_numbers


def main():
    source=Path('artifacts/native_reconstruction_process_training_epoch8_20260914.json')
    original=Path('artifacts/native_reconstruction_expanded_evaluation_20260913.json')
    trained=json.loads(source.read_text());parent=json.loads(original.read_text())
    if not trained['training_complete'] or trained['arms']['process']['epoch']!=8:raise ValueError('incomplete training')
    for report in (trained,parent):
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('changed dependency')
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private=root/'expanded_process_training_20260914';path=private/'process.pt'
    ck=torch.load(path,weights_only=False)
    if ck['bindings']!=trained['bindings'] or ck['epoch']!=8:raise ValueError('wrong checkpoint')
    assets=load_release(Path('models'));variables=assets.checkpoint['variables']
    torch.set_num_threads(4);jepa=assets.model;expand_history_projection(jepa,6);jepa.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(jepa,variables,('map',))).eval();model.load_state_dict(ck['best_state_dict'])
    saved=torch.load(root/'expanded_training_20260913/heldout_predictions.pt',weights_only=False)
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
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
                p=float(model(neutralize_numbers(e['tokens'],len(variables)),e['event_hours'],e['anchor_hour'],variables[c])[0])
                rows[role].append(dict(subject_id=s['patient'],target=variables[c],truth=e['truth_normalized'],
                    real=saved['predictions']['real'][role][key],process=p,population=saved['medians'][c]))
    result={}
    for role,records in rows.items():
        result[role]={}
        for target,part in pd.DataFrame(records).groupby('target'):
            p={a:part[a].to_numpy() for a in ('real','process','population')}
            mae={a:float(np.mean(abs(v-part.truth.to_numpy()))) for a,v in p.items()}
            if not np.isclose(mae['real'],parent['roles'][role][target]['normalized_mae']['real'],rtol=0,atol=1e-7):raise ValueError('parent alignment mismatch')
            result[role][target]=dict(patients=len(part),normalized_mae=mae,
                comparisons={ref:paired_model_errors(part,p,ref,seed=20260914) for ref in ('population','process')})
    out=Path('artifacts/native_reconstruction_process_evaluation_20260914.json')
    if out.exists():raise ValueError('existing evidence')
    torch.save(rows,private/'heldout_predictions.pt')
    out.write_text(json.dumps(dict(complete=True,roles=result,runtime_modified=False,promotion_allowed=False,
        limitation='Single reused split, nominal paired intervals. Process control not randomized organ causal evidence.',
        bindings={str(p):sha256(p) for p in (Path(__file__),source,original,path,private/'heldout_predictions.pt')}),indent=2)+'\n')
    print('complete',flush=True)


if __name__=='__main__':main()
