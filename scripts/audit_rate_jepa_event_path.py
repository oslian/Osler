"""Frozen event-path use after matched history training; not causal effects."""
import json
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import predict,batch
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.history_event_input import append_events,expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stages',action='store_true');args=parser.parse_args()
    output=Path('artifacts/rate_jepa_event_stages_20260911.json' if args.stages else 'artifacts/rate_jepa_event_path_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    ref=json.loads(Path('artifacts/rate_anchor_jepa_events_20260911.json').read_text())
    lab=np.load(root/'labels.npz');valid=lab['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    q['truth']=lab['future'][valid,0]
    arrays=dict(np.load(root/'body_arrays.npz'));rows=arrays.pop('query_rows')
    raw=np.load(root/'event_history.npz')['values'];known=np.isfinite(raw)
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    torch.set_num_threads(4);report=dict(complete=False,fitting_performed=False,runtime_modified=False,seeds=[])
    for entry in ref['seeds']:
        seed=entry['seed'];path=root/f'jepa_event_real_{seed}.pt'
        if sha256(path)!=ref['binding'][str(path)]:raise ValueError('weights changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=deepcopy(assets.model);expand_history_projection(model,saved['event_width']);model.load_state_dict(saved['state_dict'])
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
        base=restrict_input_support(arrays,np.asarray(saved['allowed_variables'],bool),model.variable_module_ids.numpy())
        numbers=np.where(known,(raw-np.asarray(saved['event_center']))/np.asarray(saved['event_scale']),0.)
        event=np.column_stack([numbers,known]).astype(np.float32)
        points={};states={}
        for mode in ('full','numbers_off','all_off'):
            e=event.copy()
            if mode=='numbers_off':e[:,:3]=0.
            if mode=='all_off':e[:]=0.
            points[mode]=predict(model,append_events(base,e),rows,col,assets.scaler.medians[col],assets.scaler.scales[col])
            if args.stages and mode in ('full','numbers_off'):
                stages={k:[] for k in ('belief_state','context_latent','predicted_target_latent')}
                model.eval();view=append_events(base,e)
                with torch.inference_mode():
                    for start in range(0,len(rows),256):
                        out=model(*batch(view,rows[start:start+256]))
                        for k in stages:
                            x=out[k][:,col] if k=='predicted_target_latent' else out[k]
                            stages[k].append(x.numpy())
                states[mode]={k:np.concatenate(v).astype(float) for k,v in stages.items()}
        roles={k:v[valid] for k,v in dict(np.load(root/f'partition_{seed}.npz')).items()}
        w=model.belief_filter.observation_projection[0].weight.detach().numpy()
        result=dict(seed=seed,event_weight_norm=float(np.linalg.norm(w[:,-7:-1])),roles={})
        for role in ('evaluation','hospital'):
            m=roles[role];f=q.loc[m]
            maes={k:patient_equal_mae(f,f.truth,p[m]) for k,p in points.items()}
            if abs(maes['full']-entry['roles'][role]['all']['mae']['event_real'])>1e-7:raise ValueError('reproduction failed')
            result['roles'][role]=dict(mae=maes,mean_absolute_prediction_shift={k:float(np.mean(np.abs(points[k][m]-points['full'][m]))) for k in ('numbers_off','all_off')})
            if args.stages:
                result['roles'][role]['stages']={}
                for k,x in states['full'].items():
                    shift=np.sqrt(np.mean((x[m]-states['numbers_off'][k][m])**2))
                    variation=np.sqrt(np.mean(np.var(x[m],axis=0)))
                    result['roles'][role]['stages'][k]=dict(rms_shift=float(shift),rms_between_row_variation=float(variation),relative_shift=float(shift/max(variation,1e-12)))
        report['seeds'].append(result)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
