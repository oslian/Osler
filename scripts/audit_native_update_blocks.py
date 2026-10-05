"""Source-evaluation weight-restoration diagnostic, not a trained candidate."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.audit_native_glucose_retention import all_points
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.native_event_encoder import native_arrays,NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events,append_events,expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911');out=Path('artifacts/native_update_blocks_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_private_update_20260911.json').read_text())
    for n,h in ref['binding'].items():
        if sha256(Path(n))!=h:raise ValueError('changed input: '+n)
    def load(n):
        with np.load(root/n) as f:return {k:f[k] for k in f.files}
    base,native,labels=load('body_arrays.npz'),load('native_tokens.npz'),load('multitarget_labels.npz')
    valid=load('labels.npz')['paired'][:,0]
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));vs=assets.checkpoint['variables'];col=vs.index('glucose')
    arrays,rows,values=native_arrays(base,native,len(vs),base['presence'].shape[1])
    report=dict(runtime_modified=False,training_performed=False,causal_claim_allowed=False,binding=ref['binding'].copy(),seeds=[])
    report['binding'][str(Path(__file__))]=sha256(Path(__file__))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for entry in ref['seeds']:
        seed=entry['seed'];part=load(f'partition_{seed}.npz');fit=part['fit'][valid]&labels['paired'][:,col];evaluation=part['evaluation'][valid]&labels['paired'][:,col]
        cp=torch.load(root/f'glucose_native_real_{seed}.pt',map_location='cpu',weights_only=False)
        events,_,_=prepare_events(values,np.unique(arrays['history'][rows[fit]]))
        view=append_events(restrict_input_support(arrays,np.asarray(cp['allowed_variables']),assets.model.variable_module_ids.numpy()),events)
        torch.manual_seed(seed);model=deepcopy(assets.model);expand_history_projection(model,6)
        nf=NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter=nf
        model.value_delta_activation=torch.nn.Identity();model.belief_input_mode='observed_steps'
        initial=deepcopy(model.state_dict());final=cp['state_dict'];idx=np.flatnonzero(evaluation)
        model.load_state_dict(final);original=all_points(model,view,rows[idx],assets.scaler.medians,assets.scaler.scales)
        cached=load(f'glucose_retention_predictions_{seed}.npz')['original'][idx]
        np.testing.assert_allclose(original,cached,atol=1e-4,rtol=1e-6)
        item=dict(seed=seed,blocks={})
        blocks={'history':('belief_filter.',),'mixers':('variable_mixer.','module_mixer.','state_encoder.'),'shared_future':('latent_predictor.',),'glucose_private':tuple(f'{n}.{col}.' for n in ('target_future_predictors','target_temporal_adapters','value_heads'))}
        for name,prefixes in blocks.items():
            state={k:(initial[k] if k.startswith(prefixes) else v) for k,v in final.items()}
            changed=sum(not torch.equal(v,final[k]) for k,v in state.items())
            model.load_state_dict(state);pred=all_points(model,view,rows[idx],assets.scaler.medians,assets.scaler.scales)
            result={}
            for j,v in enumerate(vs):
                m=labels['paired'][idx,j];f=q.iloc[idx[m]].copy();f['truth']=labels['future'][idx[m],j]
                if f.subject_id.nunique()<30:continue
                result[v]=dict(mae=patient_equal_mae(f,f.truth,pred[m,j]),original_mae=patient_equal_mae(f,f.truth,original[m,j]),comparison=paired(f,pred[m,j],original[m,j],f'blocks:{seed}:{name}:{v}',2000))
            item['blocks'][name]=dict(changed_tensors=changed,targets=result)
        report['seeds'].append(item);print('complete',seed,flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
