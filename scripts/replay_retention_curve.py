"""Source-qualified reference retention on the fixed native glucose trial."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_native_glucose_retention import retention_loss
from scripts.train_availability_encoder import batch, configure, predict, teacher_target
from scripts.audit_native_glucose_retention import all_points
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.native_event_encoder import native_arrays, NativeEventFilter
from whole_body_jepa.history_event_input import prepare_events, append_events, expand_history_projection
from whole_body_jepa.availability_inputs import restrict_input_support, source_patient_support
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_retention_curve_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    reference = json.loads(Path('artifacts/native_retention_repair_20260911.json').read_text())
    selection_path = Path('artifacts/native_retention_selection_20260911.json')
    selection = json.loads(selection_path.read_text())
    bindings = reference['binding'] | selection['binding']
    for name, digest in bindings.items():
        if sha256(Path(name)) != digest: raise ValueError('input changed: ' + name)
    def load(name):
        with np.load(root / name) as f: return {k:f[k] for k in f.files}
    base, native, labels = load('body_arrays.npz'), load('native_tokens.npz'), load('multitarget_labels.npz')
    valid = load('labels.npz')['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets = load_release(Path('models')); variables = assets.checkpoint['variables']; col = variables.index('glucose')
    medians, scales = assets.scaler.medians, assets.scaler.scales
    eligible = labels['paired'][:,col]; q['truth'] = labels['future'][:,col]; truth = (q.truth.to_numpy()-medians[col])/scales[col]
    arrays, rows, values = native_arrays(base,native,len(variables),base['presence'].shape[1])
    report = dict(complete=False,runtime_modified=False,status='candidate_only',seeds=[],binding=dict(bindings))
    for p in (Path(__file__),selection_path,Path('NATIVE_RETENTION_SELECTION_20260911.md')): report['binding'][str(p)] = sha256(p)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    original_report=json.loads(Path('artifacts/native_selected_retention_20260911.json').read_text())
    for selection_entry in selection['seeds']:
        seed = selection_entry['seed']; roles = {a:b[valid]&eligible for a,b in load(f'partition_{seed}.npz').items()}
        fit,tune = roles['fit'],roles['tune']
        pools = [np.flatnonzero(fit&q.subject_id.eq(s).to_numpy()) for s in sorted(set(q.loc[fit,'subject_id']))]
        cp = torch.load(root/f'glucose_native_real_{seed}.pt',map_location='cpu',weights_only=False)
        allowed = np.asarray(cp['allowed_variables'])
        events,center,event_scale = prepare_events(values,np.unique(arrays['history'][rows[fit]]))
        view = append_events(restrict_input_support(arrays,allowed,assets.model.variable_module_ids.numpy()),events)
        torch.manual_seed(seed);rng = np.random.default_rng(seed)
        model = deepcopy(assets.model);expand_history_projection(model,6)
        nf = NativeEventFilter(model.input_dim,model.belief_filter.hidden);nf.load_state_dict(model.belief_filter.state_dict());model.belief_filter = nf
        model.value_delta_activation = torch.nn.Identity();model.belief_input_mode = 'observed_steps'
        cached = load(f'glucose_retention_predictions_{seed}.npz')
        initial = all_points(model,view,rows,medians,scales)
        np.testing.assert_allclose(initial,cached['initial'],atol=1e-4,rtol=1e-6)
        teacher_points = torch.tensor((initial-medians)/scales,dtype=torch.float32)
        supported = source_patient_support(base,base['query_rows'],q.subject_id,fit)>=30
        supported &= np.isin(variables,selection_entry['selected']);supported[col] = False
        retain_mask = torch.tensor((view['mask'][rows]>0)&supported[None,:])
        params = configure(model,True);optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        best_mae = patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],initial[tune,col]);best = deepcopy(model.state_dict());selected=0;log=[]
        curve=[]
        target_rows={j:np.flatnonzero(tune&labels['paired'][:,j]) for j in range(len(variables))
                     if q.loc[tune&labels['paired'][:,j],'subject_id'].nunique()>=30 and q.loc[fit&labels['paired'][:,j],'subject_id'].nunique()>=30}
        def record(epoch, prediction):
            errors={variables[j]:patient_equal_mae(q.iloc[ix],labels['future'][ix,j],prediction[ix,j]) for j,ix in target_rows.items()}
            curve.append(dict(epoch=epoch,mae=errors))
        record(0,initial)
        for epoch in range(32):
            for _ in range(16):
                chosen = np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)])
                inputs=batch(view,rows[chosen]);result=model(*inputs);y=torch.tensor(truth[chosen],dtype=torch.float32)
                teacher=teacher_target(model,inputs[1],y,col)
                primary=F.smooth_l1_loss(result['value'][:,col],y)+_masked_latent_prediction_loss(result['predicted_target_latent'][:,col],teacher,torch.ones(len(y),dtype=torch.bool))
                loss=primary+retention_loss(result['value'],teacher_points[chosen],retain_mask[chosen])
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();model.update_target_encoder()
            mae=patient_equal_mae(q.loc[tune],q.loc[tune,'truth'],predict(model,view,rows[tune],col,medians[col],scales[col]))
            if mae<best_mae:best_mae,best,selected=mae,deepcopy(model.state_dict()),epoch+1
            log.append(dict(epoch=epoch+1,tune_mae=mae))
            record(epoch+1,all_points(model,view,rows,medians,scales))
        old=next(x for x in original_report['seeds'] if x['seed']==seed)
        if log!=old['log']:raise ValueError('training trajectory changed')
        item=dict(seed=seed,primary_selected_epoch=selected,curve=curve,original_log_exact=True)
        report['seeds'].append(item)
        (root/'retention_curve_progress.json').write_text(json.dumps(report,indent=2)+'\\n')
        print('complete',seed,selected,flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

