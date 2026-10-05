"""Two optimizer steps on a real source pair; integration, not efficacy."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_availability_encoder import configure, teacher_target
from whole_body_joint_jepa import _masked_latent_prediction_loss
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/native_organ_future_supervision_20260913.json')
    if output.exists(): raise ValueError('fresh output required')
    root = Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent = Path('artifacts/map_native_query_training_7_20260912.json')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name)) != h: raise ValueError('changed baseline dependency')
    q = pd.read_parquet(root/'queries.parquet'); labels = dict(np.load(root/'labels.npz'))
    fit = np.load(root/'partition_7.npz')['fit'] & labels['paired'][:,0]
    transforms = dict(np.load(root/'neural_transforms.npz')); clocks = [np.array([np.nan])]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            clocks.append(np.load(root/'unclipped_clock_shards'/p.name)['event_hours'])
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw = RaggedNativeBatch(shards(),len(q)); hours = np.concatenate(clocks)
    assets = load_release(Path('models')); jepa = assets.model
    variables = assets.checkpoint['variables']; n = len(variables); col = variables.index('map')
    views = {}
    for arm in ('own','full'):
        allowed = transforms['7_allowed'].copy()
        if arm=='own': allowed[np.arange(n)!=col] = False
        views[arm] = transformed_storage(raw, transforms['7_numeric_center'], transforms['7_numeric_scale'],
            allowed,jepa.variable_module_ids.numpy(),np.flatnonzero(fit),own=arm=='own')[0]
    expand_history_projection(jepa,6)
    jepa.belief_filter = NativeEventFilter(jepa.input_dim,jepa.belief_filter.hidden)
    jepa.belief_input_mode='observed_steps'; jepa.value_delta_activation=torch.nn.Identity()
    checkpoint = root/'native_query_own_7.pt'
    jepa.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']); jepa.eval()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.manual_seed(7)
    row = int(np.flatnonzero(fit)[0]); sequence = views['full'].sequences[row]
    x = views['own'].batch([row])
    with torch.inference_mode(): baseline = jepa(*x)
    base_value = baseline['value'][:,col:col+1].clone().detach()
    base_latent = baseline['predicted_target_latent'][:,col:col+1].clone().detach()
    model = NativeOrganStateResidual(deepcopy(jepa),variables,('map',)).eval()
    configure(model.jepa,True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    tokens = torch.from_numpy(views['full'].input[sequence].copy()); tokens[:,-7:-1]=0
    times = torch.from_numpy(hours[sequence])
    # Use recovered clocks in the candidate rather than the baseline's clipped representation.
    tokens[:,3*n] = (times/24).float(); tokens[:,-1] = (times/168).float()
    anchor = float((pd.Timestamp(q.iloc[row]['t'])-pd.Timestamp('2100-01-01')).total_seconds()/3600)
    y = torch.tensor([(labels['future'][row,0]-assets.scaler.medians[col])/assets.scaler.scales[col]],dtype=torch.float32)
    args = (tokens,times,anchor,x[1][:,col:col+1],x[2][:,col:col+1],
        x[5].reshape(1,1),base_value,base_latent)
    log = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True); result = model(*args)
        if step==0:
            torch.testing.assert_close(result['value'],base_value,rtol=0,atol=0)
            torch.testing.assert_close(result['future_latent'],base_latent,rtol=0,atol=0)
        value_loss = torch.nn.functional.smooth_l1_loss(result['value'][:,0],y)
        latent_loss = _masked_latent_prediction_loss(result['future_latent'][:,0],
            teacher_target(model.jepa,x[1],y,col),torch.ones(1,dtype=torch.bool))
        loss=value_loss+latent_loss; loss.backward()
        grad=model.layer.organ_update_gate[0].weight.grad
        norm=0. if grad is None else float(grad.norm())
        if not torch.isfinite(loss) or not np.isfinite(norm): raise ValueError('nonfinite supervised path')
        log.append(dict(step=step,value_loss=float(value_loss.detach()),latent_loss=float(latent_loss.detach()),organ_update_gradient_norm=norm))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();model.jepa.update_target_encoder()
    if log[-1]['organ_update_gradient_norm']<=0: raise ValueError('state correction remains disconnected')
    paths=[parent,checkpoint,Path(__file__),Path('whole_body_jepa/native_organ_state_residual.py'),
        Path('whole_body_jepa/native_organ_observations.py'),Path('whole_body_jepa/physiology_state.py')]
    output.write_text(json.dumps(dict(complete=True,role='one_source_fit_pair',steps=log,
        zero_start_exact=True,efficacy_evaluated=False,weights_saved=False,runtime_modified=False,
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(log))


if __name__=='__main__': main()
