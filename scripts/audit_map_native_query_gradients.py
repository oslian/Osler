"""Source-only value/latent gradient probes for the new native MAP task."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scripts.train_availability_encoder import configure, teacher_target
from scripts.audit_magnesium_auxiliary_gradient import geometry
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    parent=Path('artifacts/map_native_query_training_summary_20260912.json')
    output=Path('artifacts/map_native_query_gradients_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    for name,h in json.loads(parent.read_text())['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed dependency')
    q=pd.read_parquet(root/'queries.parquet');labels=dict(np.load(root/'labels.npz'))
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q))
    assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('map')
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    truth=(labels['future'][:,0]-median)/scale
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    probes=[]
    for seed in (7,19,31,43,59,71,83):
        fit=np.load(root/f'partition_{seed}.npz')['fit']&labels['paired'][:,0]
        pools=[np.asarray(g.index) for _,g in q.loc[fit].groupby('uniquepid',sort=True)]
        rng=np.random.default_rng(seed+20260912)
        batches=[np.array([rng.choice(pools[i]) for i in rng.integers(len(pools),size=32)]) for _ in range(3)]
        predictions=dict(np.load(root/f'native_query_predictions_{seed}.npz'))
        for arm in ('own','full'):
            ck=torch.load(root/f'native_query_{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            view,ec,es=transformed_storage(raw,np.array(ck['numeric_center'],np.float32),
                np.array(ck['numeric_scale'],np.float32),np.array(ck['allowed_variables'],bool),
                assets.model.variable_module_ids.numpy(),np.flatnonzero(fit),own=arm=='own')
            np.testing.assert_array_equal(ec,ck['event_center']);np.testing.assert_array_equal(es,ck['event_scale'])
            model=deepcopy(assets.model);expand_history_projection(model,6)
            model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            model.belief_input_mode='observed_steps';model.value_delta_activation=torch.nn.Identity()
            model.load_state_dict(ck['state_dict']);configure(model,True)
            named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and
                   n.split('.')[0] in ('belief_filter','variable_mixer','module_mixer','state_encoder')]
            for bi,chosen in enumerate(batches):
                x=view.batch(chosen);result=model(*x)
                np.testing.assert_allclose(result['value'][:,col].detach().numpy()*scale+median,
                    predictions[arm][chosen],rtol=1e-6,atol=1e-4)
                y=torch.tensor(truth[chosen],dtype=torch.float32)
                value=F.smooth_l1_loss(result['value'][:,col],y)
                latent=_masked_latent_prediction_loss(result['predicted_target_latent'][:,col],
                    teacher_target(model,x[1],y,col),torch.ones(len(y),dtype=torch.bool))
                a=torch.autograd.grad(value,[p for _,p in named],retain_graph=True,allow_unused=True)
                b=torch.autograd.grad(latent,[p for _,p in named],allow_unused=True)
                groups={'all':geometry(a,b)}
                for group in sorted(set(n.split('.')[0] for n,_ in named)):
                    ids=[i for i,(n,_) in enumerate(named) if n.startswith(group+'.')]
                    groups[group]=geometry([a[i] for i in ids],[b[i] for i in ids])
                probes.append(dict(seed=seed,arm=arm,batch=bi,value_loss=float(value.detach()),
                    latent_loss=float(latent.detach()),groups=groups))
            del model,view,ck,result,x,a,b
        print('complete',seed,flush=True)
    result=dict(complete=True,probes=probes,role='source_fit_only',runtime_modified=False,
        training_performed=False,predictions_reproduced=True,
        scope='selected local raw gradient geometry; not Adam trajectory or loss ablation',
        bindings={str(p):sha256(p) for p in (parent,Path(__file__),Path('scripts/audit_magnesium_auxiliary_gradient.py'))})
    output.write_text(json.dumps(result,indent=2)+'\n')
    for arm in ('own','full'):
        gs=[p['groups']['all'] for p in probes if p['arm']==arm]
        print(arm,'opposing',sum(g['cosine']<0 for g in gs),'median_ratio',np.median([g['retention_to_primary'] for g in gs]))


if __name__=='__main__':main()
