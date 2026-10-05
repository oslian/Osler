"""Measure direct mixing retention on source-fit histories, not total memory."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_numeric_control import without_other_numbers
from whole_body_jepa.native_carry_audit import post_target_direct_carry
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/map_native_carry_20260913.json')
    if output.exists():raise ValueError('fresh output required')
    q=pd.read_parquet(root/'queries.parquet');eligible=np.load(root/'labels.npz')['paired'][:,0]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q));assets=load_release(Path('models'))
    col=assets.checkpoint['variables'].index('map');n=len(assets.checkpoint['variables'])
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results=[];paths=[Path(__file__),Path('whole_body_jepa/native_carry_audit.py')]
    for seed in (7,19,31,43,59,71,83):
        fit=np.load(root/f'partition_{seed}.npz')['fit']&eligible
        pools=[np.asarray(g.index) for _,g in q.loc[fit].groupby('uniquepid',sort=True)]
        rng=np.random.default_rng(seed+20260913)
        selected=np.array([rng.choice(pools[i]) for i in rng.choice(len(pools),96,replace=False)])
        for arm in ('own','full','control'):
            parent=Path(f'artifacts/map_native_{"numeric_control" if arm=="control" else "query_training"}_{seed}_20260912.json')
            for name,h in json.loads(parent.read_text())['bindings'].items():
                if sha256(Path(name))!=h:raise ValueError('changed dependency')
            paths.append(parent)
            ckpath=root/(f'native_numeric_control_{seed}.pt' if arm=='control' else f'native_query_{arm}_{seed}.pt')
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            storage,ec,es=transformed_storage(raw,np.array(ck['numeric_center'],np.float32),
                np.array(ck['numeric_scale'],np.float32),np.array(ck['allowed_variables'],bool),
                assets.model.variable_module_ids.numpy(),np.flatnonzero(fit),own=arm=='own')
            np.testing.assert_array_equal(ec,ck['event_center']);np.testing.assert_array_equal(es,ck['event_scale'])
            if arm=='control':storage=without_other_numbers(storage,col)
            model=deepcopy(assets.model);expand_history_projection(model,6)
            model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
            model.belief_input_mode='observed_steps';model.value_delta_activation=torch.nn.Identity()
            model.load_state_dict(ck['state_dict']);model.eval()
            captured=[]
            hook=model.belief_filter.update_gate.register_forward_hook(lambda m,i,o:captured.append(o.detach().sigmoid()))
            x=storage.batch(selected)
            with torch.inference_mode():pred=model(*x)['value'][:,col].numpy()*scale+median
            hook.remove()
            saved=np.load(root/(f'native_numeric_control_predictions_{seed}.npz' if arm=='control' else f'native_query_predictions_{seed}.npz'))[arm][selected]
            np.testing.assert_allclose(pred,saved,rtol=1e-6,atol=1e-4)
            gates=torch.stack(captured,dim=1)
            active=x[0][:,:,n:2*n].bool().any(-1)|x[0][:,:,-2].bool()
            factor,count,valid=post_target_direct_carry(gates,active,x[0][:,:,n+col].bool())
            supported=factor[valid].mean(dim=1).numpy()
            results.append(dict(seed=seed,arm=arm,queries=len(selected),with_target=int(valid.sum()),
                extra_updates_median=float(count[valid].float().median()),
                direct_carry_mean=float(supported.mean()),direct_carry_median=float(np.median(supported))))
            del model,storage,x,captured,gates
        print('complete',seed,flush=True)
    output.write_text(json.dumps(dict(complete=True,results=results,role='source_fit_only',
        predictions_reproduced=True,training_performed=False,runtime_modified=False,
        limitation='Direct (1-gate) path only: excludes decay and innovation-mediated retention; not total Jacobian or information.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
