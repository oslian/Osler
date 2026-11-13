"""Source-tune factorial input-path replay of the trained MAP support control."""
from copy import copy, deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.train_map_native_queries import predict
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.native_numeric_control import without_other_numbers
from whole_body_jepa.native_event_encoder import NativeEventFilter
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.runtime import load_release
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    output=Path('artifacts/map_control_paths_20260912.json')
    if output.exists():raise ValueError('fresh output required')
    q=pd.read_parquet(root/'queries.parquet').rename(columns={'uniquepid':'subject_id'})
    labels=dict(np.load(root/'labels.npz'));q['truth']=labels['future'][:,0]
    def shards():
        for p in sorted((root/'input_shards').glob('shard_*.npz')):
            yield dict(np.load(p)),dict(np.load(root/'native_query_shards'/p.name))
    raw=RaggedNativeBatch(shards(),len(q));assets=load_release(Path('models'))
    variables=assets.checkpoint['variables'];col=variables.index('map')
    median,scale=assets.scaler.medians[col],assets.scaler.scales[col]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    results=[];paths=[Path(__file__),Path('whole_body_jepa/native_numeric_control.py')]
    for seed in (7,19,31,43,59,71,83):
        parent=Path(f'artifacts/map_native_numeric_control_{seed}_20260912.json')
        for name,h in json.loads(parent.read_text())['bindings'].items():
            if sha256(Path(name))!=h:raise ValueError(f'changed dependency: {name}')
        paths.append(parent)
        roles=np.load(root/f'partition_{seed}.npz');eligible=labels['paired'][:,0]
        rows=np.flatnonzero(roles['tune']&eligible);fit=np.flatnonzero(roles['fit']&eligible)
        ck=torch.load(root/f'native_numeric_control_{seed}.pt',map_location='cpu',weights_only=False)
        allowed=np.array(ck['allowed_variables'],bool)
        def view(own):
            a=allowed.copy()
            if own:a[np.arange(len(a))!=col]=False
            s,ec,es=transformed_storage(raw,np.array(ck['numeric_center'],np.float32),
                np.array(ck['numeric_scale'],np.float32),a,assets.model.variable_module_ids.numpy(),fit,own=own)
            np.testing.assert_array_equal(ec,ck['event_center']);np.testing.assert_array_equal(es,ck['event_scale'])
            return s
        control=without_other_numbers(view(False),col);own=view(True)
        model=deepcopy(assets.model);expand_history_projection(model,6)
        model.belief_filter=NativeEventFilter(model.input_dim,model.belief_filter.hidden)
        model.belief_input_mode='observed_steps';model.value_delta_activation=torch.nn.Identity()
        model.load_state_dict(ck['state_dict']);model.eval()
        predictions={}
        for h in ('control','own'):
            for query in ('control','own'):
                storage=copy(control if h=='control' else own)
                storage.query=(control if query=='control' else own).query
                predictions[f'{h}_history_{query}_query']=predict(model,storage,rows,col,median,scale)
        original='control_history_control_query'
        saved=np.load(root/f'native_numeric_control_predictions_{seed}.npz')['control'][rows]
        np.testing.assert_array_equal(predictions[original],saved)
        frame=q.iloc[rows]
        comparison=paired_model_errors(frame,predictions,original,seed=20260912,repetitions=20000)
        results.append(dict(seed=seed,mae={k:patient_equal_mae(frame,frame.truth,p) for k,p in predictions.items()},
            comparisons=comparison))
        print('complete',seed,json.dumps(results[-1]),flush=True)
        del model,control,own,storage
    output.write_text(json.dumps(dict(complete=True,results=results,role='source_tune_only',
        original_predictions_exact=True,training_performed=False,runtime_modified=False,
        limitation='Off-training-distribution replay; content/support/cadence change together within history.',
        bindings={str(p):sha256(p) for p in paths}),indent=2)+'\n')


if __name__=='__main__':main()
