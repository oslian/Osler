"""Read-only diagnostics of the availability encoder pilot; no model selection."""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.train_availability_encoder import batch, teacher_target, CONTEXT_PREFIXES
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import build_arrays
from whole_body_jepa.runtime import load_release
from whole_body_joint_jepa import _masked_latent_prediction_loss


def main():
    torch.set_num_threads(4)
    assets = load_release(Path('models'))
    idx = assets.checkpoint['variables'].index('glucose')
    scale = float(assets.scaler.scales[idx])
    median = float(assets.scaler.medians[idx])
    cap = .75 * scale
    report = dict(status='diagnostic_only',runtime_modified=False,
                  scale=scale,absolute_delta_cap=cap,results={},gradients=[])
    c = duckdb.connect()
    starts = c.execute("SELECT stay_id,intime FROM read_csv_auto('/Users/chunyouchang/Downloads/mimic-iv-3.1/icu/icustays.csv.gz')").df()
    c.close()
    raw = {db:pd.read_parquet(f'/private/tmp/availability_jepa_{db}_labs.parquet') for db in ('eicu','mimic')}
    origins = {'mimic':dict(zip('mimic:'+starts.stay_id.astype(str),starts.intime)),
               'eicu':{s:pd.Timestamp('2100-01-01') for s in raw['eicu'].stay_id.unique()}}
    for role in ('development','eicu_hospital','mimic'):
        frame = pd.read_parquet(f'/private/tmp/availability_glucose_{role}.parquet')
        valid = (frame.fresh & frame.truth.notna()).to_numpy()
        f = frame.loc[valid].reset_index(drop=True)
        delta = f.truth.to_numpy()-f.current.to_numpy()
        # This uses future truth solely as an unattainable best-case bound.
        oracle = f.current.to_numpy()+np.clip(delta,-cap,cap)
        result = dict(rows=len(f),fraction_truth_outside_cap=float(np.mean(np.abs(delta)>cap)),
                      cap_oracle_mae=patient_equal_mae(f,f.truth,oracle),
                      persistence_mae=patient_equal_mae(f,f.truth,f.current),arms={})
        db = 'mimic' if role == 'mimic' else 'eicu'
        for view in ('own','whole'):
            a,r = build_arrays(frame,raw[db],assets,origins[db],target_only=view=='own',truthful=True)
            r = r[valid]
            print(f'built {role} {view}',flush=True)
            for seed in (7,19,31):
                arm = view+'_e2e'
                saved = torch.load(f'/private/tmp/availability_encoder_weights_20260910/{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                model = assets.model
                model.load_state_dict(saved['state_dict'])
                model.eval()
                pred=[]
                with torch.no_grad():
                    for start in range(0,len(r),256):
                        out = model(*batch(a,r[start:start+256]))
                        pred.extend((out['value'][:,idx]*scale+median).numpy())
                pred=np.asarray(pred)
                change=pred-f.current.to_numpy()
                result['arms'][f'{arm}_{seed}']=dict(
                    mae=patient_equal_mae(f,f.truth,pred),
                    fraction_at_95pct_cap=float(np.mean(np.abs(change)>.95*cap)),
                    predicted_abs_delta_median=float(np.median(np.abs(change))),
                    delta_correlation=float(np.corrcoef(change,delta)[0,1]))
                if view == 'own':
                    # Same whole-trained weights; only input availability changes.
                    whole = torch.load(f'/private/tmp/availability_encoder_weights_20260910/whole_e2e_{seed}.pt',map_location='cpu',weights_only=False)
                    model.load_state_dict(whole['state_dict'])
                    masked=[]
                    with torch.no_grad():
                        for start in range(0,len(r),256):
                            out=model(*batch(a,r[start:start+256]))
                            masked.extend((out['value'][:,idx]*scale+median).numpy())
                    result['arms'][f'whole_weights_own_input_{seed}']=dict(
                        mae=patient_equal_mae(f,f.truth,np.asarray(masked)))
                    model.load_state_dict(saved['state_dict'])
                if role=='development' and seed==7:
                    params=[p for n,p in model.named_parameters() if n.startswith(CONTEXT_PREFIXES) and p.requires_grad]
                    rng=np.random.default_rng(730)
                    for repeat in range(3):
                        positions=rng.choice(len(r),64,replace=False)
                        inputs=batch(a,r[positions])
                        out=model(*inputs)
                        truth=torch.tensor((f.truth.to_numpy()[positions]-median)/scale,dtype=torch.float32)
                        target=teacher_target(model,inputs[1],truth,idx)
                        lv=F.smooth_l1_loss(out['value'][:,idx],truth)
                        lj=_masked_latent_prediction_loss(out['predicted_target_latent'][:,idx],target,torch.ones(64,dtype=torch.bool))
                        gv=torch.autograd.grad(lv,params,retain_graph=True,allow_unused=True)
                        gj=torch.autograd.grad(lj,params,allow_unused=True)
                        v=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gv,params)])
                        j=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g,p in zip(gj,params)])
                        report['gradients'].append(dict(view=view,repeat=repeat,value_norm=float(v.norm()),
                            jepa_norm=float(j.norm()),cosine=float(F.cosine_similarity(v,j,dim=0)),
                            value_loss=float(lv.detach()),jepa_loss=float(lj.detach())))
        report['results'][role]=result
    Path('artifacts/encoder_bottleneck_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
