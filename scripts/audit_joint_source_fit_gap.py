"""Exact selected-checkpoint fit/selection replay, all source targets retained."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    out=Path('artifacts/native_reconstruction_joint_fit_gap_20260918.json')
    saved=Path('data/research_recovery/broad_source_20260917/joint_fit_gap.pt')
    if out.exists()or saved.exists():raise ValueError('Existing fit diagnostic')
    tp=Path('artifacts/native_reconstruction_joint_source_training_20260917.json')
    tr=json.loads(tp.read_text())
    for p,h in tr['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Training changed')
    ip=Path('data/research_recovery/broad_source_20260917/joint_source_manifest.pt')
    cp=Path('data/research_recovery/broad_source_20260917/joint_training/real.pt')
    ck=torch.load(cp,weights_only=False);index=torch.load(ip,weights_only=False)
    if ck['bindings']!=tr['bindings']:raise ValueError('Checkpoint mismatch')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assets=deepcopy(load_release(Path('models')));variables=assets.checkpoint['variables']
    expand_history_projection(assets.model,6);assets.model.belief_input_mode='observed_steps'
    model=NativeBodyReconstruction(NativeOrganStateResidual(assets.model,variables,('map',))).eval()
    model.load_state_dict(ck['best_state_dict']);rows=[]
    with torch.inference_mode():
        for i,item in enumerate(index['shards']):
            p=Path(item['path'])
            if sha256(p)!=item['sha256']:raise ValueError('Shard changed')
            for c,e in torch.load(p,weights_only=False)['samples'].items():
                t=variables[c]
                if t not in index['targets']:continue
                pred=float(model(input_tokens(e,'real',len(variables)),e['event_hours'],e['anchor_hour'],t)[0])
                rows.append(dict(patient=item['patient'],group=item['group'],role=item['role'],target=t,
                    truth=e['truth_normalized'],prediction=pred,population=ck['medians'][c]))
            if (i+1)%200==0:print('Replayed',i+1,flush=True)
    torch.save(rows,saved);result={}
    chosen=tr['arms']['real']['log'][ck['selected_epoch']-1]['group_target_scores']
    for g in ('selected','broad'):
        result[g]={}
        for t in index['targets']:
            per={}
            for role in ('fit','encoder_tune'):
                rr=[r for r in rows if r['group']==g and r['target']==t and r['role']==role]
                y=np.array([r['truth']for r in rr]);pred=np.array([r['prediction']for r in rr]);p=np.array([r['population']for r in rr])
                mae=float(abs(pred-y).mean());base=float(abs(p-y).mean())
                per[role]=dict(patients=len(rr),mae=mae,population_mae=base,increment=mae-base)
                if role=='encoder_tune'and not np.isclose(mae,chosen[g][t],atol=1e-7,rtol=0):raise ValueError('Selection score mismatch')
            result[g][t]=per
    out.write_text(json.dumps(dict(groups=result,bindings={str(p):sha256(p)for p in (Path(__file__),tp,ip,cp,saved)},
        limitation='Fit versus selected-tune diagnostic, not independent generalization proof. No new fitting or promotion.',runtime_modified=False),indent=2)+'\n')
    for g,ts in result.items():
        print(g,'fit helps but tune harms',[t for t,r in ts.items()if r['fit']['increment']<0<r['encoder_tune']['increment']])
        print(g,'fit harms',[t for t,r in ts.items()if r['fit']['increment']>=0])


if __name__=='__main__':main()
