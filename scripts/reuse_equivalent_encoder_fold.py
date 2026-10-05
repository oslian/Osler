"""Reuse one control only after exact inputs and an independent rerun agree."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from whole_body_jepa.multisource_intervals import sha256


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left,right):
            raise ValueError('Different tensor')
    elif isinstance(left,np.ndarray):
        if not isinstance(right,np.ndarray) or left.dtype != right.dtype or not np.array_equal(left,right):
            raise ValueError('Different array')
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('Different mapping')
        for key in left:assert_same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('Different sequence')
        for a,b in zip(left,right):assert_same(a,b)
    elif type(left)!=type(right) or left!=right:
        raise ValueError('Different scalar')


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    folder=root/'encoder_crossfit_training_20260920'
    fp=root/'encoder_crossfit_folds_20260920.pt';sp=root/'encoder_fold_scaling_20260920.pt'
    out=Path('artifacts/native_reconstruction_encoder_fold_equivalence_20260920.json')
    dest=folder/'fold1_process.pt'
    report_dest=Path('artifacts/native_reconstruction_encoder_crossfit_fold1_process_20260920.json')
    if any(p.exists() for p in (out,dest,report_dest)):raise ValueError('Existing equivalence result')
    plan=torch.load(fp,weights_only=False);scalers=torch.load(sp,weights_only=False)['scalers']
    fit0=[s for s in plan['shards'] if plan['folds'][0][s['patient']]=='fit']
    fit1=[s for s in plan['shards'] if plan['folds'][1][s['patient']]=='fit']
    assert_same(fit0,fit1);assert_same(scalers[0],scalers[1])
    paths=[folder/f'fold{f}_{a}.pt' for f,a in ((0,'real'),(1,'real'),(0,'process'))]
    a,b,control=[torch.load(p,weights_only=False) for p in paths]
    reports=[Path(f'artifacts/native_reconstruction_encoder_crossfit_fold{f}_{arm}_20260920.json') for f,arm in ((0,'real'),(1,'real'),(0,'process'))]
    for path,rp,ck in zip(paths,reports,(a,b,control)):
        if ck['epoch']!=8 or sha256(path)!=json.loads(rp.read_text())['private_sha256']:
            raise ValueError('Incomplete or changed checkpoint')
        for p,h in ck['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed dependency')
    for field in ('state_dict','optimizer','rng','torch_rng','log','medians','variables','targets','bindings'):
        assert_same(a[field],b[field])
    for field in ('rng','torch_rng','medians','variables','targets','bindings'):
        assert_same(a[field],control[field])
    bindings={str(p):sha256(p) for p in [Path(__file__),fp,sp,*paths,*reports]}
    cloned=deepcopy(control);cloned['fold']=1
    cloned['equivalent_reuse']=dict(source_fold=0,bindings=bindings,
        additional_optimizer_steps_executed=0,independent_rerun_real_arm_bitwise_equal=True)
    torch.save(cloned,dest)
    parent=json.loads(reports[2].read_text())
    parent.update(fold=1,private_sha256=sha256(dest),equivalent_reuse=cloned['equivalent_reuse'],
        limitation='Fold1 control reused from fold0 after exact ordered fit-shard/scaler equality and independently rerun real-arm model/optimizer/RNG/log equality. Zero additional process-arm optimization; not an independent model replicate. Outer test patients remain different and unscored.')
    report_dest.write_text(json.dumps(parent,indent=2)+'\n')
    result=dict(complete=True,fit_patients=len(fit0),scalers_equal=True,
        independently_rerun_real_weights_optimizer_rng_logs_equal=True,
        source_fold=0,destination_fold=1,arm='process',additional_optimizer_steps_executed=0,
        available_fixed_arm_epochs=32,executed_fixed_arm_epochs=24,
        bindings=bindings,reused_checkpoint_sha256=sha256(dest),runtime_modified=False,promotion_allowed=False,
        limitation='Exact computational reuse, not new training evidence or independent replication. Outer folds1-4 remain part of the unchanged full evaluation scope; folds2-4 training not yet complete.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('bindings','limitation','reused_checkpoint_sha256')})


if __name__=='__main__':main()
