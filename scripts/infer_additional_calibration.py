"""Calibration-role inference with frozen matching models, no outcome tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import transform_example, inverse_value
from scripts.train_encoder_crossfit import fresh_encoder
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def verify_roles(frame, scaling, roles, fold, components):
    people=list(scaling['people']);variables=list(scaling['variables'])
    pi={p:i for i,p in enumerate(people)};vi={v:i for i,v in enumerate(variables)}
    centers=np.asarray(scaling['centers'])
    expected=np.isfinite(centers)&np.array([roles[p]=='calibration' for p in people])[:,None]
    actual=np.zeros_like(expected)
    ix=np.array([pi[p] for p in frame.patient]);iy=np.array([vi[t] for t in frame.target])
    actual[ix,iy]=True
    if frame.duplicated(['patient','target']).any() or not np.array_equal(expected,actual):
        raise ValueError('Calibration identities mismatch')
    if not (frame.fold==fold).all() or not np.allclose(frame.truth,centers[ix,iy],rtol=1e-6,atol=1e-6):
        raise ValueError('Calibration labels/role mismatch')
    if not np.array_equal(frame.component.to_numpy(),frame.patient.map(components).to_numpy()):
        raise ValueError('Component mismatch')
    if not np.isfinite(frame[['truth','fit_scale','real','process','population']].to_numpy()).all() or (frame.fit_scale<=0).any():
        raise ValueError('Invalid predictions')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    f=parser.parse_args().fold
    root=Path('data/research_recovery/additional_development_domains_20260921')
    folder=root/'calibration_20260924';folder.mkdir(exist_ok=True)
    dest=folder/f'fold{f}.parquet';meta=dest.with_suffix('.json')
    if dest.exists() or meta.exists():raise ValueError('Existing calibration prediction')
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt'
    pp=root/'provenance.parquet';cp=Path('models/whole_body_joint_jepa.pt')
    protocol=Path('RECONSTRUCTION_ADDITIONAL_CALIBRATION_PROTOCOL_20260924.md')
    paths=[Path(__file__),fp,sp,pp,cp,protocol,Path('scripts/encoder_fold_scaling.py'),Path('scripts/train_encoder_crossfit.py')]
    roles=torch.load(fp,weights_only=False)['folds'][f]
    scaling=torch.load(sp,weights_only=False);scaler=scaling['scalers'][f]
    variables=scaling['variables'];schema=torch.load(cp,weights_only=False)
    if schema['variables']!=variables:raise ValueError('Schema mismatch')
    links=pd.read_parquet(pp)[['uniquepid','component']].drop_duplicates()
    if links.uniquepid.duplicated().any():raise ValueError('Ambiguous component')
    components=dict(links.itertuples(index=False,name=None))
    ck={};checked={}
    outer_meta=root/f'outer_20260924/fold{f}.json';paths.append(outer_meta)
    bound=json.loads(outer_meta.read_text())['bindings']
    for name,digest in bound.items():
        if sha256(Path(name))!=digest:raise ValueError('Changed outer dependency')
    for arm in ('real','process'):
        p=root/f'training_20260923/fold{f}_{arm}.pt';paths.append(p)
        if sha256(p)!=bound[str(p)]:raise ValueError('Changed model')
        ck[arm]=torch.load(p,weights_only=False)
        if ck[arm]['epoch']!=8 or ck[arm]['fold']!=f or ck[arm]['arm']!=arm:raise ValueError('Wrong model')
        for name,digest in ck[arm]['bindings'].items():
            if name not in checked:checked[name]=sha256(Path(name))
            if checked[name]!=digest:raise ValueError('Changed training dependency')
    reused=False
    if f==2:
        for arm in ('real','process'):
            other=torch.load(root/f'training_20260923/fold0_{arm}.pt',weights_only=False)
            if ck[arm]['medians']!=other['medians'] or any(not torch.equal(v,other['state_dict'][k]) for k,v in ck[arm]['state_dict'].items()):
                raise ValueError('Nonidentical reused function')
        for k,v in scaler.items():np.testing.assert_array_equal(v,scaling['scalers'][0][k])
        old=root/'outer_20260924/fold0.parquet';om=old.with_suffix('.json');saved=json.loads(om.read_text())
        if sha256(old)!=saved['prediction_sha256']:raise ValueError('Changed reusable predictions')
        for name,digest in saved['bindings'].items():
            if sha256(Path(name))!=digest:raise ValueError('Changed reusable dependency')
        frame=pd.read_parquet(old);frame['fold']=f;paths.extend((old,om));reused=True
    else:
        cache_path=root/'full_histories_20260922.sqlite'
        report_path=Path('artifacts/native_reconstruction_additional_full_cache_20260922.json');paths.append(report_path)
        if sha256(cache_path)!=json.loads(report_path.read_text())['private_sha256']:raise ValueError('Changed histories')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        models={}
        for arm in ck:
            model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(schema),variables,('map',))).eval()
            model.load_state_dict(ck[arm]['state_dict'],strict=True);models[arm]=model
        cache=ReconstructionCache(cache_path);rows=[]
        try:
            keys=[(p,int(c)) for p,c in cache.db.execute('SELECT patient,target FROM examples ORDER BY patient,target') if roles[p]=='calibration']
            with torch.inference_mode():
                for i,(p,c) in enumerate(keys,1):
                    original=cache.get(p,c);e=transform_example(original,scaler)
                    row=dict(patient=p,target=variables[c],fold=f,component=components[p],truth=original['truth_normalized'],fit_scale=float(scaler['scale'][c]),population=float(inverse_value(ck['real']['medians'][c],c,scaler)))
                    for arm,model in models.items():
                        x=e['tokens']
                        if arm=='process':x=x.clone();x[:,:len(variables)]=0
                        row[arm]=float(inverse_value(float(model(x,e['event_hours'],e['anchor_hour'],variables[c])[0]),c,scaler))
                    rows.append(row)
                    if i%10000==0:print('Calibration fold',f,i,'of',len(keys),flush=True)
        finally:cache.close()
        frame=pd.DataFrame(rows)
    verify_roles(frame,scaling,roles,f,components)
    tmp=dest.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(dest)
    result=dict(fold=f,role='calibration',examples=len(frame),patients=int(frame.patient.nunique()),
        verified_equivalent_reuse=reused,prediction_sha256=sha256(dest),
        bindings={str(p):sha256(p) for p in paths},runtime_modified=False,promotion_allowed=False)
    meta.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'}),flush=True)


if __name__=='__main__':main()
