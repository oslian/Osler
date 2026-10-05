"""All-target outer predictions with source-only scalers and frozen epoch8."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from scripts.train_encoder_crossfit import fresh_encoder
from scripts.encoder_fold_support import supported_example
from scripts.encoder_fold_scaling import inverse_value
from scripts.preflight_encoder_crossfit import components
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.multisource_intervals import sha256


def mean_ci(values,seed):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    rng=np.random.default_rng(seed);means=[]
    for _ in range(50):
        means.extend(values[rng.integers(0,len(values),(100,len(values)))].mean(axis=1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def summarize(frame,seed):
    if frame.patient.duplicated().any():raise ValueError('Patient-target appears twice')
    good=frame.loc[frame.status=='ok']
    result=dict(total=len(frame),eligible=len(good),abstentions=frame.loc[frame.status!='ok','status'].value_counts().to_dict())
    if not len(good):return dict(result,descriptive_screen_pass=False)
    errors={a:(good[a]-good.truth).abs().to_numpy() for a in ('real','process','population')}
    result['mae']={a:dict(patient=float(v.mean()),site=float(pd.Series(v).groupby(good.site.to_numpy()).mean().mean()),
        component=float(pd.Series(v).groupby(good.component.to_numpy()).mean().mean())) for a,v in errors.items()}
    checks={}
    for j,arm in enumerate(('process','population')):
        delta=errors['real']-errors[arm]
        ev=evidence(delta,good.component.to_numpy())
        grouped=pd.Series(delta).groupby(good.component.to_numpy()).mean().to_numpy()
        checks[arm]=dict(ev,patient_mean_ci95=mean_ci(delta,seed+j),
            component_mean_ci95=mean_ci(grouped,seed+100+j))
    result['comparisons']=checks
    result['descriptive_screen_pass']=confirmed(checks,len(good))
    return result


def main():
    root=Path('data/research_recovery/broad_source_20260917');folder=root/'encoder_crossfit_training_20260920'
    out=Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    if out.exists():raise ValueError('Existing outer result')
    fp=root/'encoder_crossfit_folds_20260920.pt';sp=root/'encoder_fold_scaling_20260920.pt'
    mp=root/'site_balance_manifest_20260920.pt';jp=root/'joint_source_manifest.pt'
    raw=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/patient.csv.gz')
    cp=Path('models/whole_body_joint_jepa.pt')
    dependencies=[Path(__file__),fp,sp,mp,jp,raw,cp,Path('scripts/encoder_fold_support.py'),
        Path('scripts/encoder_fold_scaling.py'),Path('scripts/train_encoder_crossfit.py'),
        Path('scripts/audit_confirmed_reconstruction_heads.py'),Path('scripts/preflight_encoder_crossfit.py'),
        Path('RECONSTRUCTION_ENCODER_CROSSFIT_EVALUATION_PROTOCOL_20260921.md')]
    # All training must finish before any held-out inference, including resumed work.
    for f in range(5):
        for arm in ('real','process'):
            p=folder/f'fold{f}_{arm}.pt'
            rp=Path(f'artifacts/native_reconstruction_encoder_crossfit_fold{f}_{arm}_20260920.json')
            report=json.loads(rp.read_text())
            if report['epoch']!=8 or sha256(p)!=report['private_sha256']:raise ValueError('Training incomplete/changed')
            for path,h in report['bindings'].items():
                if sha256(Path(path))!=h:raise ValueError('Changed training dependency')
            dependencies.extend((p,rp))
    plan=torch.load(fp,weights_only=False);scalers=torch.load(sp,weights_only=False)['scalers']
    manifest=torch.load(mp,weights_only=False);schema=torch.load(jp,weights_only=False)
    variables,targets=schema['variables'],schema['targets'];cols=[variables.index(t) for t in targets]
    source_schema=torch.load(cp,map_location='cpu',weights_only=False)
    patients=pd.read_csv(raw,usecols=['uniquepid','hospitalid'],dtype=str)
    people={s['patient'] for s in plan['shards']}
    sites=patients.loc[patients.uniquepid.isin(people)].groupby('uniquepid').hospitalid.agg(set).to_dict()
    comp=components(sites)
    bindings={str(p):sha256(p) for p in dependencies}
    private=root/'encoder_crossfit_outer_20260921';private.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    all_frames=[];prediction_hashes={}
    for f,roles in enumerate(plan['folds']):
        path=private/f'fold{f}.parquet';meta=private/f'fold{f}.json'
        if path.exists() and meta.exists():
            saved=json.loads(meta.read_text())
            if saved['bindings']!=bindings or sha256(path)!=saved['sha256']:raise ValueError('Changed prediction cache')
            all_frames.append(pd.read_parquet(path));prediction_hashes[str(f)]=saved['sha256'];continue
        models={};checkpoints={}
        for arm in ('real','process'):
            ck=torch.load(folder/f'fold{f}_{arm}.pt',weights_only=False);checkpoints[arm]=ck
            model=NativeBodyReconstruction(NativeOrganStateResidual(fresh_encoder(source_schema),variables,('map',))).eval()
            model.load_state_dict(ck['state_dict'],strict=True);models[arm]=model
        rows=[];scaler=scalers[f]
        with torch.inference_mode():
            for s in plan['shards']:
                person=s['patient']
                if roles[person]!='test':continue
                p=Path(s['path'])
                if sha256(p)!=s['sha256']:raise ValueError('Changed input shard')
                for c,e in torch.load(p,weights_only=False)['samples'].items():
                    if c not in cols:continue
                    row=dict(patient=person,target=variables[c],fold=f,group=s['group'],
                        site=str(manifest['sample_sites'][s['group'],'fit',person,c]),component=comp[person],
                        truth=e['truth_normalized'],status='ok',real=np.nan,process=np.nan,
                        population=float(inverse_value(checkpoints['real']['medians'][c],c,scaler)))
                    try:changed=supported_example(e,scaler)
                    except ValueError as exc:
                        if str(exc) not in ('No source-supported independent history remains','Target has no source normalization support'):raise
                        row['status']=str(exc);rows.append(row);continue
                    for arm,model in models.items():
                        x=changed['tokens']
                        if arm=='process':x=x.clone();x[:,:len(variables)]=0
                        value=float(model(x,changed['event_hours'],changed['anchor_hour'],variables[c])[0])
                        row[arm]=float(inverse_value(value,c,scaler))
                        if not np.isfinite(row[arm]):raise ValueError('Nonfinite prediction')
                    rows.append(row)
        frame=pd.DataFrame(rows);tmp=path.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(path)
        h=sha256(path);meta.write_text(json.dumps(dict(bindings=bindings,sha256=h),indent=2)+'\n')
        prediction_hashes[str(f)]=h;all_frames.append(frame)
        print('Outer inference completed fold',f,'pairs',len(frame),flush=True)
    frame=pd.concat(all_frames,ignore_index=True)
    if frame.duplicated(['patient','target']).any():raise ValueError('Outer patient-target duplication')
    if len(frame)!=247611:raise ValueError('Outer target coverage changed')
    results={}
    for i,target in enumerate(targets):
        part=frame.loc[frame.target==target];results[target]={}
        for j,scope in enumerate(('all','selected','broad')):
            subset=part if scope=='all' else part.loc[part.group==scope]
            results[target][scope]=summarize(subset,20260921+1000*i+10*j)
        results[target]['both_groups_screen_pass']=all(results[target][g]['descriptive_screen_pass'] for g in ('selected','broad'))
    summary=dict(targets=len(targets),pairs=len(frame),eligible_pairs=int((frame.status=='ok').sum()),
        patient_count=int(frame.patient.nunique()),
        descriptive_passes={g:[t for t,r in results.items() if r[g]['descriptive_screen_pass']] for g in ('all','selected','broad')},
        both_group_passes=[t for t,r in results.items() if r['both_groups_screen_pass']],
        macro={g:{a:float(np.mean([r[g]['mae'][a]['patient'] for r in results.values() if 'mae' in r[g]])) for a in ('real','process','population')} for g in ('all','selected','broad')})
    out.write_text(json.dumps(dict(summary=summary,targets=results,bindings=bindings,prediction_hashes=prediction_hashes,
        runtime_modified=False,promotion_allowed=False,
        limitation='Current-value masked reconstruction only. Outer data excluded from this fold encoder optimization and scaling; source development reused in project design. Overlapping training and identical fold0/1 fitted functions make bootstrap/t-tests descriptive, not complete algorithm or external promotion inference.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
