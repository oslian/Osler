"""Verify fit-only scalers and input reversibility for all crossfit patients."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.encoder_fold_scaling import patient_centers, fit_scaler, transform_example, inverse_value
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    fp = root/'encoder_crossfit_folds_20260920.pt'
    report = Path('artifacts/native_reconstruction_encoder_crossfit_preflight_20260920.json')
    out = Path('artifacts/native_reconstruction_encoder_fold_scaling_20260920.json')
    private = root/'encoder_fold_scaling_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing scaling evidence')
    parent = json.loads(report.read_text())
    if sha256(fp) != parent['private_sha256']:
        raise ValueError('Changed fold assignments')
    plan = torch.load(fp, weights_only=False)
    for p,h in plan['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed fold provenance')
    schema_path = root/'joint_source_manifest.pt'
    variables = torch.load(schema_path, weights_only=False)['variables']
    n=len(variables); people=[]; values=[]
    for i,s in enumerate(plan['shards']):
        if sha256(Path(s['path'])) != s['sha256']:
            raise ValueError('Changed source history')
        data = torch.load(s['path'], weights_only=False)
        if data['patient'] != s['patient']:
            raise ValueError('Patient mismatch')
        people.append(s['patient']); values.append(patient_centers(data['samples'], n))
        if (i+1)%2000 == 0:
            print('Summarized source patients',i+1,flush=True)
    centers=np.stack(values); scalers=[]; summaries=[]
    for roles in plan['folds']:
        train=np.array([roles[p]=='fit' for p in people])
        scaler=fit_scaler(centers,train); scalers.append(scaler)
        summaries.append(dict(fit_patients=int(train.sum()),
            variables={v:dict(count=int(scaler['counts'][c]),location=float(scaler['location'][c]),scale=float(scaler['scale'][c])) for c,v in enumerate(variables)},
            checked_examples=0,unsupported_target_examples=0,unsupported_input_examples=0,max_roundtrip_error=0.))
    # Each example is checked under its own outer-test fold without fitting to it.
    for i,s in enumerate(plan['shards']):
        fold=next(k for k,r in enumerate(plan['folds']) if r[s['patient']]=='test')
        scaler=scalers[fold]; summary=summaries[fold]
        data=torch.load(s['path'],weights_only=False)
        for e in data['samples'].values():
            try:
                changed=transform_example(e,scaler)
            except ValueError as exc:
                if str(exc)=='Target has no source normalization support':
                    summary['unsupported_target_examples']+=1;continue
                if str(exc)=='Observed variable has no source normalization support':
                    summary['unsupported_input_examples']+=1;continue
                raise
            torch.testing.assert_close(changed['tokens'][:,n:],e['tokens'][:,n:],rtol=0,atol=0)
            mask=e['tokens'][:,n:2*n].bool()
            restored=changed['tokens'][:,:n].double()*torch.tensor(scaler['scale'])+torch.tensor(scaler['location'])
            original=e['tokens'][:,:n].double()
            error=float((restored[mask]-original[mask]).abs().max()) if mask.any() else 0.
            if not torch.allclose(restored[mask],original[mask],rtol=2e-6,atol=2e-6):
                raise ValueError('Input affine roundtrip failed')
            if not np.isclose(inverse_value(changed['truth_normalized'],e['target_index'],scaler),e['truth_normalized'],rtol=1e-12,atol=1e-12):
                raise ValueError('Label roundtrip failed')
            summary['max_roundtrip_error']=max(summary['max_roundtrip_error'],error)
            summary['checked_examples']+=1
    bindings={str(p):sha256(p) for p in (Path(__file__),Path('scripts/encoder_fold_scaling.py'),fp,report,schema_path)}
    torch.save(dict(scalers=scalers,people=people,centers=centers,bindings=bindings),private)
    result=dict(complete=True,patients=len(people),folds=summaries,bindings=bindings,private_sha256=sha256(private),
        runtime_modified=False,promotion_allowed=False,
        limitation='Preprocessing audit only, no encoder retraining. Affine transformation removes dependence on inherited numerical units but does not certify historical cohort selection or pretrained weights. Source variable summaries prefer one anchor label per patient, otherwise within-patient median of stored observed history rows; not a raw event deduplication.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(fold=i,**{k:v for k,v in s.items() if k!='variables'}) for i,s in enumerate(summaries)],indent=2),flush=True)


if __name__=='__main__':
    main()
