"""Certify selection0 predictions from identical epoch10 outer2 computation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.certify_monitored_continuation_fold2 import same
from whole_body_jepa.multisource_intervals import sha256


def certify_queries(source, donor_base, donor_new, source_roles, donor_roles):
    selected={p for p,r in source_roles.items() if r=='selection'}
    tested={p for p,r in donor_roles.items() if r=='test'}
    if selected!=tested or not set(source.patient)<=selected:
        raise ValueError('Different query populations')
    if {p for p,r in source_roles.items() if r=='fit'}!={p for p,r in donor_roles.items() if r=='fit'}:
        raise ValueError('Different fit populations')
    for frame,fold in ((source,0),(donor_base,2),(donor_new,2)):
        if frame.empty or frame.duplicated(['patient','target']).any() or not frame.fold.eq(fold).all():
            raise ValueError('Invalid identity grid')
    cols=[c for c in source.columns if c!='fold']
    pd.testing.assert_frame_equal(source[cols],donor_base[cols])
    pd.testing.assert_frame_equal(source[['patient','target']],donor_new[['patient','target']])
    if not np.isfinite(donor_new[['real10','process10']]).all().all():
        raise ValueError('Invalid continued predictions')
    result=donor_new.copy()
    result['fold']=0
    return result


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    directory=root/'continuation_selection_20260930'
    dest=directory/'fold0.parquet'
    if dest.exists() or dest.with_suffix('.json').exists():
        raise ValueError('Existing source inference')
    proof_path=Path('artifacts/native_reconstruction_monitored_fold2_reuse_20260929.json')
    proof=json.loads(proof_path.read_text())
    if not proof['fit_identity_equal'] or not proof['scalers_equal']:
        raise ValueError('Uncertified equivalent fitting')
    donor_path=root/'continuation_outer_20260929/fold2.parquet'
    source_path=root/'coverage_selection_20260926/fold0.parquet'
    base_path=root/'coverage_outer_20260927/fold2.parquet'
    reports={p:json.loads(p.with_suffix('.json').read_text()) for p in (donor_path,source_path,base_path)}
    for path,meta in reports.items():
        if sha256(path)!=meta['prediction_sha256']:
            raise ValueError('Changed prediction payload')
    donor=reports[donor_path]
    if (donor['fold'],donor['source_fold'],donor['role'])!=(2,0,'test'):
        raise ValueError('Unexpected donor model/role')
    if reports[source_path]['fold']!=0:
        raise ValueError('Unexpected source fold')
    checked={}
    for bindings in (proof['bindings'],donor['bindings']):
        for name,digest in bindings.items():
            if name not in checked:checked[name]=sha256(Path(name))
            if checked[name]!=digest:raise ValueError('Changed certified dependency: '+name)
    fp=root/'partitions_20260922.pt';sp=root/'scaling_20260922.pt'
    plan=torch.load(fp,weights_only=False)['folds']
    scaling=torch.load(sp,weights_only=False)
    if not same(scaling['scalers'][0],scaling['scalers'][2]):
        raise ValueError('Different fit-only transforms')
    for arm in ('real','process'):
        cp=root/f'monitored_continuation_20260929/fold0_{arm}_epoch10.pt'
        if sha256(cp)!=proof['arms'][arm]['10']['checkpoint_sha256']:
            raise ValueError('Changed continued weights')
    source,base,fresh=[pd.read_parquet(p) for p in (source_path,base_path,donor_path)]
    result=certify_queries(source,base,fresh,plan[0],plan[2])
    # Full eligibility check prevents equal-but-incomplete query grids being certified.
    people=scaling['people'];variables=scaling['variables']
    expected={(p,variables[c]) for i,p in enumerate(people) if plan[0][p]=='selection'
              for c in np.flatnonzero(np.isfinite(scaling['centers'][i]))}
    if set(zip(result.patient,result.target))!=expected:
        raise ValueError('Incomplete source eligibility grid')
    directory.mkdir(exist_ok=True)
    temp=dest.with_suffix('.tmp');result.to_parquet(temp,index=False);temp.replace(dest)
    paths=[Path(__file__),Path('scripts/certify_monitored_continuation_fold2.py'),
           proof_path,fp,sp,*reports.keys(),*[p.with_suffix('.json') for p in reports]]
    meta=dict(fold=0,source_fold=0,role='selection',epoch=10,
        examples=len(result),patients=int(result.patient.nunique()),targets=int(result.target.nunique()),
        prediction_sha256=sha256(dest),bindings={str(p):sha256(p) for p in paths},
        reused_from=str(donor_path),equivalent_inference_reuse=True,independent_replication=False,
        candidate_only=True,promotion_allowed=False,runtime_modified=False,
        limitation='Same fitted model, scaler, native cache, complete query keys and baseline values as outer2. Metadata role is relative to fold0. No new independent evidence, fitting or selection.')
    dest.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k not in ('bindings','reused_from')},indent=2))


if __name__=='__main__':main()
