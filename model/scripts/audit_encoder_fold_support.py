"""Replay every source-fit patient's outer fold with fit-only input support."""
import json
from pathlib import Path
import torch
from scripts.encoder_fold_support import supported_example
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('data/research_recovery/broad_source_20260917')
    pp=Path('artifacts/native_reconstruction_encoder_fold_scaling_20260920.json')
    sp=root/'encoder_fold_scaling_20260920.pt'
    fp=root/'encoder_crossfit_folds_20260920.pt'
    out=Path('artifacts/native_reconstruction_encoder_fold_support_20260920.json')
    if out.exists():raise ValueError('Existing replay')
    parent=json.loads(pp.read_text())
    if sha256(sp)!=parent['private_sha256']:raise ValueError('Changed scaler')
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed scaler dependencies')
    scales=torch.load(sp,weights_only=False)['scalers']
    plan=torch.load(fp,weights_only=False)
    schema=torch.load(root/'joint_source_manifest.pt',weights_only=False)
    trained={schema['variables'].index(t) for t in schema['targets']}
    records=[dict(examples=0,accepted=0,unsupported_target=0,empty_history=0,
        trained_examples=0,trained_accepted=0,changed_examples=0,removed_observations=0,
        removed_event_rows=0) for _ in plan['folds']]
    for s in plan['shards']:
        if sha256(Path(s['path']))!=s['sha256']:raise ValueError('Changed source history')
        fold=next(i for i,r in enumerate(plan['folds']) if r[s['patient']]=='test')
        scaler=scales[fold];r=records[fold];n=len(scaler['counts'])
        for c,e in torch.load(s['path'],weights_only=False)['samples'].items():
            r['examples']+=1;r['trained_examples']+=int(c in trained)
            unsupported=torch.as_tensor(scaler['counts']==0)
            removed=int(e['tokens'][:,n:2*n][:,unsupported].sum())
            r['removed_observations']+=removed;r['changed_examples']+=int(removed>0)
            try:changed=supported_example(e,scaler)
            except ValueError as exc:
                if str(exc)=='Target has no source normalization support':r['unsupported_target']+=1
                elif str(exc)=='No source-supported independent history remains':r['empty_history']+=1
                else:raise
                continue
            r['accepted']+=1;r['trained_accepted']+=int(c in trained)
            r['removed_event_rows']+=len(e['tokens'])-len(changed['tokens'])
            if not torch.isfinite(changed['tokens']).all():raise ValueError('Nonfinite transformed stream')
    paths=[Path(__file__),Path('scripts/encoder_fold_support.py'),pp,sp,fp,root/'joint_source_manifest.pt']
    out.write_text(json.dumps(dict(complete=True,folds=records,bindings={str(p):sha256(p) for p in paths},
        runtime_modified=False,promotion_allowed=False,
        limitation='Input contract replay only. Unseen variables omitted from candidate inputs using fit-only support, not deleted from observed evidence or held-out denominator. Empty histories and unsupported targets abstain. No trained encoder or accuracy gain yet.'),indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)


if __name__=='__main__':main()
