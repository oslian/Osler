"""Finish saved evaluation after source cleanup interrupted report serialization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.change_evaluation import paired_model_errors
from whole_body_jepa.reconstruction_partitions import patient_roles


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911/map_native_anchor_support')
    private=root/'reconstruction_replication_v2_83_20260914'
    source=Path('artifacts/native_reconstruction_replication_v2_83_epoch8_20260914.json')
    out=Path('artifacts/native_reconstruction_replication_v2_83_evaluation_20260914.json')
    if out.exists():raise ValueError('existing report')
    trained=json.loads(source.read_text())
    for name,h in trained['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed training dependency')
    origin=json.loads(Path('artifacts/native_reconstruction_expanded_20260913.json').read_text())
    for name,h in origin['bindings'].items():
        if sha256(Path(name))!=h:raise ValueError('changed restored source')
    rows=torch.load(private/'heldout_predictions.pt',weights_only=False)
    roles=patient_roles(pd.read_parquet(root/'queries.parquet'),dict(np.load(root/'partition_83.npz')))
    index=torch.load(root/'expanded_reconstruction_20260913/index.pt',weights_only=False)
    expected={r:{} for r in rows}
    fit={t:[] for t in trained['source_targets']}
    variables=index['variables']
    for shard in index['shards']:
        if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('changed shard')
        s=torch.load(shard['path'],weights_only=False)
        if s['patient']!=shard['patient']:raise ValueError('patient mismatch')
        role=roles[s['patient']]
        for c,e in s['samples'].items():
            t=variables[c]
            if t not in fit:continue
            if role=='fit':fit[t].append(e['truth_normalized'])
            elif role in expected:expected[role][(s['patient'],t)]=e['truth_normalized']
    medians={t:float(np.median(v)) for t,v in fit.items()}
    for role,records in rows.items():
        keys=[(v['subject_id'],v['target']) for v in records]
        if len(keys)!=len(set(keys)) or set(keys)!=set(expected[role]):raise ValueError('saved support mismatch')
        for v in records:
            if v['truth']!=expected[role][(v['subject_id'],v['target'])]:raise ValueError('truth mismatch')
            if v['population']!=medians[v['target']]:raise ValueError('wrong fit median')
    for arm in ('real','process','placebo'):
        ck=torch.load(private/f'{arm}.pt',weights_only=False)
        if ck['epoch']!=8 or ck['bindings']!=trained['bindings']:raise ValueError('wrong checkpoint')
    result={}
    for role,records in rows.items():
        result[role]={}
        for target,part in pd.DataFrame(records).groupby('target'):
            predictions={a:part[a].to_numpy() for a in ('real','process','placebo','population')}
            result[role][target]=dict(patients=len(part),normalized_mae={a:float(np.mean(abs(v-part.truth.to_numpy()))) for a,v in predictions.items()},
                comparisons={ref:paired_model_errors(part,predictions,ref,seed=20260914) for ref in ('population','process','placebo')})
    deps=[Path(__file__),source,Path('scripts/evaluate_reconstruction_replication_v2.py'),
          root/'queries.parquet',root/'partition_83.npz',root/'expanded_reconstruction_20260913/index.pt',
          private/'heldout_predictions.pt',Path('whole_body_jepa/change_evaluation.py')]
    deps += [private/f'{a}.pt' for a in ('real','process','placebo')]
    out.write_text(json.dumps(dict(complete=True,seed=83,roles=result,source_unsupported=trained['source_unsupported'],
        runtime_modified=False,promotion_allowed=False,recovered_after_report_serialization_failure=True,
        limitation='Saved completed predictions; original source hashes restored, support/truth/fit medians checked. Nominal repeated-split comparisons.',
        bindings={str(p):sha256(p) for p in deps}),indent=2)+'\n')
    print('complete',flush=True)


if __name__=='__main__':main()
