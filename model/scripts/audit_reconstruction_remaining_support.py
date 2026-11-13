"""Known-at-anchor support strata of frozen strict-site reconstruction."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import mask_native_target


def main():
    output = Path('artifacts/native_reconstruction_remaining_support_20260917.json')
    if output.exists():
        raise ValueError('existing audit')
    ep = Path('artifacts/native_reconstruction_site_evaluation_20260917.json')
    report = json.loads(ep.read_text())
    for p,h in report['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('changed predictions')
    saved = Path('data/research_recovery/site_reconstruction_training_20260917/evaluation_predictions.pt')
    rows = torch.load(saved, weights_only=False)
    lookup = {(r['subject_id'],r['target']):r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('duplicate prediction')
    ip = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support/expanded_reconstruction_20260913/index.pt')
    index = torch.load(ip, weights_only=False)
    variables = index['variables']; n=len(variables)
    records = {t:[] for t in report['targets']}
    seen = set()
    for shard in index['shards']:
        if not any((shard['patient'],t) in lookup for t in records):
            continue
        p=durable(shard['path'])
        if sha256(p)!=shard['sha256']:
            raise ValueError('changed input shard')
        sample=torch.load(p,weights_only=False)
        for c,e in sample['samples'].items():
            key=(sample['patient'],variables[c])
            if key not in lookup:
                continue
            row=lookup[key]
            if row['truth']!=e['truth_normalized'] or key in seen:
                raise ValueError('truth or key mismatch')
            seen.add(key)
            clean,times=mask_native_target(e['tokens'],e['event_hours'],variables,variables[c])
            observed=clean[:,n:2*n].bool()
            since_available=(e['anchor_hour']-times).double()
            if (since_available<0).any():
                raise ValueError('future event')
            # Event lag is NOT specimen age; keep this distinction explicit.
            fresh=observed[since_available<=1].any(0) if (since_available<=1).any() else torch.zeros(n,dtype=torch.bool)
            records[variables[c]].append(dict(lag=float(since_available[-1]),
                recent_variables=int(fresh.sum()), total_variables=int(observed.any(0).sum()),
                real_error=abs(row['real']-row['truth']), process_error=abs(row['process']-row['truth'])))
    if seen!=set(lookup):
        raise ValueError('missing scored example')
    results={}
    for t,rs in records.items():
        strata={}
        for name,predicate in [('within_1h',lambda r:r['lag']<=1),('1_to_6h',lambda r:1<r['lag']<=6),('over_6h',lambda r:r['lag']>6)]:
            part=[r for r in rs if predicate(r)]
            strata[name]=dict(patients=len(part),mean_real_minus_process=None if not part else float(np.mean([r['real_error']-r['process_error'] for r in part])))
        for arm in ('real','process'):
            if not np.isclose(np.mean([r[arm+'_error'] for r in rs]),report['targets'][t]['normalized_mae'][arm],atol=1e-10,rtol=0):
                raise ValueError('score mismatch')
        results[t]=dict(patients=len(rs),median_latest_remaining_event_lag=float(np.median([r['lag'] for r in rs])),
            median_recent_variable_count=float(np.median([r['recent_variables'] for r in rs])),strata=strata)
    output.write_text(json.dumps(dict(complete=True,targets=results,unsupported=report['unsupported'],
        bindings={str(p):sha256(p) for p in (Path(__file__),ep,saved,ip,Path('whole_body_jepa/native_body_reconstruction.py'))},
        runtime_modified=False,promotion_allowed=False,
        limitation='Fixed known-at-anchor availability-lag bins, descriptive reused evaluation only. Event availability lag is not specimen age. No routing fitted, no exclusions or significance.'),indent=2)+'\n')
    for t in ('lipase','triglycerides','creatinine','bilirubin','glucose'):
        print(t,results[t])


if __name__=='__main__':main()
