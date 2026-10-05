"""Matched-size correction fitting, same held-out query people, source only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.try_robust_source_correction import robust_predict
from whole_body_jepa.multisource_intervals import sha256


def priority(patient, purpose):
    return hashlib.sha256(f'20260925:{purpose}:{patient}'.encode()).hexdigest()


def query_people(frame):
    people = frame[['patient','component']].drop_duplicates()
    if people.patient.duplicated().any():
        raise ValueError('Patient in multiple components')
    selected = set()
    for _, part in people.groupby('component'):
        ordered = sorted(part.patient, key=lambda p: priority(p, 'query'))
        selected.update(ordered[:max(1, len(ordered)//5)])
    return selected


def matched_fits(frame, queries, component):
    train = frame.loc[~frame.patient.isin(queries)]
    unseen = train.loc[train.component != component]
    order = sorted(range(len(train)), key=lambda i: priority(train.patient.iloc[i], 'fit'))
    seen = train.iloc[order[:len(unseen)]]
    query = frame.loc[frame.patient.isin(queries) & (frame.component == component)]
    if len(seen) != len(unseen) or set(query.patient)&(set(seen.patient)|set(unseen.patient)):
        raise ValueError('Invalid paired fit')
    return seen, unseen, query


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    rp = Path('artifacts/native_reconstruction_robust_source_20260925.json')
    sp = Path('artifacts/native_reconstruction_variable_snapshot_20260925.json')
    protocol = Path('RECONSTRUCTION_SEEN_DOMAIN_PROTOCOL_20260925.md')
    out = Path('artifacts/native_reconstruction_seen_domain_20260925.json')
    if out.exists():
        raise ValueError('Existing domain experiment')
    old = json.loads(rp.read_text()); snapshot_meta = json.loads(sp.read_text())
    snapshot = root/'variable_snapshot_20260925.parquet'
    for p, h in old['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError('Changed dependency')
    if sha256(snapshot) != snapshot_meta['private_sha256']:
        raise ValueError('Changed snapshot')
    frames = {f:pd.read_parquet(root/f'calibration_20260924/fold{f}.parquet') for f in range(5)}
    queries = {f:query_people(p) for f,p in frames.items()}
    result = {}
    with threadpool_limits(limits=4):
        for i, target in enumerate(old['targets']):
            snap = pd.read_parquet(snapshot, filters=[('target','=',target)])
            result[target] = {}
            for f in range(5):
                source = frames[f].loc[frames[f].target == target].merge(snap, on=['patient','target'], how='left', validate='one_to_one', indicator=True)
                if not (source._merge == 'both').all():
                    raise ValueError('Missing snapshot')
                rows = []
                for component in sorted(source.loc[source.patient.isin(queries[f]), 'component'].unique()):
                    seen, unseen, query = matched_fits(source, queries[f], component)
                    entry = dict(component=str(component), query_patients=len(query), matched_fit_patients=len(seen),
                        seen_same_component_fit_patients=int((seen.component == component).sum()))
                    if entry['seen_same_component_fit_patients'] == 0:
                        rows.append(dict(entry,status='no_seen_exposure'))
                        continue
                    ps, si = robust_predict(seen, query, snapshot_meta['variables'])
                    pu, ui = robust_predict(unseen, query, snapshot_meta['variables'])
                    entry.update(seen_fit=si, unseen_fit=ui)
                    if si['status'] != 'fitted' or ui['status'] != 'fitted':
                        rows.append(dict(entry,status='incomplete_fit'))
                        continue
                    truth = query.truth.to_numpy()
                    mae = {'unchanged':float(np.mean(abs(query.real.to_numpy()-truth)))}
                    for kind,preds in (('seen',ps),('unseen',pu)):
                        for a,p in preds.items():
                            mae[f'{kind}_{a}'] = float(np.mean(abs(p-truth)))
                    rows.append(dict(entry,status='paired',mae=mae))
                complete = bool(rows) and all(r['status']=='paired' for r in rows)
                supported = complete and sum(r['query_patients']>=30 for r in rows)>=3
                record = dict(complete=complete,supported=supported,components=rows)
                if complete:
                    record['component_equal_mae'] = {a:float(np.mean([r['mae'][a] for r in rows])) for a in rows[0]['mae']}
                result[target][str(f)] = record
            print('Matched domain target', i+1, 'of', len(old['targets']), flush=True)
    assessed = [r for z in result.values() for r in z.values() if r['supported']]
    matrix = dict(seen_better_unseen_better=0,seen_better_unseen_not_better=0,seen_not_better_unseen_better=0,neither_better=0)
    for r in assessed:
        m = r['component_equal_mae']; a = m['seen_numeric_context']<m['unchanged']; b = m['unseen_numeric_context']<m['unchanged']
        key = ('seen_better_unseen_better' if b else 'seen_better_unseen_not_better') if a else ('seen_not_better_unseen_better' if b else 'neither_better')
        matrix[key] += 1
    summary = dict(target_folds=235,supported_complete=len(assessed),numeric_vs_unchanged=matrix,
        seen_numeric_better_than_unseen=sum(r['component_equal_mae']['seen_numeric_context']<r['component_equal_mae']['unseen_numeric_context'] for r in assessed))
    out.write_text(json.dumps(dict(summary=summary,targets=result,
        bindings={str(p):sha256(p) for p in (rp,sp,snapshot,protocol,Path(__file__),Path('scripts/try_robust_source_correction.py'))},
        runtime_modified=False,promotion_allowed=False,outer_evaluation_performed=False,
        limitation='Fixed source-only paired query/matched row-count diagnosis. Training component composition, weights and input support still differ. Conditional means, not causal attribution or independent promotion tests.'),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
