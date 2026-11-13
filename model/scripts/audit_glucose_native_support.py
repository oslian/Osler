"""Source-defined joint support strata, not an outcome-selected router."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    reference = Path('artifacts/rate_glucose_jepa_20260911.json')
    output = Path('artifacts/glucose_native_support_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    ref = json.loads(reference.read_text())
    for name, digest in ref['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('changed input')
    valid = np.load(root/'labels.npz')['paired'][:,0]
    labels = np.load(root/'multitarget_labels.npz')
    k = list(labels['variables']).index('glucose')
    frame = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True)
    frame = frame.rename(columns={'uniquepid':'subject_id'})
    frame['truth'] = labels['future'][:,k]
    native = np.load(root/'native_tokens.npz')
    ptr, ids = native['query_ptr'], native['history_indices']
    rate = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        values = native['event_values'][ids[ptr[i]:ptr[i+1]],2]
        finite = values[np.isfinite(values)]
        if len(finite): rate[i] = finite[-1]
    x = np.column_stack([labels['current'][:,k],labels['measurement_age'][:,k],rate])
    results = []
    for entry in ref['seeds']:
        seed = entry['seed']; partitions = np.load(root/f'partition_{seed}.npz')
        roles = {r:partitions[r][valid]&labels['paired'][:,k] for r in ('fit','evaluation','hospital')}
        fit = roles['fit']
        # Each fitting patient contributes one median state to the bin boundaries.
        source = pd.DataFrame(x[fit],columns=['current','age','rate'])
        source['subject_id'] = frame.loc[fit,'subject_id'].to_numpy()
        medians = source.groupby('subject_id').median()
        cuts = [np.unique(medians[c].dropna().quantile([.25,.5,.75]).to_numpy()) for c in medians.columns]
        bins = np.column_stack([np.where(np.isfinite(x[:,j]),np.searchsorted(cuts[j],x[:,j],side='right'),-1) for j in range(3)])
        support = {}
        for i in np.flatnonzero(fit):
            support.setdefault(tuple(bins[i]),set()).add(frame.subject_id.iloc[i])
        count = np.array([len(support.get(tuple(b),set())) for b in bins])
        points = np.load(root/f'glucose_native_predictions_{seed}.npz')
        item = dict(seed=seed, cuts=[c.tolist() for c in cuts], roles={})
        for role in ('evaluation','hospital'):
            groups = {}
            for name, condition in [('all',np.ones(len(frame),bool)),('under30',count<30),('atleast30',count>=30)]:
                mask = roles[role]&condition; f=frame.loc[mask]
                groups[name] = dict(rows=int(mask.sum()),patients=int(f.subject_id.nunique()),
                    comparison=paired(f,points['real'][mask],points['own'][mask],f'glucose-support:{seed}:{role}:{name}',2000))
            item['roles'][role]=groups
        results.append(item)
    output.write_text(json.dumps(dict(complete=True,status='support_diagnostic',runtime_modified=False,
        cuts_fit_on_source_only=True, min_fit_patients=30,
        binding={str(p):sha256(p) for p in (reference,Path(__file__))}, seeds=results),indent=2)+'\n')
    for role in ('evaluation','hospital'):
        for group in ('under30','atleast30'):
            z=[s['roles'][role][group] for s in results]
            print(role,group,'patients',[v['patients'] for v in z],
                  'paired delta',[round(v['comparison']['mean'],4) if v['comparison']['mean'] is not None else None for v in z],
                  'gain/harm',sum(v['comparison']['ci95'] is not None and v['comparison']['ci95'][1]<0 for v in z),
                  sum(v['comparison']['ci95'] is not None and v['comparison']['ci95'][0]>0 for v in z))


if __name__=='__main__': main()
