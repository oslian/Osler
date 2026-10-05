"""Algebraic decomposition of fixed outer error gains; no model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def decompose(a,b):
    """Separate common-component weighting from within-component mean differences."""
    common=sorted(set(a.index)&set(b.index))
    if not common:raise ValueError('No common components')
    aa=a.loc[common];bb=b.loc[common]
    wa=aa['count'].to_numpy()/aa['count'].sum();wb=bb['count'].to_numpy()/bb['count'].sum()
    ma=aa['mean'].to_numpy();mb=bb['mean'].to_numpy()
    within=float(np.sum((wa+wb)*.5*(ma-mb)))
    composition=float(np.sum((wa-wb)*.5*(ma+mb)))
    difference=float(wa@ma-wb@mb)
    if not np.isclose(within+composition,difference,rtol=1e-12,atol=1e-12):raise ValueError('Decomposition identity failed')
    return dict(common_components=len(common),selected_common_patients=int(aa['count'].sum()),
        broad_common_patients=int(bb['count'].sum()),
        selected_common_fraction=float(aa['count'].sum()/a['count'].sum()),
        broad_common_fraction=float(bb['count'].sum()/b['count'].sum()),
        common_patient_mean_gap=difference,within_component_term=within,composition_term=composition,
        common_equal_component_gap=float((ma-mb).mean()))


def group_stats(frame):
    table=frame.groupby('component').delta.agg(['mean','count'])
    means=table['mean'].to_numpy();counts=table['count'].to_numpy()
    positive=np.maximum(means,0)
    loo=[float(np.delete(means,i).mean()) for i in range(len(means))] if len(means)>1 else []
    summary=dict(patients=len(frame),components=len(table),patient_mean=float(frame.delta.mean()),
        equal_component_mean=float(means.mean()),improving_components=int((means<0).sum()),
        worsening_components=int((means>0).sum()),components_under30=int((counts<30).sum()),
        positive_component_mass_from_under30=float(positive[counts<30].sum()/positive.sum()) if positive.sum()>0 else None,
        leave_one_component_out_min=min(loo) if loo else None,
        leave_one_component_out_max=max(loo) if loo else None)
    return table,summary


def main():
    rp=Path('artifacts/native_reconstruction_encoder_crossfit_evaluation_20260921.json')
    out=Path('artifacts/native_reconstruction_crossfit_group_shift_20260921.json')
    if out.exists():raise ValueError('Existing diagnostic')
    parent=json.loads(rp.read_text())
    for p,h in parent['bindings'].items():
        if sha256(Path(p))!=h:raise ValueError('Changed outer dependency')
    root=Path('data/research_recovery/broad_source_20260917/encoder_crossfit_outer_20260921')
    paths=[root/f'fold{i}.parquet' for i in range(5)]
    for i,p in enumerate(paths):
        if sha256(p)!=parent['prediction_hashes'][str(i)]:raise ValueError('Changed prediction file')
    frame=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    if frame.duplicated(['patient','target']).any():raise ValueError('Duplicate patient-target')
    good=frame.loc[frame.status=='ok'];results={}
    for target in parent['targets']:
        part=good.loc[good.target==target].copy();results[target]={}
        for control in ('process','population'):
            part['delta']=(part.real-part.truth).abs()-(part[control]-part.truth).abs()
            tables={};stats={}
            for group in ('selected','broad'):
                tables[group],stats[group]=group_stats(part.loc[part.group==group])
                old=parent['targets'][target][group]['comparisons'][control]
                if not np.isclose(stats[group]['patient_mean'],old['mean'],atol=1e-12,rtol=0):raise ValueError('Patient score changed')
                if not np.isclose(stats[group]['equal_component_mean'],old['site_mean'],atol=1e-12,rtol=0):raise ValueError('Component score changed')
            common=sorted(set(tables['selected'].index)&set(tables['broad'].index))
            rows=[]
            for number,c in enumerate(common):
                rows.append(dict(component_ordinal=number,selected_count=int(tables['selected'].loc[c,'count']),
                    broad_count=int(tables['broad'].loc[c,'count']),
                    selected_mean=float(tables['selected'].loc[c,'mean']),broad_mean=float(tables['broad'].loc[c,'mean'])))
            results[target][control]=dict(groups=stats,decomposition=decompose(tables['selected'],tables['broad']),common_component_rows=rows)
    summary={a:dict(targets=len(results),selected_worse_common_mean=sum(x[a]['decomposition']['common_patient_mean_gap']>0 for x in results.values()),
        within_component_term_positive=sum(x[a]['decomposition']['within_component_term']>0 for x in results.values()),
        composition_term_positive=sum(x[a]['decomposition']['composition_term']>0 for x in results.values())) for a in ('process','population')}
    out.write_text(json.dumps(dict(summary=summary,targets=results,bindings={str(p):sha256(p) for p in (Path(__file__),rp,*paths)},
        runtime_modified=False,promotion_allowed=False,
        limitation='Post-evaluation descriptive arithmetic. Component labels anonymized to per-target ordinal. Common-support decomposition conditions on observed groups, not matched physiology or causal effects. Under30 and leave-one-out summaries diagnose sensitivity only; no rows/sites excluded from original screen.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    for t in ('albumin','bun','hematocrit','total_protein','glucose','ptt','creatinine'):
        print(t,json.dumps(results[t]['process'],separators=(',',':')))


if __name__=='__main__':main()
