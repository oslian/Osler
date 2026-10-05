"""Pre-anchor timing support only, without reading future outcome values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.potassium_treatment_context import changed_rate_brackets


def main():
    output=Path('artifacts/rate_measurement_brackets_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    private=Path('/private/tmp/insulin_potassium_pairs_20260911')
    prior=json.loads(Path('artifacts/insulin_potassium_partition_20260911.json').read_text())
    for name in ('queries.parquet','labels.npz'):
        if sha256(private/name)!=prior['binding'][str(private/name)]:raise ValueError('cache changed')
    labels=np.load(private/'labels.npz');paired=labels['paired'][:,0]
    q=pd.read_parquet(private/'queries.parquet').loc[paired].reset_index(drop=True)
    age=labels['measurement_age'][paired,0];sequence=labels['sequence'][paired]
    events=read_insulin_rates(Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/infusionDrug.csv.gz'))
    events['stay_id']='eicu:'+events.stay_id.astype(str)
    groups=dict(tuple(events.groupby('stay_id')))
    after_draw=np.zeros(len(q),bool);change_after_draw=after_draw.copy()
    bracket_after=after_draw.copy();bracket_straddles=after_draw.copy()
    record_lag=np.full(len(q),np.nan)
    for i,row in enumerate(q.itertuples()):
        ev=groups.get(row.stay_id,events.iloc[:0])
        ev=ev[(ev.offset>row.offset-360)&(ev.offset<=row.offset)].sort_values('offset')
        if not len(ev):continue
        draw=row.offset-age[i]*60
        after_draw[i]=bool((ev.offset>draw).any())
        record_lag[i]=(row.offset-ev.offset.iloc[-1])/60
        if not sequence[i]:continue
        if ev.label.nunique()!=1 or ev.groupby('offset').value.nunique().max()>1:raise ValueError('sequence mismatch')
        ev=ev.drop_duplicates('offset')
        changed=np.abs(np.diff(ev.value.to_numpy(float)))>1e-8
        change_after_draw[i]=bool(np.any(changed&(ev.offset.to_numpy()[1:]>draw)))
        bracket_after[i],bracket_straddles[i]=changed_rate_brackets(ev.offset,ev.value,draw)
    def stats(mask):
        n=int(mask.sum())
        return dict(rows=n,patients=int(q.loc[mask,'uniquepid'].nunique()),
                    current_age_median_hours=float(np.median(age[mask])),
                    latest_rate_age_median_hours=float(np.nanmedian(record_lag[mask])),
                    rate_record_after_current_draw=int((mask&after_draw).sum()),
                    rate_change_after_current_draw=int((mask&change_after_draw).sum()),
                    bracket_after_rows=int((mask&bracket_after).sum()),
                    bracket_after_patients=int(q.loc[mask&bracket_after,'uniquepid'].nunique()),
                    straddle_only_rows=int((mask&bracket_straddles&~bracket_after).sum()),
                    straddle_only_patients=int(q.loc[mask&bracket_straddles&~bracket_after,'uniquepid'].nunique()))
    report=dict(complete=True,status='timing_support_not_efficacy',future_values_read=False,runtime_modified=False,
                limitations='record time is not verified action time or knowledge time; temporal precedence does not establish mediation',
                all=stats(np.ones(len(q),bool)),sequence=stats(sequence),seeds=[])
    for entry in prior['seeds']:
        p=private/f"partition_{entry['seed']}.npz"
        if sha256(p)!=entry['partition_sha256']:raise ValueError('partition changed')
        roles=np.load(p)
        report['seeds'].append(dict(seed=entry['seed'],roles={k:stats(roles[k]&sequence) for k in ('fit','evaluation','hospital')}))
    report['binding']={str(p):sha256(p) for p in [Path(__file__),private/'labels.npz',private/'queries.parquet']}
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
