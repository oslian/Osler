"""Audit saved anion-gap identity and exact-time panel consistency, no correction."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def panels(raw):
    names = ['anion_gap','sodium','chloride','bicarbonate','potassium']
    f = raw.loc[raw.variable.isin(names)].copy()
    keys = ['stay_id','measured_at','variable']
    groups = f.groupby(keys).value.agg(['nunique','first'])
    # Never select arbitrarily among disagreeing results at the same time.
    v = groups['first'].where(groups['nunique'] == 1)
    wide = v.unstack('variable').reindex(columns=names).dropna()
    wide['without_k_error'] = wide.anion_gap-(wide.sodium-wide.chloride-wide.bicarbonate)
    wide['with_k_error'] = wide.without_k_error-wide.potassium
    return wide.reset_index()


def summarize(f):
    return dict(rows=len(f),stays=int(f.stay_id.nunique()),
        without_k_median_abs=float(f.without_k_error.abs().median()) if len(f) else None,
        with_k_median_abs=float(f.with_k_error.abs().median()) if len(f) else None,
        within_one_without=int((f.without_k_error.abs()<=1).sum()),
        within_one_with=int((f.with_k_error.abs()<=1).sum()),
        neither_within_one=int(((f.without_k_error.abs()>1)&(f.with_k_error.abs()>1)).sum()))


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    parent = Path('artifacts/map_native_anchor_inputs_20260912.json')
    output = Path('artifacts/map_anion_contract_20260912.json')
    if output.exists():
        raise ValueError('fresh output required')
    ref = json.loads(parent.read_text())
    path = root/'body_labs.parquet'
    assert sha256(path) == ref['bindings'][str(path)]
    labs = pd.read_parquet(path)
    saved = labs.loc[labs.variable.eq('anion_gap')].copy()
    c = duckdb.connect()
    c.execute('SET threads=4')
    wanted = saved[['source_id']].drop_duplicates()
    c.register('wanted',wanted)
    raw = c.execute('''SELECT cast(l.labid as varchar) source_id,l.labname,
        cast(l.labresult as double) AS "value",l.labmeasurenamesystem system_unit,
        l.labmeasurenameinterface interface_unit,
        timestamp '2100-01-01'+l.labresultoffset*interval '1 minute' measured_at,
        timestamp '2100-01-01'+greatest(l.labresultoffset,l.labresultrevisedoffset)*interval '1 minute' available_at
        FROM read_csv_auto(?) l JOIN wanted w ON cast(l.labid as varchar)=w.source_id''',
        ['/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/lab.csv.gz']).df()
    c.close()
    joined = saved.merge(raw,on='source_id',validate='one_to_one',suffixes=('_saved','_raw'))
    assert len(joined) == len(saved)
    for key in ('value','measured_at','available_at'):
        np.testing.assert_array_equal(joined[key+'_saved'],joined[key+'_raw'])
    p = panels(labs)
    q = pd.read_parquet(root/'map_native_anchor_support/queries.parquet')
    roles = dict(np.load(root/'map_native_anchor_support/partition_7.npz'))
    result = {role:summarize(p.loc[p.stay_id.isin(q.loc[mask,'stay_id'])]) for role,mask in roles.items()}
    output.write_text(json.dumps(dict(complete=True,runtime_modified=False,training_performed=False,
        raw_matched_rows=len(joined),source_names=raw.labname.value_counts().to_dict(),
        units=raw.groupby(['system_unit','interface_unit'],dropna=False).size().reset_index(name='rows').fillna('').to_dict('records'),
        panel_policy='Exact measurement timestamp, complete five-analyte panel, conflicting values excluded; descriptive only.',
        roles=result,bindings={str(p):sha256(p) for p in (parent,path,Path(__file__))}),indent=2)+'\n')
    print('Exact raw matches',len(joined),'names',raw.labname.value_counts().to_dict())
    print(result)


if __name__ == '__main__':
    main()
