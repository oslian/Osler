"""Verify cached magnesium against raw eICU values and both unit fields."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    cache=root/'body_labs.parquet';parent=Path('artifacts/rate_multitarget_labels_20260911.json')
    ref=json.loads(parent.read_text());assert sha256(cache)==ref['binding'][str(cache)]
    out=Path('artifacts/magnesium_raw_units_20260911.json')
    if out.exists():raise ValueError('fresh output required')
    wanted=pd.read_parquet(cache);wanted=wanted.loc[wanted.variable.eq('magnesium')].copy()
    wanted['native_id']=wanted.source_id.astype('int64')
    raw_path=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/lab.csv.gz')
    c=duckdb.connect();c.execute('SET threads=4');c.register('wanted',wanted)
    raw=c.execute('''SELECT w.value cached_value,l.labresult raw_value,l.labname,
        l.labmeasurenamesystem system_unit,l.labmeasurenameinterface interface_unit
        FROM read_csv_auto(?) l JOIN wanted w ON l.labid=w.native_id''',[str(raw_path)]).df();c.close()
    assert len(raw)==len(wanted)
    np.testing.assert_allclose(raw.cached_value,raw.raw_value,rtol=0,atol=1e-12)
    inventory=raw.fillna('').groupby(['labname','system_unit','interface_unit']).size().reset_index(name='rows').to_dict('records')
    report=dict(complete=True,rows=len(raw),raw_values_exact=True,unit_inventory=inventory,
        runtime_modified=False,no_rows_removed=True,
        binding={str(p):sha256(p) for p in (parent,cache,Path(__file__))},
        raw_file_size=raw_path.stat().st_size,raw_file_mtime_ns=raw_path.stat().st_mtime_ns)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(inventory,indent=2))


if __name__=='__main__':main()
