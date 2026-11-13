"""Private numeric nursing cache and aggregate label/time inventory."""
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from whole_body_jepa.multisource_intervals import sha256


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    source=Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0/nurseCharting.csv.gz')
    private=root/'numeric_nurse_charting.parquet'
    output=Path('artifacts/native_bedside_inventory_20260911.json')
    if private.exists() or output.exists():raise ValueError('fresh outputs required')
    q=pd.read_parquet(root/'queries.parquet').loc[np.load(root/'labels.npz')['paired'][:,0]]
    stays=pd.DataFrame({'stay':q.stay_id.astype(str).str.removeprefix('eicu:').unique()})
    con=duckdb.connect();con.execute("SET threads=1");con.execute("SET memory_limit='2GB'")
    con.execute("SET preserve_insertion_order=false")
    con.register('selected_stays',stays)
    con.read_csv(str(source),header=True,all_varchar=True).create_view('raw_nursing')
    con.sql("SELECT n.* FROM raw_nursing n SEMI JOIN selected_stays s ON n.patientunitstayid=s.stay WHERE try_cast(nursingchartvalue AS DOUBLE) IS NOT NULL").write_parquet(str(private))
    con.read_parquet(str(private)).create_view('cached')
    rows=con.execute("""SELECT nursingchartcelltypecat AS category,
        nursingchartcelltypevallabel AS label,nursingchartcelltypevalname AS name,
        count(*) AS rows,count(DISTINCT patientunitstayid) AS stays,
        count(*) FILTER(WHERE try_cast(nursingchartentryoffset AS DOUBLE)>=try_cast(nursingchartoffset AS DOUBLE)) AS ordered_times,
        median(try_cast(nursingchartentryoffset AS DOUBLE)-try_cast(nursingchartoffset AS DOUBLE)) AS median_delay_minutes
        FROM cached GROUP BY ALL ORDER BY rows DESC""").fetchdf()
    report=dict(complete=True,runtime_modified=False,training_performed=False,
        selected_stays=len(stays),numeric_rows=int(rows.rows.sum()),
        label_groups=rows.replace({np.nan:None}).to_dict('records'),
        scope='raw numeric inventory, no unit mapping or legal query-window support established',
        binding={str(p):sha256(p) for p in [private,root/'queries.parquet',root/'labels.npz',Path(__file__)]},
        source_size_bytes=source.stat().st_size)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(selected_stays=len(stays),numeric_rows=report['numeric_rows'],top_labels=report['label_groups'][:25]),indent=2))


if __name__=='__main__':main()
