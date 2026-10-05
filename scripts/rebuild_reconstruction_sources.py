"""Rebuild lost event files; restore only byte-identical historical artifacts."""
import json
from pathlib import Path
import shutil
import duckdb
import pandas as pd

from scripts.audit_availability_jepa import extract
from scripts.audit_eicu_insulin_sequence_inventory import read_insulin_rates
from extractors.eicu_sepsis_transition_extract import PLAUSIBLE
from whole_body_jepa.bedside_events import nursing_events
from whole_body_jepa.native_history_schedule import native_history_schedule
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    stage = Path('data/research_recovery/rebuilt_sources_20260916')
    stage.mkdir(exist_ok=False)
    source = Path('artifacts/native_reconstruction_expanded_20260913.json')
    bindings = json.loads(source.read_text())['bindings']
    query_path = root/'map_native_anchor_support/queries.parquet'
    if sha256(query_path) != bindings[str(query_path)]:
        raise ValueError('changed surviving cohort')
    q = pd.read_parquet(query_path)
    wanted = set(q.stay_id)
    assets = load_release(Path('models'))
    print('rebuilding labs', flush=True)
    labs, _, _ = extract('eicu', wanted, assets.checkpoint['variables'], output=stage/'body_labs.parquet')
    labs = labs.rename(columns={'available_at':'event_time'})
    labs['source_kind'] = 'lab_available'
    data = Path('/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0')
    rates = read_insulin_rates(data/'infusionDrug.csv.gz')
    rates['stay_id'] = 'eicu:'+rates.stay_id.astype(str)
    rates = rates.loc[rates.stay_id.isin(wanted)].copy()
    rates['event_time'] = pd.Timestamp('2100-01-01')+pd.to_timedelta(rates.offset,unit='m')
    rates['source_kind'] = 'insulin_record'
    rates['variable'] = 'insulin_rate'
    rates['measured_at'] = pd.NaT
    rates['unit'] = 'units/hour'
    cols = ['stay_id','source_kind','source_id','event_time','measured_at','value','variable','unit']
    combined = pd.concat([labs[cols],rates[cols]],ignore_index=True)
    combined['source_id'] = combined.source_id.astype(str)
    # Canonical event output is independent of query schedules; do not rebuild
    # discarded forecasts or invent the unavailable historical rate anchors.
    events, _ = native_history_schedule(combined, q.iloc[:0])
    events.to_parquet(stage/'native_events.parquet',index=False)
    print('rebuilding bedside', flush=True)
    con = duckdb.connect()
    con.execute('SET threads=1')
    con.execute("SET memory_limit='2GB'")
    con.execute('SET preserve_insertion_order=false')
    stays = pd.DataFrame({'stay':sorted(s.removeprefix('eicu:') for s in wanted)})
    con.register('selected_stays',stays)
    con.read_csv(str(data/'nurseCharting.csv.gz'),header=True,all_varchar=True).create_view('raw_nursing')
    raw = con.sql("SELECT n.* FROM raw_nursing n SEMI JOIN selected_stays s ON n.patientunitstayid=s.stay WHERE try_cast(nursingchartvalue AS DOUBLE) IS NOT NULL AND nursingchartcelltypecat='Vital Signs'").df()
    con.close()
    bedside = nursing_events(raw)
    bedside['quality_valid'] = False
    for variable,part in bedside.groupby('variable'):
        lower, upper = PLAUSIBLE[variable]
        bedside.loc[part.index,'quality_valid'] = part.value.between(lower,upper)
    bedside.to_parquet(stage/'qualified_bedside_events.parquet',index=False)
    results = {}
    for name in ('native_events.parquet','qualified_bedside_events.parquet'):
        path = stage/name
        expected = bindings[str(root/name)]
        actual = sha256(path)
        equal = actual == expected
        results[name] = dict(expected_sha256=expected,actual_sha256=actual,byte_identical=equal)
        if equal:
            if (root/name).exists():
                raise ValueError('do not overwrite existing event file')
            shutil.copy2(path,root/name)
    out = Path('artifacts/reconstruction_sources_rebuilt_20260916.json')
    if out.exists():
        raise ValueError('existing recovery report')
    out.write_text(json.dumps(dict(complete=True,source_recovery_complete=all(v['byte_identical'] for v in results.values()),
        files=results,runtime_modified=False,training_performed=False,
        bindings={str(p):sha256(p) for p in (Path(__file__),source,query_path)}),indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
