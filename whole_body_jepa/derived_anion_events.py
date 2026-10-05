"""Timestamp-matched derived gap; never reinterpret reported anion-gap events."""
import json
import pandas as pd
from whole_body_jepa.raw_lab_replay import canonical_events

VARIABLE = 'derived_anion_gap_without_k_same_time'
COMPONENTS = ('sodium','chloride','bicarbonate')
COLUMNS = ['stay_id','source_id','variable','measured_at','available_at','value',
           'unit','quality_valid','component_source_ids']


def derive_events(raw):
    events = canonical_events(raw.loc[raw.variable.isin(COMPONENTS)])
    output = []
    for (stay,measured),bundle in events.groupby(['stay_id','measured_at'],sort=False):
        versions = {v:{} for v in COMPONENTS}
        for available,arrival in bundle.groupby('available_at',sort=True):
            for e in arrival.itertuples():
                versions[e.variable][str(e.source_id)] = e
            values = {}
            good = True
            for var,source in versions.items():
                rows = list(source.values())
                distinct = {float(e.value) for e in rows}
                valid = bool(rows) and len(distinct)==1 and all(
                    str(e.unit).strip().lower() in ('mmol/l','meq/l') and
                    not pd.isna(getattr(e,'quality_valid',True)) and bool(getattr(e,'quality_valid',True))
                    for e in rows)
                good &= valid
                if valid:
                    values[var] = next(iter(distinct))
            # Invalid versions block stale derived values; zero is only a masked sentinel.
            value = values['sodium']-values['chloride']-values['bicarbonate'] if good else 0.
            output.append(dict(stay_id=stay,source_id=f'derived_no_k:{pd.Timestamp(measured).value}',
                variable=VARIABLE,measured_at=measured,available_at=available,value=value,
                unit='mmol/L',quality_valid=bool(good),
                component_source_ids=json.dumps({k:sorted(v) for k,v in versions.items()},sort_keys=True)))
    return pd.DataFrame(output,columns=COLUMNS)
