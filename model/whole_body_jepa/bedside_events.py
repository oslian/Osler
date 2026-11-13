"""Time-qualified eICU nursing vitals; units retain their label provenance."""
import numpy as np
import pandas as pd

from whole_body_jepa.raw_lab_replay import canonical_events


# Celsius only: Fahrenheit sibling rows must not double-update temperature.
LABELS = (
    ('Heart Rate', 'Heart Rate', 'heart_rate', 'bpm'),
    ('O2 Saturation', 'O2 Saturation', 'o2sat', '%'),
    ('Respiratory Rate', 'Respiratory Rate', 'respiratory_rate', 'breaths/min'),
    ('Non-Invasive BP', 'Non-Invasive BP Mean', 'map', 'mmHg'),
    ('Invasive BP', 'Invasive BP Mean', 'map', 'mmHg'),
    ('Temperature', 'Temperature (C)', 'temperature', 'C'),
)


def nursing_events(raw):
    """Reject invalid times, preserve zeros and true equal-valued remeasurements.

    No clinical plausibility or anchor freshness filter is applied here.
    Non-temperature units are label conventions, not explicit raw unit fields.
    """
    mapping = pd.DataFrame(LABELS, columns=[
        'nursingchartcelltypevallabel', 'nursingchartcelltypevalname', 'variable', 'unit'])
    frame = raw.loc[raw.nursingchartcelltypecat.eq('Vital Signs')].merge(
        mapping, on=['nursingchartcelltypevallabel', 'nursingchartcelltypevalname'],
        how='inner', validate='many_to_one')
    numeric = frame[['nursingchartoffset', 'nursingchartentryoffset', 'nursingchartvalue']].apply(
        pd.to_numeric, errors='coerce').astype(float)
    ids = frame[['patientunitstayid', 'nursingchartid']].astype('string')
    valid = (ids.patientunitstayid.str.fullmatch(r'\d+', na=False)
             & ids.nursingchartid.str.fullmatch(r'\d+', na=False))
    valid &= np.isfinite(numeric).all(axis=1)
    valid &= numeric.nursingchartentryoffset >= numeric.nursingchartoffset
    frame, numeric, ids = frame.loc[valid].copy(), numeric.loc[valid], ids.loc[valid]
    origin = pd.Timestamp('2100-01-01')
    frame['stay_id'] = 'eicu:' + ids.patientunitstayid
    frame['source_id'] = 'nurse:' + ids.nursingchartid
    frame['measured_at'] = origin + pd.to_timedelta(numeric.nursingchartoffset, unit='m', errors='coerce')
    frame['available_at'] = origin + pd.to_timedelta(numeric.nursingchartentryoffset, unit='m', errors='coerce')
    frame['value'] = numeric.nursingchartvalue.astype(float)
    frame['source_kind'] = 'bedside_available'
    frame['unit_basis'] = np.where(frame.variable.eq('temperature'), 'explicit_label', 'label_convention')
    frame['measurement_method'] = frame.nursingchartcelltypevalname
    return canonical_events(frame[[
        'stay_id', 'source_id', 'measured_at', 'available_at', 'variable', 'value',
        'unit', 'unit_basis', 'source_kind', 'measurement_method']])
