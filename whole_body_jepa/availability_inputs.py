"""Availability-safe laboratory snapshots for the existing full JEPA schema."""
import numpy as np
import pandas as pd

from whole_body_jepa.raw_lab_replay import canonical_events
from whole_body_jepa.runtime import HISTORY_LAGS_HOURS


# Strict canonical-unit whitelist. Unknown units remain unavailable, not guessed.
UNITS = {}
for names, units in (
    ('glucose creatinine bun bilirubin bilirubin_direct calcium magnesium phosphate triglycerides uric_acid', ('mg/dl','mg/dl.')),
    ('sodium potassium chloride bicarbonate anion_gap', ('mmol/l','meq/l')),
    ('lactate ionized_calcium', ('mmol/l',)),
    ('hemoglobin albumin total_protein', ('g/dl',)),
    ('hematocrit', ('%',)),
    ('wbc platelets', ('k/ul','k/mcl','x1000','10^3/ul','1000/ul')),
    ('ast alt alkaline_phos lipase amylase cpk ldh', ('iu/l','u/l','units/l')),
    ('paco2 pao2', ('mmhg',)),
    ('ph', ('units','ph')),
    ('inr', ('ratio',)),
    ('ptt', ('sec','seconds','s')),
    ('fibrinogen', ('mg/dl',)),
    ('tsh', ('uiu/ml','miu/l','mcu/ml')),
    ('free_t4', ('ng/dl',)),
    ('cortisol', ('ug/dl','mcg/dl')),
    ('troponin_i troponin_t myoglobin ferritin', ('ng/ml',)),
    ('bnp nt_pro_bnp', ('pg/ml',)),
    ('serum_osmolality', ('mosm/kg','mosmol/kg','mosm/kg h2o')),
    ('esr', ('mm/hr',)),
):
    for name in names.split():
        UNITS[name] = units


def filter_units(raw, bounds):
    raw = raw.copy()
    unit = raw.unit.fillna('').str.lower().str.strip()
    valid = np.zeros(len(raw), bool)
    for var, allowed in UNITS.items():
        if var in bounds:
            lo, hi = bounds[var]
            valid |= (raw.variable.eq(var) & unit.isin(allowed) & raw.value.between(lo,hi)).to_numpy()
    return canonical_events(raw.loc[valid])


def snapshots(raw, times, variables, max_age):
    """Single-stay chronological replay. Late old results never replace newer draws."""
    if raw.stay_id.nunique() > 1:
        raise ValueError('snapshots require one stay')
    raw = canonical_events(raw)
    ordered = list(raw.itertuples())
    pos = {v:i for i,v in enumerate(variables)}
    state = {}
    versions = {}
    values = np.full((len(times),len(variables)), np.nan)
    ages = values.copy()
    cursor = 0
    for row,t in enumerate(times):
        changed = set()
        while cursor < len(ordered) and ordered[cursor].available_at <= t:
            e = ordered[cursor]
            versions.setdefault(e.variable, {})[str(e.source_id)] = e
            changed.add(e.variable)
            cursor += 1
        for var in changed:
            known = list(versions[var].values())
            latest_time = max(e.measured_at for e in known)
            latest = [e for e in known if e.measured_at == latest_time]
            qualities = [getattr(e, 'quality_valid', True) for e in latest]
            quality_ok = all(not pd.isna(q) and bool(q) for q in qualities)
            state[var] = (latest[0], len({float(e.value) for e in latest}) == 1 and quality_ok)
        for var,(e,unambiguous) in state.items():
            if var not in pos:
                continue
            age = (t-e.measured_at).total_seconds()/3600
            ages[row,pos[var]] = age
            if unambiguous and 0 <= age <= max_age:
                values[row,pos[var]] = e.value
    return values,ages


def build_arrays(frame, raw, assets, origins, *, target_only=False, truthful=True, query_horizons=None):
    """Use exact lag timestamps, including genuinely empty early history slots."""
    variables = assets.checkpoint['variables']
    frames, history, count = [], np.empty((len(frame),len(HISTORY_LAGS_HOURS)),int), 0
    groups = dict(tuple(raw.groupby('stay_id',sort=False)))
    for stay, queries in frame.groupby('stay_id',sort=False):
        requests = np.asarray([[t-pd.Timedelta(hours=h) for h in HISTORY_LAGS_HOURS] for t in queries.t], dtype='datetime64[ns]')
        times = pd.DatetimeIndex(np.unique(requests))
        events = groups.get(stay, raw.iloc[:0])
        if target_only:
            events = events.loc[events.variable.eq('glucose')]
        values,ages = snapshots(events,times,variables,assets.max_age_hours)
        f = pd.DataFrame({f'{v}_t':values[:,i] for i,v in enumerate(variables)})
        for i,v in enumerate(variables):
            f[f'{v}_age_hr'] = ages[:,i]
        f['hours_since_onset'] = (times-pd.Timestamp(origins[stay])).total_seconds()/3600
        f['_time_hours'] = f.hours_since_onset
        frames.append(f)
        history[queries.index] = np.searchsorted(times.to_numpy(),requests)+count
        count += len(f)
    allrows = pd.concat(frames,ignore_index=True)
    current,mask,ages = assets.scaler.normalize_current(allrows)
    presence = np.zeros((len(allrows),len(assets.checkpoint['modules'])),np.float32)
    ids = assets.model.variable_module_ids.cpu().numpy()
    for i in range(presence.shape[1]):
        presence[:,i] = mask[:,ids==i].any(axis=1)
    if not truthful:
        mask = np.ones_like(mask)
        ages = np.zeros_like(ages)
        presence = np.ones_like(presence)
    onset = np.clip(allrows.hours_since_onset.to_numpy(),0,168)[:,None]/24
    elapsed = np.clip(allrows._time_hours.to_numpy(),0,168)[:,None]/168
    matrix = np.concatenate([current,mask,ages,onset,presence,elapsed],axis=1).astype(np.float32)
    if matrix.shape[1] != assets.checkpoint['input_dim']:
        raise ValueError('checkpoint input width mismatch')
    # Append lightweight query rows, pointing their histories at the raw snapshots.
    query_rows = history[:,-1]
    arrays = dict(input=matrix, current=current, mask=mask, ages=ages, presence=presence,
                  history=np.zeros((len(matrix),len(HISTORY_LAGS_HOURS)),int),
                  horizon=np.full(len(matrix),6,np.float32))
    arrays['history'][query_rows] = history
    if query_horizons is not None:
        requested = np.asarray(query_horizons, dtype=np.float32)
        if requested.shape != (len(frame),) or not np.isfinite(requested).all() or (requested <= 0).any():
            raise ValueError('query horizons must be finite positive hours per query')
        if len(np.unique(query_rows)) != len(query_rows):
            raise ValueError('duplicate query anchors cannot carry separate horizons')
        arrays['horizon'][query_rows] = requested
    return arrays,query_rows


def source_patient_support(arrays, rows, subjects, fit):
    """Count distinct source-fit patients with an observed input at any legal lag."""
    observed=arrays['mask'][arrays['history'][rows[fit]]].any(axis=1)
    return pd.DataFrame(observed).groupby(np.asarray(subjects)[fit]).any().sum().to_numpy()


def restrict_input_support(arrays, allowed, module_ids):
    """Make unsupported channels unknown in both context and complete history."""
    result={k:v.copy() for k,v in arrays.items()}
    allowed=np.asarray(allowed,dtype=bool)
    n=len(allowed)
    result['current'][:,~allowed]=0
    result['mask'][:,~allowed]=0
    result['ages'][:,~allowed]=7.  # Existing unknown-age encoding:168h /24.
    for i in range(result['presence'].shape[1]):
        result['presence'][:,i]=result['mask'][:,module_ids==i].any(axis=1)
    result['input'][:,:n]=result['current']
    result['input'][:,n:2*n]=result['mask']
    result['input'][:,2*n:3*n]=result['ages']
    result['input'][:,3*n+1:-1]=result['presence']
    return result


def fit_numeric_rescale(arrays,rows,fit,target_index):
    """Source-fit snapshot standardization in inherited normalized coordinates."""
    history_rows=np.unique(arrays['history'][rows[fit]])
    n=arrays['current'].shape[1]
    centers=np.zeros(n,np.float32);scales=np.ones(n,np.float32)
    for i in range(n):
        if i==target_index:continue
        observed=arrays['mask'][history_rows,i]>0
        values=arrays['current'][history_rows[observed],i]
        if len(values)<2:continue
        spread=float(np.std(values.astype(np.float64)))
        if spread>=1e-6:
            centers[i]=np.median(values);scales[i]=spread
    return centers,scales


def apply_numeric_rescale(arrays,centers,scales):
    """Keep observation semantics unchanged; only change numerical coordinates."""
    result={k:v.copy() for k,v in arrays.items()}
    result['current']=np.where(arrays['mask']>0,(arrays['current']-centers)/scales,0).astype(np.float32)
    result['input'][:,:len(centers)]=result['current']
    return result
