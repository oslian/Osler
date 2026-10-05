"""Retrospective infusion context; event timestamps are not availability proof."""
import numpy as np
import pandas as pd

from osler_jepa.treatment_units import TARGET_TREATMENT_ACTIONS, ACTION_RATE_DIMENSIONS

ACTIONS = TARGET_TREATMENT_ACTIONS['potassium']


def action_support_status(subjects, exposed, partitions, min_leaf=30):
    """Necessary support checks, not power analysis or guaranteed learnability."""
    subjects=np.asarray(subjects)
    exposed=np.asarray(exposed,dtype=bool)
    counts={name:dict(rows=int((mask&exposed).sum()),
                      patients=int(len(np.unique(subjects[mask&exposed]))))
            for name,mask in zip(('fit','tune','evaluation'),partitions)}
    fit=partitions[0]
    counts['pure_exposure_split_supported']=bool((fit&exposed).sum()>=min_leaf and (fit&~exposed).sum()>=min_leaf)
    counts['evaluation_has_exposure']=bool(counts['evaluation']['patients']>0)
    counts['interpretation']='support_only_not_efficacy_or_power'
    return counts


def order_record(row, action):
    """Order issuance only; ignore final cancellation, start/stop and dosage."""
    offset = pd.to_numeric(row.get('drugorderoffset'), errors='coerce')
    if not np.isfinite(offset):
        return None
    return dict(stay_id=f"eicu:{row['patientunitstayid']}",
                source_id=str(row['medicationid']), offset=float(offset),
                action=action, value=np.nan, dimension=None)


def context_features(queries, events):
    columns = []
    for action in ACTIONS:
        columns.extend([(action, 'count'), (action, 'age')])
        columns.extend((action, d) for d in sorted(set(ACTION_RATE_DIMENSIONS.get(action, ())) | {'volume_ml_per_hour'}))
    out = np.full((len(queries), len(columns)), np.nan)
    groups = dict(tuple(events.groupby('stay_id')))
    for i, q in enumerate(queries.itertuples()):
        e = groups.get(q.stay_id, events.iloc[:0])
        e = e[(e.offset > q.offset-360) & (e.offset <= q.offset)].sort_values(['offset', 'source_id'])
        for j, (action, kind) in enumerate(columns):
            part = e[e.action.eq(action)]
            if kind == 'count':
                out[i, j] = np.log1p(len(part))  # No recorded event, not no treatment.
            elif kind == 'age' and len(part):
                out[i, j] = (q.offset-part.offset.iloc[-1])/60
            elif kind not in ('age', 'count'):
                matching = part.loc[part.dimension.eq(kind)]
                if len(matching):
                    latest=matching.loc[matching.offset.eq(matching.offset.max()),'value'].to_numpy(float)
                    # IDs do not establish precedence among simultaneous records.
                    if np.isfinite(latest).all() and len(np.unique(latest))==1:
                        out[i,j]=latest[0]
    return out, columns


def shuffled_context(queries, values, partitions, seed):
    """Whole donor patient context, no self donor, no role or future-time crossing."""
    result = np.full_like(values, np.nan)
    rng = np.random.default_rng(seed)
    for mask in partitions:
        ids = np.unique(queries.loc[mask, 'subject_id'])
        if len(ids) < 2:
            raise ValueError('placebo requires two patients per partition')
        ids = rng.permutation(ids)
        for recipient, donor in zip(ids, np.roll(ids, 1)):
            target = np.flatnonzero(mask & queries.subject_id.eq(recipient).to_numpy())
            source = np.flatnonzero(mask & queries.subject_id.eq(donor).to_numpy())
            source = source[np.argsort(queries.offset.to_numpy()[source], kind='stable')]
            pos = np.searchsorted(queries.offset.to_numpy()[source], queries.offset.to_numpy()[target], side='right')-1
            good = pos >= 0
            result[target[good]] = values[source[pos[good]]]
    return result


def rate_history_features(queries, events):
    """Same-label rate changes only; no assumed duration or delivered dose."""
    columns=[(a,d,k) for a in ACTIONS for d in sorted(ACTION_RATE_DIMENSIONS.get(a,()))
             for k in ('range','change','span_hours')]
    out=np.full((len(queries),len(columns)),np.nan)
    groups=dict(tuple(events.groupby('stay_id')))
    for i,q in enumerate(queries.itertuples()):
        e=groups.get(q.stay_id,events.iloc[:0])
        e=e[(e.offset>q.offset-360)&(e.offset<=q.offset)]
        for j,(action,dimension,kind) in enumerate(columns):
            part=e[e.action.eq(action)&e.dimension.eq(dimension)&np.isfinite(e.value)].sort_values('offset')
            if len(part)<2 or part.original_label.nunique()!=1:continue
            if part.groupby('offset').value.nunique().max()>1:continue
            part=part.drop_duplicates('offset')
            if len(part)<2:continue
            out[i,j]={'range':part.value.max()-part.value.min(),
                      'change':part.value.iloc[-1]-part.value.iloc[0],
                      'span_hours':(part.offset.iloc[-1]-part.offset.iloc[0])/60}[kind]
    return out,columns


def rate_transition_features(queries, events):
    """Observed transitions, not continuity, delivery, or treatment effects."""
    kinds=('change_count','reversal_count','last_change_age_hours','last_change')
    columns=[(a,d,k) for a in ACTIONS for d in sorted(ACTION_RATE_DIMENSIONS.get(a,())) for k in kinds]
    out=np.full((len(queries),len(columns)),np.nan)
    groups=dict(tuple(events.groupby('stay_id')))
    for i,q in enumerate(queries.itertuples()):
        e=groups.get(q.stay_id,events.iloc[:0])
        e=e[(e.offset>q.offset-360)&(e.offset<=q.offset)]
        for action,dimension in dict.fromkeys((a,d) for a,d,_ in columns):
            part=e[e.action.eq(action)&e.dimension.eq(dimension)&np.isfinite(e.value)].sort_values('offset')
            if len(part)<2 or part.original_label.nunique()!=1:continue
            if part.groupby('offset').value.nunique().max()>1:continue
            part=part.drop_duplicates('offset')
            if len(part)<2:continue
            delta=np.diff(part.value.to_numpy(float));changed=np.flatnonzero(np.abs(delta)>1e-8)
            signs=np.sign(delta[changed])
            values=[len(changed),int(np.count_nonzero(np.diff(signs))) if len(signs)>1 else 0,
                    (q.offset-part.offset.iloc[changed[-1]+1])/60 if len(changed) else np.nan,
                    delta[changed[-1]] if len(changed) else np.nan]
            for kind,value in zip(kinds,values):out[i,columns.index((action,dimension,kind))]=value
    return out,columns


def changed_rate_brackets(offsets, values, draw):
    """Locate observed unequal pairs, not actual administration transition times."""
    offsets=np.asarray(offsets,dtype=float);values=np.asarray(values,dtype=float)
    if offsets.shape!=values.shape or offsets.ndim!=1:raise ValueError('invalid series')
    if not np.isfinite(offsets).all() or not np.isfinite(values).all() or not np.isfinite(draw):raise ValueError('nonfinite series')
    if np.any(np.diff(offsets)<=0):raise ValueError('strict event ordering required')
    changed=np.abs(np.diff(values))>1e-8
    after=changed&(offsets[:-1]>=draw)&(offsets[1:]>draw)
    straddles=changed&(offsets[:-1]<draw)&(offsets[1:]>draw)
    return bool(after.any()),bool(straddles.any())
