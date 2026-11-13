"""Bounded raw-glucose event-identity and availability audit; no runtime change."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from scripts.audit_recurrent_event_state_loop import prepare_replay, patient_equal_mae
from scripts.audit_event_aware_patient_state import patient_partition, paired
from whole_body_jepa.multisource_intervals import SEEDS, sha256
from whole_body_jepa.raw_lab_replay import canonical_events, build_replay, RawLabResidual, batch_replays, predictions


def extract_events(frames, database, root, cache):
    stays = pd.DataFrame({'stay_id': sorted(set(str(s) for f in frames for s in f.stay_id))})
    stays['native_id'] = stays.stay_id.str.split(':').str[-1].astype(int)
    inputs = [root / ('lab.csv.gz' if database == 'eicu' else 'hosp/labevents.csv.gz')]
    if database != 'eicu':
        inputs.append(root/'icu/icustays.csv.gz')
    binding = {'stays': sorted(stays.stay_id), 'source': sha256(Path(__file__)),
               'files': [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]}
    signature = json.dumps(binding, sort_keys=True)
    side = cache.with_suffix('.json')
    if cache.exists() and side.exists() and side.read_text() == signature:
        return pd.read_parquet(cache)
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.register('wanted', stays)
    if database == 'eicu':
        raw = c.execute('''SELECT w.stay_id, cast(l.labid as varchar) source_id,
            timestamp '2100-01-01' + l.labresultoffset * interval '1 minute' measured_at,
            timestamp '2100-01-01' + greatest(l.labresultoffset, l.labresultrevisedoffset) * interval '1 minute' available_at,
            cast(l.labresult as double) AS "value"
            FROM read_csv_auto(?) l JOIN wanted w ON l.patientunitstayid=w.native_id
            WHERE lower(l.labname) IN ('glucose','bedside glucose')
            AND l.labresult BETWEEN 10 AND 2000 AND l.labresultrevisedoffset IS NOT NULL
            AND lower(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,'')) IN ('mg/dl','mg/dl.')''', [str(inputs[0])]).df()
    else:
        raw = c.execute('''SELECT w.stay_id, cast(l.labevent_id as varchar) source_id,
            l.charttime measured_at, greatest(l.charttime,l.storetime) available_at, cast(l.valuenum as double) AS "value"
            FROM read_csv_auto(?) l JOIN read_csv_auto(?) s ON l.hadm_id=s.hadm_id
              AND l.subject_id=s.subject_id AND l.charttime BETWEEN s.intime AND s.outtime
            JOIN wanted w ON s.stay_id=w.native_id
            WHERE l.itemid IN (50809,50931,52027,52569) AND l.valuenum BETWEEN 10 AND 2000
            AND l.storetime IS NOT NULL AND lower(l.valueuom)='mg/dl' ''', [str(inputs[0]), str(inputs[1])]).df()
    c.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache, index=False)
    side.write_text(signature)
    return raw


def audit_and_sequences(frame, raw, center, scale):
    events = canonical_events(raw)
    frame = frame.copy()
    strict = np.zeros(len(frame), bool)
    resolved = np.zeros(len(frame), bool)
    audits = {'raw_rows': len(raw), 'deduplicated_rows': len(events), 'equal_value_distinct_event_pairs': 0,
              'cached_current_known_at_anchor': 0, 'scored_current_unavailable_or_mismatched': 0,
              'queries_reusing_latest_event': 0, 'queries_with_known_event': 0,
              'scored_rows': int(np.isfinite(frame.truth).sum()), 'scored_current_raw_match': 0,
              'scored_current_matched_but_available_later': 0, 'scored_current_no_raw_match': 0,
              'scored_future_raw_match': 0}
    delays = []
    seq = {'once': [], 'repeat_at_query': []}
    by_stay = dict(tuple(events.groupby('stay_id', sort=False)))
    for stay, query in frame.groupby('stay_id', sort=False):
        ev = by_stay.get(stay)
        if ev is None:
            continue
        by_time = ev.sort_values(['measured_at', 'available_at', 'source_id'])
        audits['equal_value_distinct_event_pairs'] += int(((by_time.value.diff()==0) & (by_time.source_id != by_time.source_id.shift())).sum())
        last_id = None
        prior_outcomes = []
        for i, row in query.sort_values('t').iterrows():
            available = ev.loc[ev.available_at <= row.t]
            raw_match = ev.loc[np.isclose(ev.value, row.current, atol=1e-4, rtol=0)
                              & ((ev.measured_at-row.current_event_time).dt.total_seconds().abs() <= .001)]
            current_match = raw_match.loc[(raw_match.available_at <= row.t)
                                          & ((row.t-raw_match.measured_at).dt.total_seconds().between(0, 6*3600))]
            if np.isfinite(row.truth):
                if len(raw_match):
                    audits['scored_current_raw_match'] += 1
                    first_available = raw_match.available_at.min()
                    if first_available > row.t:
                        audits['scored_current_matched_but_available_later'] += 1
                        delays.append((first_available-row.t).total_seconds()/3600)
                else:
                    audits['scored_current_no_raw_match'] += 1
            known = not current_match.empty
            if known:
                audits['cached_current_known_at_anchor'] += 1
            elif np.isfinite(row.truth):
                audits['scored_current_unavailable_or_mismatched'] += 1
            if len(available):
                key = (available.iloc[-1].source_id, available.iloc[-1].available_at)
                audits['queries_with_known_event'] += 1
                audits['queries_reusing_latest_event'] += int(key == last_id)
                last_id = key
            resolved[i] = any(t <= row.t for t in prior_outcomes)
            future = ev.loc[(ev.measured_at == row.future_event_time) & np.isclose(ev.value, row.truth, atol=1e-4, rtol=0)]
            if np.isfinite(row.truth) and len(future):
                audits['scored_future_raw_match'] += 1
            strict[i] = known and np.isfinite(row.truth) and len(future)>0
            if strict[i]:
                prior_outcomes.append(future.available_at.min())
        for mode in seq:
            s = build_replay(ev, query, center=center, scale=scale, repeat_at_query=mode=='repeat_at_query')
            if len(s.rows):
                seq[mode].append(s)
    frame['strict'] = strict
    frame['primary'] = strict & resolved
    # Labels never select update events, only fitting/scoring queries.
    audits.update({'strict_rows': int(strict.sum()), 'primary_rows': int(frame.primary.sum()),
                   'matched_late_current_median_delay_hours': float(np.median(delays)) if delays else None,
                   'matched_late_current_p95_delay_hours': float(np.quantile(delays,.95)) if delays else None,
                   'primary_patients': int(frame.loc[frame.primary,'subject_id'].nunique())})
    return frame, seq, audits


def fit(frame, sequences, seed, residual_scale):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    fit_mask, tune_mask = patient_partition(frame.subject_id, seed)
    tune = tune_mask & frame.strict.to_numpy()
    fit_sequences = [s for s in sequences if fit_mask[s.rows[s.rows>=0][0]]]
    model = RawLabResidual()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.002)
    truth = frame.truth.to_numpy(float)
    target = np.clip((truth-frame.frozen.to_numpy())/residual_scale, -4, 4)
    supervised = fit_mask & frame.strict.to_numpy()
    if frame.loc[supervised,'subject_id'].nunique()<50 or frame.loc[tune,'subject_id'].nunique()<30:
        raise ValueError('insufficient source fit/tune support')
    best = {'epoch': -1, 'mae': patient_equal_mae(frame.loc[tune], truth[tune], frame.frozen.to_numpy()[tune]), 'state': deepcopy(model.state_dict())}
    patience = 0
    for epoch in range(40):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(len(fit_sequences))
        for start in range(0,len(order),32):
            x,u,q,rows = batch_replays([fit_sequences[j] for j in order[start:start+32]])
            safe = np.maximum(rows,0)
            mask = (rows>=0) & supervised[safe]
            y = torch.from_numpy(np.where(mask, target[safe], 0).astype(np.float32))
            m = torch.from_numpy(mask.astype(np.float32))
            valid = m.sum(1)>0
            if not valid.any():
                continue
            pred = model(x,u,q)
            loss = ((torch.nn.functional.smooth_l1_loss(pred,y,reduction='none')*m).sum(1)/m.sum(1).clamp_min(1))[valid].mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        pred = predictions(model, sequences, frame, residual_scale)
        mae = patient_equal_mae(frame.loc[tune],truth[tune],pred[tune])
        if mae < best['mae']-1e-4:
            best = {'epoch': epoch, 'mae': mae, 'state': deepcopy(model.state_dict())}; patience=0
        else:
            patience+=1
        if patience>=10:
            break
    model.load_state_dict(best.pop('state'))
    return model,best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('refusing overwrite')
    root=Path('/Users/chunyouchang/Downloads')
    args.model_dir=Path('models'); args.scored_only=False
    args.eicu=Path('/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.mimic=Path('/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet')
    args.eicu_root=root/'eicu-collaborative-research-database-2.0'; args.mimic_root=root/'mimic-iv-3.1'
    args.eicu_context_cache=Path('/private/tmp/event_aware_eicu_glucose_6h_context.npz')
    args.mimic_context_cache=Path('/private/tmp/event_aware_mimic_glucose_6h_context.npz')
    replay,_,residual_scale,*_=prepare_replay(args)
    for db in ('eicu','mimic'):
        origin=pd.read_parquet(getattr(args,db),columns=['stay_id','t','glucose_age_hr'])
        origin['stay_id']=db+':'+origin.stay_id.astype(str)
        origin['t']=pd.to_datetime(origin.t)
        origin['current_event_time']=origin.t-pd.to_timedelta(origin.glucose_age_hr,unit='h')
        origin=origin.drop_duplicates(['stay_id','t']).set_index(['stay_id','t'])
        for role,f in replay.items():
            if (role=='mimic')==(db=='mimic'):
                replay[role]=f.join(origin[['current_event_time']],on=['stay_id','t'],validate='many_to_one')
    raw={}
    for db in ('eicu','mimic'):
        fs=[f for role,f in replay.items() if (role=='mimic')==(db=='mimic')]
        print('extract raw '+db,flush=True)
        raw[db]=extract_events(fs,db,getattr(args,db+'_root'),Path('/private/tmp/raw_glucose_identity_'+db+'.parquet'))
        print(json.dumps({'database':db,'raw_events':len(raw[db])}),flush=True)
    memory=raw['eicu'].loc[raw['eicu'].stay_id.isin(replay['memory'].stay_id)].value.to_numpy()
    center=float(np.median(memory)); scale=max(float(np.median(abs(memory-center))/.67448975),1.)
    data={role:audit_and_sequences(f,raw['mimic' if role=='mimic' else 'eicu'].loc[raw['mimic' if role=='mimic' else 'eicu'].stay_id.isin(f.stay_id)],center,scale)
          for role,f in replay.items() if role in ('development','eicu_hospital','mimic')}
    print(json.dumps({role:d[2] for role,d in data.items()}),flush=True)
    report={'status':'candidate_only','promotion_allowed':False,'runtime_modified':False,'causal_claim_allowed':False,
            'untouched_lockbox':False,'scope':'raw glucose laboratory history only; frozen JEPA anchor retained',
            'whole_body_availability_verified':False,'normalization':{'center':center,'scale':scale,'residual_scale':residual_scale},
            'support':{r:d[2] for r,d in data.items()},'binding':{str(f):sha256(f) for f in [Path(__file__),Path('whole_body_jepa/raw_lab_replay.py'),Path('RAW_LAB_IDENTITY_PROTOCOL_20260910.md')]},'seeds':[]}
    support_failures=[]
    for role in ('eicu_hospital','mimic'):
        if data[role][2]['primary_patients']<30:
            support_failures.append(role+': fewer than 30 primary patients')
    dev=data['development'][0]
    for seed in SEEDS:
        fit_mask,tune_mask=patient_partition(dev.subject_id,seed)
        fit_n=dev.loc[fit_mask & dev.strict,'subject_id'].nunique()
        tune_n=dev.loc[tune_mask & dev.strict,'subject_id'].nunique()
        if fit_n<50 or tune_n<30:
            support_failures.append(f'seed {seed}: fit/tune patient support {fit_n}/{tune_n}, requires 50/30')
    if support_failures:
        report.update({'status':'blocked_by_available_time_support','diagnostic_complete':True,
                       'training_started':False,'accuracy_comparison_complete':False,
                       'support_failures':support_failures})
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'support_failures':support_failures}),flush=True)
        return
    for seed in SEEDS:
        frame,seq,_=data['development']
        models={mode:fit(frame,s,seed,residual_scale) for mode,s in seq.items()}
        evaluation={}
        for role in ('eicu_hospital','mimic'):
            frame,seq,_=data[role]; mask=frame.primary.to_numpy(); work=frame.loc[mask]
            preds={mode:predictions(models[mode][0],seq[mode],frame,residual_scale) for mode in models}
            preds.update({'frozen':frame.frozen.to_numpy(), 'persistence':frame.current.to_numpy()})
            evaluation[role]={'rows':len(work),'patients':int(work.subject_id.nunique()),
                'mae':{name:patient_equal_mae(work,work.truth,v[mask]) for name,v in preds.items()},
                'comparisons':{name:paired(work,preds['once'][mask],preds[name][mask],f'rawidentity:{role}:{seed}:{name}',1000)
                               for name in ('repeat_at_query','frozen','persistence')}}
        report['seeds'].append({'seed':seed,'fit':{mode:item[1] for mode,item in models.items()},'evaluation':evaluation})
        print(json.dumps(report['seeds'][-1]),flush=True)
    report['complete']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
