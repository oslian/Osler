"""Matched eight vs32 epoch predictions; no external checkpoint selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--rescale',action='store_true')
    group.add_argument('--context-dropout',action='store_true')
    group.add_argument('--history-mask',action='store_true')
    group.add_argument('--source-patients',type=int,choices=(1000,2250))
    group.add_argument('--expanded-budget',action='store_true')
    group.add_argument('--actual-time',action='store_true')
    args=parser.parse_args()
    matched=args.rescale or args.context_dropout or args.history_mask
    name='history_mask' if args.history_mask else 'context_dropout' if args.context_dropout else 'rescale' if args.rescale else 'budget'
    short_path=Path('artifacts/encoder_budget_20260910.json' if matched else 'artifacts/encoder_confirm_20260910.json')
    long_path=Path(f'artifacts/encoder_{name}_20260910.json')
    if args.source_patients:
        name=f'source{args.source_patients}'
        short_path=Path('artifacts/encoder_history_mask_20260910.json')
        long_path=Path(f'artifacts/encoder_{name}_20260910.json')
    short=json.loads(short_path.read_text());long=json.loads(long_path.read_text())
    if args.actual_time:
        name='actual_time'
        short_path=Path('artifacts/encoder_source2250_64_20260910.json')
        long_path=Path(f'artifacts/encoder_{name}_20260910.json')
        short=json.loads(short_path.read_text());long=json.loads(long_path.read_text())
        if not long['settings']['actual_time']:raise ValueError('not actual time trained')
        if any(v!=long['settings'][k] for k,v in short['settings'].items()):
            raise ValueError('other training settings changed')
    if args.expanded_budget:
        name='source2250_64'
        short_path=Path('artifacts/encoder_source2250_20260910.json')
        long_path=Path(f'artifacts/encoder_{name}_20260910.json')
        short=json.loads(short_path.read_text());long=json.loads(long_path.read_text())
        if short['settings']['epochs']!=32 or long['settings']['epochs']!=64:
            raise ValueError('unexpected epoch budget')
        for key in short['settings']:
            if key!='epochs' and short['settings'][key]!=long['settings'][key]:
                raise ValueError('non-budget training change')
    if not short.get('complete') or not long.get('complete'): raise ValueError('training incomplete')
    if args.source_patients:
        if any(long['data_binding'].get(k)!=v for k,v in short['data_binding'].items()):
            raise ValueError('original input data changed')
        if long['settings']['extra_patients']!=args.source_patients-250:
            raise ValueError('unexpected source expansion')
        for report in (short,long):
            if not report['settings']['history_mask'] or report['settings']['epochs']!=32:
                raise ValueError('unmatched training contract')
    elif short['data_binding']!=long['data_binding']: raise ValueError('different input data')
    if short['settings']['seeds']!=long['settings']['seeds']: raise ValueError('seed mismatch')
    report=dict(status='candidate_only',promotion_allowed=False,runtime_modified=False,
        untouched_lockbox=False,binding={str(p):sha256(p) for p in
            (short_path,long_path,Path(__file__),Path('ENCODER_EXPANDED_BUDGET_PROTOCOL_20260910.md' if args.expanded_budget else 'ENCODER_LEARNING_CURVE_PROTOCOL_20260910.md' if args.source_patients else f'ENCODER_{name.upper()}_PROTOCOL_20260910.md'))},seeds=[])
    for early,late in zip(short['seeds'],long['seeds']):
        seed=late['seed']
        entry=dict(seed=seed,control_logs_identical={},selected_epochs={},evaluation={})
        for arm,t in late['training'].items():
            if args.source_patients and (late['fit_patients']!=args.source_patients or late['tune_patients']!=136):
                raise ValueError('unexpected fit/tune patient support')
            if not args.actual_time and not args.source_patients and not args.history_mask and (not matched or arm.startswith('own')):
                same=(t['log'] if matched else t['log'][:32 if args.expanded_budget else 8])==early['training'][arm]['log']
                entry['control_logs_identical'][arm]=same
                if not same: raise ValueError(f'control training changed:{seed}:{arm}')
            entry['selected_epochs'][arm]=dict(short=early['training'][arm]['best_epoch'],long=t['best_epoch'])
        for role in ('eicu_hospital','mimic'):
            frame_path=Path(f'/private/tmp/availability_glucose_{role}.parquet')
            if sha256(frame_path)!=long['data_binding'][str(frame_path)]: raise ValueError('query cache changed')
            f=pd.read_parquet(frame_path)
            f=f.loc[f.fresh & f.truth.notna()].reset_index(drop=True)
            dirs=[Path(r['training']['own_bounded']['checkpoint']).parent for r in (early,late)]
            with np.load(dirs[0]/f'predictions_{role}_{seed}.npz') as old, np.load(dirs[1]/f'predictions_{role}_{seed}.npz') as new:
                if matched and not args.history_mask:
                    for arm in ('own_bounded','own_unbounded'):np.testing.assert_array_equal(old[arm],new[arm])
                entry['evaluation'][role]={arm:paired(f,new[arm],old[arm],f'budget:{seed}:{role}:{arm}',1000)
                    for arm in late['training']}
                if args.actual_time:
                    entry.setdefault('retrospective_actual_query_descriptive_only',{})[role]={
                        arm:patient_equal_mae(f,f.truth,np.load(dirs[1]/f'actual_predictions_{role}_{arm}_{seed}.npy'))
                        for arm in late['training']}
        report['seeds'].append(entry)
    report['complete']=True
    Path(f'artifacts/encoder_{name}_comparison_20260910.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
