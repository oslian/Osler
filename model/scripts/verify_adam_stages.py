"""Independently check E39 stored finite losses and all scalar summaries."""
import json
from math import fsum, isfinite
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts import diagnose_adam_stages as audit
from scripts.verify_frozen_bridge import compare

OUTPUT = Path('artifacts/native_adam_stages_verification_20261005.json')


def finite_loss_parity(frame):
    for suffix in ('before', 'after'):
        p = torch.tensor(frame['prediction_'+suffix].to_numpy(), dtype=torch.float32)
        t = torch.tensor(frame.truth.to_numpy(), dtype=torch.float32)
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite stored predictions')
        for name, expected in (('loss_', F.smooth_l1_loss(p,t,reduction='none')),
                               ('mae_', (p-t).abs())):
            np.testing.assert_array_equal(expected.double().numpy(), frame[name+suffix].to_numpy())


def scalar_summary(frame):
    if frame.empty:
        return dict(status='unsupported', examples=0)
    names = (*audit.STAGES, 'none')
    harm = dict.fromkeys(audit.STAGES, 0); first = dict.fromkeys(names, 0)
    final_first = dict.fromkeys(names, 0)
    flips = {s:dict(safe_to_harm=0,harm_to_safe=0) for s in audit.STAGES[1:]}
    finite = dict(harmful=0,helpful=0,ambiguous=0,actual_dot_harm_and_finite_harm=0,
                  actual_dot_safe_but_finite_harm=0)
    unscaled, maximum, losses, maes = 0, 0., [], []
    for row in frame.to_dict('records'):
        flags = []
        for s in audit.STAGES:
            dot, tol = row[s+'_dot'], row[s+'_tolerance']
            if not isfinite(dot) or not isfinite(tol) or tol <= 0:
                raise ValueError('Invalid dot/tolerance')
            flags.append(dot > tol); harm[s] += int(flags[-1])
        first_stage = next((s for s, h in zip(audit.STAGES,flags) if h), 'none')
        first[first_stage] += 1
        if flags[-1]:final_first[first_stage] += 1
        for i, s in enumerate(audit.STAGES[1:]):
            flips[s]['safe_to_harm'] += int(not flags[i] and flags[i+1])
            flips[s]['harm_to_safe'] += int(flags[i] and not flags[i+1])
        change = row['loss_after']-row['loss_before']
        tol = 1e-12*(1+abs(row['loss_before'])+abs(row['loss_after']))
        bad, good = change > tol, change < -tol
        finite['harmful'] += int(bad); finite['helpful'] += int(good)
        finite['ambiguous'] += int(not bad and not good)
        finite['actual_dot_harm_and_finite_harm'] += int(flags[-1] and bad)
        finite['actual_dot_safe_but_finite_harm'] += int(not flags[-1] and bad)
        unscaled += int(row['unscaled_proposal_dot'] > row['unscaled_proposal_tolerance'])
        maximum = max(maximum,row['maximum_coordinate_roundoff'])
        losses.append(change); maes.append(row['mae_after']-row['mae_before'])
    return dict(status='scored',examples=len(frame),harmful=harm,first_harm=first,
        first_harm_among_actual_harm=final_first,stage_transitions=flips,finite_loss=finite,
        mean_loss_change=fsum(losses)/len(frame),mean_mae_change=fsum(maes)/len(frame),
        unscaled_proposal_harm=unscaled,maximum_coordinate_roundoff=maximum)


def main():
    if OUTPUT.exists():raise ValueError('Verification already complete')
    report = json.loads(audit.OUTPUT.read_text())
    if not report['complete'] or report['runtime_modified'] or report['promotion_allowed']:
        raise ValueError('Unexpected diagnostic contract')
    audit.parent.check_bindings(report['bindings'])
    original = json.loads(audit.parent.OUTPUT.read_text())
    audit.parent.check_bindings(original['bindings'])
    all_rows, maximum = [], 0.
    for fold in range(5):
        for arm in audit.parent.ARMS:
            p, mp = (audit.DEST/f'fold{fold}'/f'{arm}.{ext}' for ext in ('parquet','json'))
            meta = json.loads(mp.read_text()); audit.parent.check_bindings(meta['bindings'])
            if not meta['tensors_exact'] or meta['sha256'] != audit.sha256(p):
                raise ValueError('Invalid trace certificate')
            frame = pd.read_parquet(p)
            expected_keys = {(epoch, step, slot) for epoch in (1,2) for step in range(1024) for slot in (0,1)}
            keys = list(frame[['epoch','step','slot']].itertuples(index=False,name=None))
            if len(keys) != len(expected_keys) or set(keys) != expected_keys or set(frame.fold) != {fold} or set(frame.arm) != {arm}:
                raise ValueError('Incomplete trace identity')
            if any(report['inventory'][int(t)] != v for t,v in zip(frame.target,frame.variable)):
                raise ValueError('Incorrect target labels')
            finite_loss_parity(frame)
            summary = scalar_summary(frame); maximum = max(maximum, compare(summary,meta['summary']))
            source = json.loads((audit.parent.DEST/f'fold{fold}'/f'{arm}.json').read_text())
            if meta['training'] != source['training']:
                raise ValueError('Parent log mismatch')
            if summary['harmful']['actual'] != sum(x['actual_harmful_dots'] for x in source['training']):
                raise ValueError('Parent actual harm count mismatch')
            if summary['unscaled_proposal_harm'] != sum(x['proposed_harmful_dots'] for x in source['training']):
                raise ValueError('Parent proposal count mismatch')
            all_rows.append(frame)
    frame = pd.concat(all_rows,ignore_index=True)
    for arm in audit.parent.ARMS:
        f = frame[frame.arm == arm]
        result = dict(all=scalar_summary(f),same_target=scalar_summary(f[f.same_target]),
            cross_target=scalar_summary(f[~f.same_target]),
            targets={v:scalar_summary(f[f.variable == v]) for v in report['inventory']})
        maximum = max(maximum,compare(result,report['results'][arm]))
    if report['examples'] != len(frame) or len(frame) != 40960:
        raise ValueError('Incorrect total exposure')
    paths = [audit.OUTPUT,Path(__file__),Path('tests/test_adam_stages_verification.py'),
             Path('scripts/verify_frozen_bridge.py')]
    bindings = dict(report['bindings'],**{str(p):audit.sha256(p) for p in paths})
    audit.parent.check_bindings(bindings)
    audit.parent.write_json(OUTPUT,dict(complete=True,bindings=bindings,rows=len(frame),
        finite_loss_parity_exact=True,maximum_scalar_roundoff=maximum,
        runtime_modified=False,promotion_allowed=False))
    print(json.dumps(dict(rows=len(frame),maximum_scalar_roundoff=maximum,complete=True)),flush=True)


if __name__ == '__main__':main()
