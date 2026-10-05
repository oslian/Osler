"""Fixed paired coverage contrast on all outer development predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.score_additional_reconstruction import verify_frame, summarize_target, paired_summary, FAMILY_SIZE
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    dest = Path('artifacts/native_reconstruction_coverage_outer_evaluation_20260929.json')
    if dest.exists():
        raise ValueError('Existing evaluation')
    plan = torch.load(root / 'partitions_20260922.pt', weights_only=False)
    scaling = torch.load(root / 'scaling_20260922.pt', weights_only=False)
    frames, paths, checked = [], [Path(__file__), Path('scripts/score_additional_reconstruction.py')], {}
    for fold in range(5):
        p = root / f'coverage_outer_20260927/fold{fold}.parquet'
        mp = p.with_suffix('.json')
        meta = json.loads(mp.read_text())
        if sha256(p) != meta['prediction_sha256'] or meta['role'] != 'test':
            raise ValueError('Invalid prediction metadata')
        for name, digest in meta['bindings'].items():
            if name not in checked:
                checked[name] = sha256(Path(name))
            if checked[name] != digest:
                raise ValueError('Changed dependency: ' + name)
        d = pd.read_parquet(p)
        verify_frame(d, scaling, plan['folds'][fold], fold)
        old = pd.read_parquet(root / f'outer_20260924/fold{fold}.parquet')
        cols = ['patient', 'target', 'component', 'truth', 'fit_scale', 'population']
        pd.testing.assert_frame_equal(d[cols], old[cols])
        np.testing.assert_array_equal(d.replacement_real, old.real)
        np.testing.assert_array_equal(d.replacement_process, old.process)
        frames.append(d); paths.extend((p, mp))
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['patient', 'target']).any() or data.groupby('component').fold.nunique().max() != 1:
        raise ValueError('Overlapping outer identities')
    if len(data) != np.isfinite(scaling['centers']).sum() or data.target.nunique() != FAMILY_SIZE:
        raise ValueError('Incomplete target coverage')
    arms = ['real', 'process', 'replacement_real', 'replacement_process', 'population']
    results = {}
    for i, (target, part) in enumerate(data.groupby('target', sort=True)):
        base = summarize_target(part, 20260929 + 1000*i)
        errors = part[arms].subtract(part.truth, axis=0).abs()
        norm = errors.div(part.fit_scale, axis=0)
        delta = errors.real - errors.replacement_real
        groups = delta.groupby(part.component).mean()
        pair = dict(patient=paired_summary(delta, 20260929+i),
                    component=paired_summary(groups, 20261929+i))
        incremental = len(part)>=30 and len(groups)>=3 and all(s['p_less'] is not None and s['p_less'] < .05/FAMILY_SIZE and s['mean']<0 for s in pair.values())
        results[target] = dict(base_screen=base['descriptive_screen_pass'],
            incremental_screen=bool(incremental), comparisons=pair,
            base_assessment=base,
            raw_mae=dict(patient=errors.mean().to_dict(), component=errors.groupby(part.component).mean().mean().to_dict()),
            normalized_mae=dict(patient=norm.mean().to_dict(), component=norm.groupby(part.component).mean().mean().to_dict()),
            component_harms=int((groups>0).sum()), component_count=len(groups),
            component_deltas={str(k):float(v) for k,v in groups.items()})
        print('Scored',i+1,'of',FAMILY_SIZE,flush=True)
    prior = json.loads(Path('artifacts/native_reconstruction_additional_evaluation_20260924.json').read_text())['summary']['descriptive_passes']
    passes = [t for t,r in results.items() if r['base_screen']]
    summary = dict(pairs=len(data), patients=int(data.patient.nunique()),
        base_screen_passes=passes, retained=sorted(set(prior)&set(passes)),
        lost=sorted(set(prior)-set(passes)), added=sorted(set(passes)-set(prior)),
        incremental_screen_passes=[t for t,r in results.items() if r['incremental_screen']],
        target_macro={level:{arm:float(np.mean([r['normalized_mae'][level][arm] for r in results.values()])) for arm in arms} for level in ('patient','component')})
    report = dict(summary=summary, targets=results, bindings={str(p):sha256(p) for p in paths},
        promotion_allowed=False, runtime_modified=False,
        limitation='Descriptive current reconstruction in reused development cohorts, overlapping training folds and certified fold2 reuse. No clinical, forecast or untouched-external claim.')
    dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
