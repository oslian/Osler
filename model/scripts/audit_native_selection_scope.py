"""Does glucose-selected checkpoint improve source-tune multi-target error?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def main():
    root=Path('/private/tmp/rate_anchored_potassium_20260911')
    output=Path('artifacts/native_selection_scope_20260911.json')
    if output.exists():raise ValueError('fresh output required')
    ref=json.loads(Path('artifacts/native_selected_retention_20260911.json').read_text())
    for n,h in ref['binding'].items():
        if sha256(Path(n))!=h:raise ValueError('changed input')
    def load(n):
        with np.load(root/n) as f:return {k:f[k] for k in f.files}
    valid=load('labels.npz')['paired'][:,0];labels=load('multitarget_labels.npz')
    q=pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    assets=load_release(Path('models'));vs=assets.checkpoint['variables'];col=vs.index('glucose')
    report=dict(role='source_tune_descriptive_only',runtime_modified=False,binding=ref['binding'].copy(),seeds=[])
    report['binding'][str(Path(__file__))]=sha256(Path(__file__))
    for entry in ref['seeds']:
        seed=entry['seed'];parts=load(f'partition_{seed}.npz');tune=parts['tune'][valid]&labels['paired'][:,col];fit=parts['fit'][valid]&labels['paired'][:,col]
        cached=load(f'glucose_retention_predictions_{seed}.npz')
        points=dict(initial=cached['initial'],glucose_only=cached['original'],uniform=cached['final'],selected=load(f'glucose_selected_retention_predictions_{seed}.npz')['final'])
        targets={}
        for j,v in enumerate(vs):
            m=tune&labels['paired'][:,j];training=fit&labels['paired'][:,j]
            if q.loc[m].subject_id.nunique()<30 or q.loc[training].subject_id.nunique()<30:continue
            truth=labels['future'][m,j];frame=q.loc[m]
            targets[v]=dict(patients=int(frame.subject_id.nunique()),scale=float(assets.scaler.scales[j]),mae={a:patient_equal_mae(frame,truth,p[m,j]) for a,p in points.items()})
        score={a:float(np.mean([t['mae'][a]/t['scale'] for t in targets.values()])) for a in points}
        report['seeds'].append(dict(seed=seed,targets=targets,target_equal_scaled_mae=score,primary_mae=targets['glucose']['mae']))
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
