"""Choose output retention references using source tune patients only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.multisource_intervals import sha256


def main():
    root = Path('/private/tmp/rate_anchored_potassium_20260911')
    output = Path('artifacts/native_retention_selection_20260911.json')
    if output.exists(): raise ValueError('fresh output required')
    ref_path = Path('artifacts/native_retention_repair_20260911.json')
    reference = json.loads(ref_path.read_text())
    for name, digest in reference['binding'].items():
        if sha256(Path(name)) != digest: raise ValueError('bound input changed')
    def load(name):
        with np.load(root / name) as f: return {k:f[k] for k in f.files}
    labels = load('multitarget_labels.npz'); variables = labels['variables'].tolist()
    glucose = variables.index('glucose'); valid = load('labels.npz')['paired'][:,0]
    q = pd.read_parquet(root/'queries.parquet').loc[valid].reset_index(drop=True).rename(columns={'uniquepid':'subject_id'})
    report = dict(complete=False,training_performed=False,runtime_modified=False,
                  selection_role='source_tune_only',min_patients=30,criterion='patient_paired_ci95_upper_lt_zero_vs_persistence',
                  seeds=[],binding={str(ref_path):sha256(ref_path),str(Path(__file__)):sha256(Path(__file__))})
    for entry in reference['seeds']:
        seed = entry['seed']; partition_path=root/f'partition_{seed}.npz'
        prediction_path=root/f'glucose_retention_predictions_{seed}.npz'
        report['binding'][str(partition_path)]=sha256(partition_path)
        report['binding'][str(prediction_path)]=sha256(prediction_path)
        # Final retention predictions must not influence selection.
        with np.load(prediction_path) as f: initial=f['initial']
        tune=load(partition_path.name)['tune'][valid] & labels['paired'][:,glucose]
        selected=[]; targets={}
        for j,v in enumerate(variables):
            if j==glucose: continue
            m=tune & labels['paired'][:,j]; frame=q.loc[m].copy(); frame['truth']=labels['future'][m,j]
            count=int(frame.subject_id.nunique())
            if count<30:
                targets[v]=dict(patients=count,selected=False,reason='insufficient_source_support');continue
            contrast=paired(frame,initial[m,j],labels['current'][m,j],f'retention_selection:{seed}:{v}',2000)
            keep=bool(contrast['ci95'][1]<0)
            targets[v]=dict(patients=count,selected=keep,contrast=contrast,
                initial_mae=patient_equal_mae(frame,frame.truth,initial[m,j]),
                persistence_mae=patient_equal_mae(frame,frame.truth,labels['current'][m,j]))
            if keep:selected.append(v)
        report['seeds'].append(dict(seed=seed,selected=selected,targets=targets))
        print(seed,selected,flush=True)
    report['complete']=True;output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
