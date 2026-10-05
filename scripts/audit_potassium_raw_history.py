"""Matched source partitions: raw numeric context versus observation support."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from scripts.train_potassium_matched_encoder import three_way_partition
from scripts.audit_event_aware_patient_state import paired
from scripts.audit_recurrent_event_state_loop import patient_equal_mae
from whole_body_jepa.availability_inputs import build_arrays, source_patient_support, restrict_input_support
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def history_features(arrays, rows, col, mode):
    if mode not in ('own','full','support'):raise ValueError('unknown feature mode')
    h=arrays['history'][rows]
    blocks=[]
    for key in ('current','mask','ages'):
        v=arrays[key][h].copy()
        if mode=='own':v=v[:,:,col:col+1]
        if mode=='support' and key=='current':
            keep=v[:,:,col].copy();v[:]=0;v[:,:,col]=keep
        blocks.append(v.reshape(len(rows),-1))
    return np.concatenate(blocks,axis=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trees',action='store_true')
    args=parser.parse_args()
    root=Path('/private/tmp');label=root/'auxiliary_supervision_20260910.npz'
    training=Path('artifacts/potassium_three_way_20260911.json');prior=json.loads(training.read_text())
    raw_paths=[root/'availability_jepa_eicu_labs.parquet',root/'encoder_source_expansion_20260910/labs.parquet']
    for p in [label]+raw_paths:
        if sha256(p)!=prior['binding'][str(p)]:raise ValueError('source changed')
    y=np.load(label);assets=load_release(Path('models'));col=assets.checkpoint['variables'].index('potassium')
    frame=pd.DataFrame(dict(stay_id=y['stay_id'],subject_id=y['subject_id'],t=y['query_time'],truth=y['future'][:,col]))
    raw=pd.concat([pd.read_parquet(p) for p in raw_paths],ignore_index=True)
    raw=raw.loc[raw.stay_id.isin(frame.stay_id)]
    a,rows=build_arrays(frame,raw,assets,{s:pd.Timestamp('2100-01-01') for s in frame.stay_id.unique()})
    valid=y['paired'][:,col];f=frame.loc[valid].reset_index(drop=True);r=rows[valid]
    current=y['current'][valid,col];scale=assets.scaler.scales[col]
    delta=(f.truth.to_numpy()-current)/scale
    protocol=Path('POTASSIUM_TREE_HISTORY_PROTOCOL_20260911.md' if args.trees else 'POTASSIUM_RAW_HISTORY_PROTOCOL_20260911.md')
    report=dict(status='source_diagnostic_only',runtime_modified=False,promotion_allowed=False,external_data_read=False,
        model='hist_gradient_boosting' if args.trees else 'ridge',
        binding={str(p):sha256(p) for p in [label,training,Path(__file__),protocol]},seeds=[])
    for entry in prior['seeds']:
        seed=entry['seed'];fit,tune,evaluation=three_way_partition(f.subject_id,np.ones(len(f),bool),seed)
        original=np.load(root/f'potassium_three_way_20260911/partition_{seed}.npz')
        for key,m in zip(('fit','tune','evaluation'),(fit,tune,evaluation)):np.testing.assert_array_equal(m,original[key][valid])
        full_fit=np.zeros(len(frame),bool);full_fit[valid]=fit
        allowed=source_patient_support(a,rows,frame.subject_id,full_fit)>0
        view=restrict_input_support(a,allowed,assets.model.variable_module_ids.numpy())
        subjects=f.loc[fit,'subject_id'];w=1./subjects.groupby(subjects).transform('size').to_numpy(float)
        w*=len(w)/w.sum()
        predictions={'persistence':current[evaluation]};item=dict(seed=seed,selection={})
        for mode in ('own','full','support'):
            x=history_features(view,r,col,mode).astype(np.float64);best=None;board=[]
            for alpha in ((7,15,31) if args.trees else (1.,10.,100.,1000.,10000.)):
                model=(HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,learning_rate=.05,
                    max_leaf_nodes=alpha,min_samples_leaf=30,l2_regularization=10.,max_bins=63,
                    early_stopping=False,random_state=seed) if args.trees else Ridge(alpha=alpha))
                model.fit(x[fit],delta[fit],sample_weight=w)
                p=current[tune]+scale*model.predict(x[tune]);mae=patient_equal_mae(f.loc[tune],f.loc[tune,'truth'],p)
                board.append({('max_leaf_nodes' if args.trees else 'alpha'):alpha,'tune_mae':mae})
                if best is None or mae<best[0]:best=(mae,alpha,model)
            predictions[mode]=current[evaluation]+scale*best[2].predict(x[evaluation])
            item['selection'][mode]={('max_leaf_nodes' if args.trees else 'alpha'):best[1],'board':board}
        predictions['jepa_full']=np.load(root/f'potassium_three_way_20260911/whole_{seed}_evaluation.npy')
        item['mae']={k:patient_equal_mae(f.loc[evaluation],f.loc[evaluation,'truth'],p) for k,p in predictions.items()}
        item['comparisons']={f'{c}_vs_{b}':paired(f.loc[evaluation],predictions[c],predictions[b],f'raw-history-{seed}-{c}-{b}',1000)
            for c,b in (('full','own'),('full','support'),('full','persistence'),('own','persistence'),('full','jepa_full'))}
        report['seeds'].append(item)
    report['complete']=True
    Path('artifacts/potassium_tree_history_20260911.json' if args.trees else
         'artifacts/potassium_raw_history_20260911.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
