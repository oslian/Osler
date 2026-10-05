"""Source-only influence diagnosis; never select patients for deployment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.audit_event_aware_patient_state import patient_partition
from whole_body_jepa.multisource_intervals import sha256


def influence(subjects, truth, candidate, reference):
    values=[np.asarray(x,dtype=float) for x in (truth,candidate,reference)]
    if any(x.ndim!=1 or len(x)!=len(subjects) or not np.isfinite(x).all() for x in values):
        raise ValueError('finite aligned predictions required')
    delta=np.abs(values[1]-values[0])-np.abs(values[2]-values[0])
    means=pd.Series(delta,index=np.asarray(subjects).astype(str)).groupby(level=0).mean()
    x=means.to_numpy();n=len(x)
    if n<2:raise ValueError('at least two patients required')
    loo=(x.sum()-x)/(n-1)
    absolute=np.sort(np.abs(x))[::-1]
    total=absolute.sum()
    return dict(patients=n,rows=len(delta),patient_mean=float(x.mean()),row_mean=float(delta.mean()),
        patient_median=float(np.median(x)),benefit_fraction=float((x<0).mean()),
        harm_fraction=float((x>0).mean()),patient_delta_quantiles=np.quantile(x,[.1,.25,.5,.75,.9]).tolist(),
        leave_one_out_mean_range=[float(loo.min()),float(loo.max())],
        any_single_patient_reverses_sign=bool(np.any(np.sign(loo)!=np.sign(x.mean()))),
        top_ten_percent_absolute_contribution=float(absolute[:max(1,int(np.ceil(.1*n)))].sum()/total) if total else 0.)


def main():
    root=Path('/private/tmp');label_path=root/'auxiliary_supervision_20260910.npz'
    labels=np.load(label_path);col=list(labels['variables']).index('potassium')
    f=pd.DataFrame(dict(subject_id=labels['subject_id'],truth=labels['future'][:,col]))
    old=f.subject_id.isin(pd.read_parquet(root/'availability_glucose_development.parquet').subject_id).to_numpy()
    result=dict(status='source_diagnostic_only',runtime_modified=False,promotion_allowed=False,
        external_data_read=False,selection_patients_not_independent_test=True,
        binding={str(label_path):sha256(label_path),str(Path(__file__)):sha256(Path(__file__))},methods={})
    patient_sets=[]
    for name in ('potassium_matched_encoder_20260911','potassium_mae_encoder_20260911','potassium_event_balanced_20260911'):
        report_path=Path('artifacts')/f'{name}.json'
        report=json.loads(report_path.read_text())
        if not report.get('complete'):raise ValueError('unfinished training')
        result['binding'][str(report_path)]=sha256(report_path)
        output=[]
        for item in report['seeds']:
            seed=item['seed'];_,b=patient_partition(f.loc[old,'subject_id'],seed)
            tune=np.zeros(len(f),bool);tune[old]=b;tune&=labels['paired'][:,col]
            work=f.loc[tune];pred={}
            for arm in ('own','whole'):
                path=root/name/f'{arm}_{seed}_tune.npy';pred[arm]=np.load(path)
                result['binding'][str(path)]=sha256(path)
                mae=float(pd.Series(np.abs(pred[arm]-work.truth.to_numpy()),index=work.subject_id).groupby(level=0).mean().mean())
                np.testing.assert_allclose(mae,item['arms'][arm]['tune_mae'],rtol=0,atol=1e-12)
            stats=influence(work.subject_id,work.truth,pred['whole'],pred['own'])
            np.testing.assert_allclose(stats['patient_mean'],item['comparisons']['whole_vs_own']['mean'],atol=1e-12,rtol=0)
            output.append(dict(seed=seed,whole_vs_own=stats))
            if name=='potassium_matched_encoder_20260911':patient_sets.append(set(work.subject_id))
        result['methods'][name]=output
    overlaps=[len(a&b)/len(a|b) for i,a in enumerate(patient_sets) for b in patient_sets[i+1:]]
    result['tune_support']=dict(unique_patients=len(set.union(*patient_sets)),
        split_patient_total=sum(map(len,patient_sets)),pairwise_jaccard_range=[min(overlaps),max(overlaps)],
        all_tune_from_original_cohort=True,expanded_patients_used_for_fitting_only=True)
    Path('artifacts/potassium_patient_influence_20260911.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
