"""Source-only site heterogeneity of fixed reconstruction predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.audit_reconstruction_source_fit import durable
from whole_body_jepa.multisource_intervals import sha256


def main():
    out = Path('artifacts/native_reconstruction_source_hospitals_20260917.json')
    if out.exists():
        raise ValueError('existing evidence')
    root = Path('data/research_recovery/rate_anchored_potassium_20260911/map_native_anchor_support')
    qp = root / 'queries.parquet'
    ip = root / 'expanded_reconstruction_20260913/index.pt'
    q = pd.read_parquet(qp)
    index = torch.load(ip, weights_only=False)
    variables = index['variables']
    if q.hospitalid.isna().any():
        raise ValueError('missing hospital provenance')
    hospital_counts = q.groupby('uniquepid').hospitalid.nunique()
    known_hospitals = q.loc[q.uniquepid.isin(hospital_counts[hospital_counts==1].index)].groupby('uniquepid').hospitalid.first()
    sites = {}
    for shard in index['shards']:
        p = durable(shard['path'])
        if sha256(p) != shard['sha256']:
            raise ValueError('changed shard')
        sample = torch.load(p, weights_only=False)
        for c, e in sample['samples'].items():
            if sample['patient'] not in hospital_counts.index:
                raise ValueError('patient absent from cohort provenance')
            key = (sample['patient'], variables[c])
            if key in sites:
                raise ValueError('duplicate patient-target')
            # query_position belongs to a patient-local anchor table, not q.
            sites[key] = known_hospitals.get(sample['patient'], np.nan)
    bindings = {str(p): sha256(p) for p in (Path(__file__), qp, ip)}
    results = {}
    for seed in (19, 31, 43, 59, 71, 83):
        source_path = Path(f'artifacts/native_reconstruction_source_fit_{seed}_20260917.json')
        eval_path = Path(f'artifacts/native_reconstruction_replication_v2_{seed}_evaluation_20260914.json')
        source, evaluation = [json.loads(p.read_text()) for p in (source_path, eval_path)]
        for p in (source_path, eval_path):
            bindings[str(p)] = sha256(p)
        original_queries = [h for p,h in evaluation['bindings'].items() if Path(p).name == 'queries.parquet']
        original_index = [h for p,h in evaluation['bindings'].items() if Path(p).name == 'index.pt']
        if original_queries != [sha256(qp)] or original_index != [sha256(ip)]:
            raise ValueError('source identity changed')
        def load_rows(report, filename):
            matched = [(durable(p),h) for p,h in report['bindings'].items() if Path(p).name == filename]
            if len(matched) != 1:
                raise ValueError('ambiguous predictions')
            p,h = matched[0]
            if sha256(p) != h:
                raise ValueError('changed saved predictions')
            bindings[str(p)] = h
            return torch.load(p, weights_only=False)
        a = load_rows(source, 'source_predictions.pt')
        b = load_rows(evaluation, 'heldout_predictions.pt')
        roles = {}
        for role, rows in (('tune', a['tune']), ('evaluation', b['evaluation'])):
            frame = pd.DataFrame(rows)
            frame['site'] = [sites[(r.subject_id,r.target)] for r in frame.itertuples()]
            frame['increment'] = abs(frame.real-frame.truth)-abs(frame.process-frame.truth)
            target_results = {}
            for target, part in frame.groupby('target'):
                per_site = part.groupby('site').increment.agg(['count','sum','mean'])
                supported = per_site.loc[per_site['count']>=30]
                pooled = float(part.increment.mean())
                reference = source if role == 'tune' else evaluation
                old = reference['roles'][role][target]['normalized_mae']
                if not np.isclose(pooled,old['real']-old['process'],rtol=0,atol=1e-7):
                    raise ValueError('pooled comparison changed')
                leave_out = [(part.increment.sum()-r['sum'])/(len(part)-r['count'])
                             for _,r in per_site.iterrows() if r['count']<len(part)]
                target_results[target] = dict(patients=len(part), sites=len(per_site),
                    unknown_hospital_patients=int(part.site.isna().sum()),
                    sites_with_30_patients=len(supported),
                    supported_site_patient_counts=sorted(supported['count'].astype(int).tolist()),
                    supported_site_mean_differences=sorted(supported['mean'].tolist()),
                    supported_mean_gain_sites=int((supported['mean']<0).sum()),
                    supported_mean_harm_sites=int((supported['mean']>0).sum()),
                    pooled_difference=pooled,
                    leave_one_site_out_difference_range=[float(min(leave_out)),float(max(leave_out))] if leave_out else None)
            roles[role] = target_results
        results[str(seed)] = roles
    out.write_text(json.dumps(dict(complete=True,seeds=results,bindings=bindings,
        runtime_modified=False,promotion_allowed=False,
        ambiguous_cohort_patients=int((hospital_counts>1).sum()),
        limitation='Source tune/evaluation only. Eight multi-site cohort patients retain unknown site, included in pooled errors but not assigned to a site. Site means descriptive, not significance; support cutoff30 is diagnostic. Leave-one-site-out on fixed predictions is not site-excluded retraining. Sites and patients overlap across splits; no causal attribution or independent replications.'),indent=2,allow_nan=False)+'\n')
    for target in ('albumin','bilirubin','o2sat','triglycerides','creatinine','glucose'):
        print(target,[(s,r['evaluation'].get(target)) for s,r in results.items()],flush=True)


if __name__ == '__main__':
    main()
