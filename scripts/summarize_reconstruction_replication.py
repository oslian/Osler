"""Aggregate all prespecified splits without treating them as independent studies."""
import json
from pathlib import Path
from whole_body_jepa.multisource_intervals import sha256


def main():
    base=Path('artifacts')
    catalog=json.loads((base/'native_reconstruction_expanded_20260913.json').read_text())['source_eligible']
    sources=[base/'native_reconstruction_expanded_evaluation_20260913.json',
             base/'native_reconstruction_process_evaluation_20260914.json']
    first,process=(json.loads(p.read_text()) for p in sources)
    reports={7:first}
    for role,targets in first['roles'].items():
        for target,item in targets.items():
            other=process['roles'][role][target]
            if abs(item['normalized_mae']['real']-other['normalized_mae']['real'])>1e-7:
                raise ValueError('seed7 real comparison mismatch')
            item['comparisons']['process']=other['comparisons']['process']
    for seed in (19,31,43,59,71,83):
        p=base/f'native_reconstruction_replication_v2_{seed}_evaluation_20260914.json'
        reports[seed]=json.loads(p.read_text());sources.append(p)
    rows={}
    for target in catalog:
        details={}
        for seed,r in reports.items():
            scores=[]
            for role in ('evaluation','hospital'):
                item=r['roles'][role].get(target)
                for reference in ('population','process','placebo'):
                    stat=None if item is None else item['comparisons'][reference]['real']
                    scores.append(dict(role=role,reference=reference,stat=stat,
                        nominal_gain=bool(stat and stat['supported'] and stat['ci95'][1]<0)))
            details[str(seed)]=dict(comparisons=scores,all_comparisons_gain=all(s['nominal_gain'] for s in scores))
        rows[target]=dict(seeds=details,nominal_gain_splits=sum(d['all_comparisons_gain'] for d in details.values()))
    out=base/'native_reconstruction_seven_split_summary_20260917.json'
    if out.exists():raise ValueError('existing summary')
    summary=dict(complete=True,targets=rows,all_seven_nominal=[t for t,v in rows.items() if v['nominal_gain_splits']==7],
        runtime_modified=False,promotion_allowed=False,
        limitation='Same hospital group, overlapping patients, split and optimization seeds confounded; no pooled independent-sample p value or multiplicity-adjusted promotion.',
        bindings={str(p):sha256(p) for p in [Path(__file__),*sources]})
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print('all seven',summary['all_seven_nominal'])
    print('counts',{t:v['nominal_gain_splits'] for t,v in rows.items()})


if __name__=='__main__':main()
