"""Verify and summarize the seven frozen native-query neural comparisons."""
import json
from pathlib import Path
import numpy as np
from whole_body_jepa.multisource_intervals import sha256


def main():
    output = Path('artifacts/map_native_query_training_summary_20260912.json')
    if output.exists():
        raise ValueError('fresh summary required')
    records, bindings = [], {}
    for seed in (7,19,31,43,59,71,83):
        path = Path(f'artifacts/map_native_query_training_{seed}_20260912.json')
        record = json.loads(path.read_text())
        if not record['complete'] or record['seed'] != seed:
            raise ValueError('incomplete seed')
        for name, expected in record['bindings'].items():
            if name not in bindings:
                bindings[name] = sha256(Path(name))
            if bindings[name] != expected:
                raise ValueError('changed dependency')
        bindings[str(path)] = sha256(path)
        records.append(record)
    summary = {}
    for role in ('evaluation','hospital'):
        values = [r['roles'][role] for r in records]
        summary[role] = dict(
            median_mae={k:float(np.median([v['mae'][k] for v in values]))
                        for k in ('persistence','own','full')},
            comparisons={k:dict(
                median_paired_delta=float(np.median([v['comparisons'][k]['mean'] for v in values])),
                significant_gain=sum(v['comparisons'][k]['ci95'][1]<0 for v in values),
                significant_harm=sum(v['comparisons'][k]['ci95'][0]>0 for v in values))
                for k in ('full_vs_own','full_vs_persistence','own_vs_persistence')})
    result = dict(complete=True, seeds=7, fits=14, summary=summary,
        runtime_modified=False, status='candidate_only',
        selected_epochs={a:[r['arms'][a]['selected_epoch'] for r in records] for a in ('own','full')},
        scope='repeated source/hospital diagnostic, no independent lockbox or promotion',
        bindings={**bindings,str(Path(__file__)):sha256(Path(__file__))})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='bindings'})


if __name__ == '__main__':
    main()
