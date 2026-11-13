"""Full fixed-anchor raw history cache; resumable hospital transactions."""
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.reconstruction_cache import ReconstructionCache


def main():
    root=Path('data/research_recovery/additional_development_domains_20260921')
    output=Path('artifacts/native_reconstruction_additional_full_cache_20260922.json')
    if output.exists():raise ValueError('Completed report already exists')
    sr=Path('artifacts/native_reconstruction_additional_domain_scaling_20260922.json')
    mr=Path('artifacts/native_reconstruction_additional_masked_history_full_20260921.json')
    sp=root/'scaling_20260922.pt';mp=root/'masked_history_full/index.pt'
    reports=[json.loads(p.read_text()) for p in (sr,mr)]
    for report in reports:
        for p,h in report['bindings'].items():
            if sha256(Path(p))!=h:raise ValueError('Changed upstream dependency')
    if sha256(sp)!=reports[0]['private_sha256'] or sha256(mp)!=reports[1]['private_index_sha256']:
        raise ValueError('Changed private inputs')
    scaling=torch.load(sp,weights_only=False);variables=scaling['variables'];n=len(variables)
    people={p:i for i,p in enumerate(scaling['people'])}
    cp=Path('models/whole_body_joint_jepa.pt');schema=torch.load(cp,weights_only=False)
    ids=schema['state_dict']['variable_module_ids'].numpy()
    if variables!=schema['variables']:raise ValueError('Schema mismatch')
    indices={kind:{s['hospitalid']:s for s in torch.load(root/kind/'event_index.pt',weights_only=False)}
             for kind in ('labs','bedside')}
    paths=[Path(__file__),sr,mr,sp,mp,cp,Path('whole_body_jepa/reconstruction_cache.py')]
    paths+=list(Path('whole_body_jepa').glob('native_*.py'))
    paths += [Path('whole_body_jepa/nowcast.py'),Path('whole_body_jepa/history_event_input.py')]
    bindings={str(p):sha256(p) for p in paths}
    cache_path=root/'full_histories_20260922.sqlite'
    cache=ReconstructionCache(cache_path,bindings)
    completed={row[0] for row in cache.db.execute('SELECT shard FROM completed')}
    torch.set_num_threads(2);started=time.monotonic();manifest=torch.load(mp,weights_only=False)
    try:
        for shard in manifest:
            site=str(shard['hospitalid'])
            if sha256(Path(shard['path']))!=shard['sha256']:raise ValueError('Changed decisions')
            if site in completed:continue
            if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('Less than 5GiB free; resume after capacity repair')
            q=pd.read_parquet(shard['path']);q=q.loc[q.status.eq('accepted')]
            if q.empty:
                cache.finish_shard(site);continue
            parts=[]
            for kind in indices:
                source=indices[kind][shard['hospitalid']]
                if sha256(Path(source['path']))!=source['sha256']:raise ValueError('Changed source shard')
                parts.append(pd.read_parquet(source['path'],filters=[('stay_id','in',q.stay_id.unique().tolist())]))
            events=pd.concat(parts,ignore_index=True)
            grouped={s:g for s,g in events.groupby('stay_id',sort=False)}
            built=0;unsupported=np.zeros((len(scaling['scalers']),n),dtype=np.int64)
            for stay,requests in q.groupby('stay_id',sort=False):
                requests=requests.sort_values(['t','variable']).reset_index(drop=True);count=0
                for e in reconstruction_examples(grouped[stay],requests,variables,np.zeros(n),np.ones(n),ids,len(schema['modules'])):
                    row=requests.iloc[e['query_position']];col=e['target_index']
                    if variables[col]!=row.variable:continue
                    expected=scaling['centers'][people[row.uniquepid],col]
                    np.testing.assert_allclose(e['truth_normalized'],expected,rtol=1e-6,atol=1e-6)
                    mask=e['tokens'][:,n:2*n].bool()
                    forbidden={row.variable}|related_siblings(row.variable)
                    if any(mask[:,i].any() for i,v in enumerate(variables) if v in forbidden):
                        raise ValueError('Target/sibling leakage')
                    if len(mask)!=row.independent_event_groups:raise ValueError('History count mismatch')
                    observed=mask.any(0).numpy()
                    for fold,scaler in enumerate(scaling['scalers']):
                        unsupported[fold]+=observed & (scaler['counts']==0)
                    cache.add(row.uniquepid,e);count+=1;built+=1
                if count!=len(requests):raise ValueError('Lost accepted anchors')
                if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('Less than 5GiB free; current shard will roll back')
            cache.db.execute('INSERT INTO metadata VALUES (?,?)',
                ('support:'+site,json.dumps(dict(examples=built,unsupported=unsupported.tolist()))))
            cache.finish_shard(site)
            print('Completed hospital shard',len(completed)+1,'of',len(manifest),'examples',built,flush=True)
            completed.add(site)
        total=cache.db.execute('SELECT count(*),count(DISTINCT patient) FROM examples').fetchone()
        if total[0]!=reports[0]['accepted_labels'] or total[1]!=reports[0]['people_with_labels']:
            raise ValueError('Full cache count mismatch')
        integrity=cache.db.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity!='ok':raise ValueError('SQLite integrity check failed')
        support=np.zeros((len(scaling['scalers']),n),dtype=np.int64)
        for (value,) in cache.db.execute("SELECT value FROM metadata WHERE key LIKE 'support:%'"):
            support+=np.asarray(json.loads(value)['unsupported'])
    finally:
        cache.close()
    result=dict(examples=total[0],patients=total[1],completed_hospitals=len(completed),
        unsupported_input_examples_by_fold=[{v:int(row[i]) for i,v in enumerate(variables) if row[i]} for row in support],
        cache_bytes=cache_path.stat().st_size,private_sha256=sha256(cache_path),bindings=bindings,
        invocation_seconds=time.monotonic()-started,sqlite_integrity=integrity,
        runtime_modified=False,promotion_allowed=False,
        limitation='Full raw-unit current reconstruction cache; not training or future forecast. Fit-label normalization support is checked against all actual histories; unsupported inputs require fold-local masking before training. No external lockbox claim.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
