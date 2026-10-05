"""Match selected-source samples to raw stay histories, without resampling."""
import json
from pathlib import Path

import pandas as pd
import torch

from scripts.recover_source_sample_sites import assert_same_example
from scripts.audit_reconstruction_source_fit import durable, BACKUP
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.runtime import load_release


def unique_match(matches):
    if len(matches) != 1:
        raise ValueError('Sample must have exactly one verified stay match')
    return matches[0]


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_selected_sites_20260920.json')
    dest = root/'selected_sample_sites_20260920.pt'
    if out.exists() or dest.exists():
        raise ValueError('Existing recovery')
    manifest = root/'joint_source_manifest.pt'
    m = torch.load(manifest,weights_only=False)
    shards = {s['patient']:s for s in m['shards'] if s['group']=='selected'}
    qp = BACKUP/'map_native_anchor_support/queries.parquet'
    q = pd.read_parquet(qp)
    links = q.loc[q.uniquepid.isin(shards),['stay_id','uniquepid','hospitalid']].drop_duplicates()
    if links.stay_id.duplicated().any() or set(links.uniquepid)!=set(shards):
        raise ValueError('Stay identity mismatch')
    sites = links.groupby('uniquepid').hospitalid.agg(lambda x:set(map(str,x)))
    ambiguous = {p for p,s in sites.items() if len(s)>1}
    rawroot = Path('/private/tmp/rate_anchored_potassium_20260911')
    paths = [rawroot/'native_events.parquet',rawroot/'qualified_bedside_events.parquet']
    original_report = Path('artifacts/native_reconstruction_expanded_20260913.json')
    original = json.loads(original_report.read_text())
    for p in paths:
        if sha256(p) != original['bindings'][str(p)]:
            raise ValueError('Original raw history changed')
    ids = links.loc[links.uniquepid.isin(ambiguous),'stay_id'].tolist()
    lab = pd.read_parquet(paths[0],filters=[('stay_id','in',ids)])
    lab['quality_valid'] = True
    bedside = pd.read_parquet(paths[1],filters=[('stay_id','in',ids)])
    bedside['event_time'] = bedside.available_at
    events = pd.concat([lab,bedside],ignore_index=True)
    events = events.loc[events.source_kind.isin(['lab_available','bedside_available'])]
    assets = load_release(Path('models'));variables=m['variables']
    if variables!=assets.checkpoint['variables']:
        raise ValueError('Variable mismatch')
    torch.set_num_threads(4);resolved={}
    for patient in sorted(ambiguous):
        shard=shards[patient];p=durable(shard['path'])
        if sha256(p)!=shard['sha256']:
            raise ValueError('Sample changed')
        samples=torch.load(p,weights_only=False)['samples'];per={}
        patientlinks=links.loc[links.uniquepid.eq(patient)]
        history=events.loc[events.stay_id.isin(patientlinks.stay_id)]
        for c,old in samples.items():
            # Match actual raw timestamps, avoiding float-hour roundtrip drift.
            possible=history.loc[history.variable.eq(variables[c]),['stay_id','event_time']].drop_duplicates()
            hours=(possible.event_time-pd.Timestamp('2100-01-01')).dt.total_seconds()/3600
            possible=possible.loc[hours.eq(old['anchor_hour'])]
            matches=[]
            for anchor in possible.itertuples():
                query=pd.DataFrame([dict(stay_id=anchor.stay_id,t=anchor.event_time)])
                for new in reconstruction_examples(history,query,variables,assets.scaler.medians,assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
                    if new['target_index']!=c:
                        continue
                    # Query index is local to the one-query replay, not time.
                    new['query_position']=old['query_position']
                    try:
                        assert_same_example(old,new)
                    except ValueError:
                        continue
                    site=unique_match(patientlinks.loc[patientlinks.stay_id.eq(anchor.stay_id),'hospitalid'].astype(str).tolist())
                    matches.append(dict(site=site,stay_id=anchor.stay_id,anchor_time=pd.Timestamp(anchor.event_time).isoformat()))
            per[c]=unique_match(matches)
        resolved[patient]=per
        print('Verified selected multi-site patient',len(resolved),'pairs',len(per),flush=True)
    bindings={str(p):sha256(p) for p in [Path(__file__),manifest,qp,original_report,*paths,Path('models/whole_body_joint_jepa.pt')]}
    torch.save(dict(single_site_patients={p:next(iter(s)) for p,s in sites.items() if len(s)==1},ambiguous_patient_samples=resolved,bindings=bindings),dest)
    out.write_text(json.dumps(dict(complete=True,source_patients=len(shards),ambiguous_patients=len(ambiguous),verified_pairs=sum(map(len,resolved.values())),bindings=bindings,private_sha256=sha256(dest),runtime_modified=False,promotion_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
