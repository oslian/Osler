"""Recover ambiguous sample sites without changing any historical tensors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_source_site_alignment import unique_site_map
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.runtime import load_release


def assert_same_example(old, rebuilt):
    if old.keys() != rebuilt.keys():
        raise ValueError('Example schema changed')
    for key in old:
        same = torch.equal(old[key], rebuilt[key]) if isinstance(old[key], torch.Tensor) else old[key] == rebuilt[key]
        if not same:
            raise ValueError(f'Reconstructed example differs: {key}')


def main():
    root = Path('data/research_recovery/broad_source_20260917')
    out = Path('artifacts/native_reconstruction_sample_sites_20260920.json')
    private = root / 'sample_sites_20260920.pt'
    if out.exists() or private.exists():
        raise ValueError('Existing recovery')
    rp = Path('artifacts/native_reconstruction_full_broad_samples_20260918.json')
    report = json.loads(rp.read_text())
    for p, h in report['bindings'].items():
        if sha256(Path(p)) != h:
            raise ValueError(f'Changed source: {p}')
    pp, ip, ep = root/'provenance.parquet', root/'full_samples/index.pt', root/'event_index.pt'
    provenance = pd.read_parquet(pp)
    single, ambiguous = unique_site_map(provenance)
    index = torch.load(ip, weights_only=False)
    shards = {s['patient']: s for s in index['shards']}
    reused = {p for p, s in shards.items() if s['reused']}
    needed = reused | ambiguous
    subset = provenance.loc[provenance.uniquepid.isin(needed)]
    pieces = []
    for s in torch.load(ep, weights_only=False):
        p = Path(s['path'])
        if sha256(p) != s['sha256']:
            raise ValueError('Event shard changed')
        ids = subset.loc[subset.hospitalid.astype(int).eq(int(s['hospitalid'])), 'stay_id'].tolist()
        if ids:
            pieces.append(pd.read_parquet(p, filters=[('stay_id', 'in', ids)]))
    events = pd.concat(pieces, ignore_index=True).drop(columns=['role', 'hospitalid'])
    events = events.merge(subset[['stay_id', 'uniquepid', 'role', 'hospitalid']], on='stay_id', validate='many_to_one')
    cp = Path('artifacts/native_reconstruction_nested_selection_20260917.json')
    catalog = list(json.loads(cp.read_text())['target_counts'])
    age = (events.event_time-events.measured_at).dt.total_seconds()/3600
    events['eligible'] = age.between(0,1) & events.quality_valid.fillna(False).astype(bool) & events.variable.isin(catalog) & np.isfinite(events.value)
    assets = load_release(Path('models'))
    variables = index['variables']
    if variables != assets.checkpoint['variables']:
        raise ValueError('Schema mismatch')
    columns = {v:i for i,v in enumerate(variables)}
    torch.set_num_threads(4)
    rng = np.random.default_rng(20260917)
    resolved = {}
    for patient, history in events.groupby('uniquepid', sort=True):
        candidates = history.loc[history.eligible, ['stay_id','event_time','variable']].drop_duplicates().sort_values(['variable','stay_id','event_time'])
        if patient in reused:
            generator = rng
        else:
            seed = int.from_bytes(hashlib.sha256(('20260918:'+str(patient)).encode()).digest()[:8], 'little')
            generator = np.random.default_rng(seed)
        chosen = [part.iloc[int(generator.integers(len(part)))] for _,part in candidates.groupby('variable',sort=True)]
        if patient not in ambiguous:
            continue
        s = shards[patient]
        if sha256(Path(s['path'])) != s['sha256']:
            raise ValueError('Historical sample changed')
        saved = torch.load(s['path'], weights_only=False)['samples']
        recovered = {}
        if chosen:
            chosen = pd.DataFrame(chosen)
            anchors = chosen[['stay_id','event_time']].drop_duplicates().rename(columns={'event_time':'t'}).sort_values(['stay_id','t']).reset_index(drop=True)
            requested = {(r.stay_id,pd.Timestamp(r.event_time),columns[r.variable]) for r in chosen.itertuples()}
            for e in reconstruction_examples(history.drop(columns=['eligible','uniquepid','role','hospitalid']),anchors,variables,assets.scaler.medians,assets.scaler.scales,assets.model.variable_module_ids.numpy(),len(assets.checkpoint['modules'])):
                anchor = anchors.iloc[e['query_position']]
                c = e['target_index']
                if (anchor.stay_id,pd.Timestamp(anchor.t),c) not in requested:
                    continue
                if c not in saved or c in recovered:
                    raise ValueError('Sample membership changed')
                assert_same_example(saved[c],e)
                sites = subset.loc[subset.stay_id.eq(anchor.stay_id),'hospitalid'].astype(str).unique()
                if len(sites) != 1:
                    raise ValueError('Ambiguous stay site')
                recovered[c] = dict(site=sites[0],stay_id=anchor.stay_id,anchor_time=pd.Timestamp(anchor.t).isoformat())
        if recovered.keys() != saved.keys():
            raise ValueError('Incomplete recovery')
        resolved[patient] = recovered
        print('Verified ambiguous patient',len(resolved),'/',len(ambiguous),'pairs',len(recovered),flush=True)
    if set(resolved) != ambiguous:
        raise ValueError('Missing ambiguous patients')
    paths = [Path(__file__),rp,pp,ip,ep,cp,Path('models/whole_body_joint_jepa.pt'),Path('whole_body_jepa/native_reconstruction_examples.py')]
    bindings = {str(p):sha256(p) for p in paths}
    torch.save(dict(single_site_patients=single,ambiguous_patient_samples=resolved,bindings=bindings),private)
    pairs = sum(map(len,resolved.values()))
    out.write_text(json.dumps(dict(complete=True,ambiguous_patients=len(resolved),verified_ambiguous_pairs=pairs,
        reused_ambiguous_patients=len(ambiguous & reused),singleton_patients=len(single),
        bindings=bindings,private_sha256=sha256(private),runtime_modified=False,promotion_allowed=False,
        limitation='Site provenance repair only. No historical sample, model, prediction, split or gate changed.'),indent=2)+'\n')


if __name__ == '__main__':
    main()
