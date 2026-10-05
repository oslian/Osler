"""Outcome-blind, role-unassigned anchor selection before masked-history checks."""
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import torch

from whole_body_jepa.multisource_intervals import sha256


def priority(patient, variable, stay, timestamp):
    value = f'20260921|{patient}|{variable}|{stay}|{pd.Timestamp(timestamp).isoformat()}'
    return hashlib.md5(value.encode()).hexdigest()


def choose_anchors(candidates):
    """Stable uniform hash-priority sample; no values, roles or future outcomes."""
    keys = ['uniquepid', 'variable', 'stay_id', 't']
    result = candidates[keys].drop_duplicates().copy()
    result['priority'] = [priority(*row) for row in result.itertuples(index=False, name=None)]
    return result.sort_values(['priority', *keys], kind='stable').drop_duplicates(
        ['uniquepid', 'variable']).reset_index(drop=True)


def main():
    root = Path('data/research_recovery/additional_development_domains_20260921')
    out = Path('artifacts/native_reconstruction_additional_domain_anchors_20260921.json')
    folder = root/'anchors'
    if out.exists() or folder.exists():
        raise ValueError('Existing anchor selection')
    pp = root/'provenance.parquet'; provenance = pd.read_parquet(pp)
    provenance['stay_id'] = 'eicu:' + provenance.patientunitstayid
    cp = Path('models/whole_body_joint_jepa.pt')
    variables = torch.load(cp, weights_only=False)['variables']
    bindings = {str(p):sha256(p) for p in (Path(__file__), pp, cp)}
    indices = {}
    for kind in ('labs', 'bedside'):
        rp = Path(f'artifacts/native_reconstruction_additional_domain_{kind}_20260921.json')
        report = json.loads(rp.read_text())
        if report['bindings'][str(pp)] != sha256(pp):
            raise ValueError('Changed event provenance')
        ip = root/kind/'event_index.pt'
        if sha256(ip) != report['private_index_sha256']:
            raise ValueError('Changed event index')
        index = torch.load(ip, weights_only=False)
        for item in index:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError('Changed event shard')
        indices[kind] = {s['hospitalid']:s for s in index}
        bindings.update({str(p):sha256(p) for p in (rp, ip)})
    folder.mkdir()
    sites = sorted(provenance.hospitalid.unique())
    db = duckdb.connect(); db.execute('SET threads=2'); db.execute("SET memory_limit='2GB'")
    db.register('variables', pd.DataFrame({'variable':variables}))
    shard_index = []; proposed = []
    for number, site in enumerate(sites):
        paths = [indices[kind][site]['path'] for kind in indices]
        db.read_parquet(paths, union_by_name=True).create_view('events', replace=True)
        candidates = db.sql("""SELECT DISTINCT uniquepid,variable,stay_id,event_time AS t
            FROM events JOIN variables USING(variable)
            WHERE quality_valid AND isfinite(value)
                AND event_time=available_at AND measured_at<=available_at
                AND available_at-measured_at<=INTERVAL '1 hour'
                AND source_kind IN ('lab_available','bedside_available')""").fetchdf()
        # Per-site minima can be combined exactly into the global patient minimum.
        chosen = choose_anchors(candidates)
        destination = folder/f'candidates_{number:03d}.parquet'
        chosen.to_parquet(destination, index=False)
        shard_index.append(dict(path=str(destination), sha256=sha256(destination), hospitalid=site))
        proposed.append(chosen)
    db.close()
    anchors = pd.concat(proposed, ignore_index=True).sort_values(
        ['priority','uniquepid','variable','stay_id','t'], kind='stable').drop_duplicates(['uniquepid','variable'])
    anchors = anchors.merge(provenance[['stay_id','uniquepid','hospitalid','component']],
                            on=['stay_id','uniquepid'], validate='many_to_one')
    if anchors[['hospitalid','component']].isna().any().any():
        raise ValueError('Missing anchor provenance')
    path = folder/'queries.parquet'; anchors.to_parquet(path,index=False)
    index_path = folder/'site_candidate_index.pt'; torch.save(shard_index,index_path)
    counts = {}
    for target in variables:
        part = anchors.loc[anchors.variable.eq(target)]
        domains = part.groupby('component').uniquepid.nunique()
        counts[target] = dict(proposed_pairs=len(part), components=len(domains),
                             components_at_least30=int((domains>=30).sum()))
    summary = dict(candidate_patients=int(provenance.uniquepid.nunique()),
        proposed_patients=int(anchors.uniquepid.nunique()), proposed_pairs=len(anchors),
        proposed_targets=int(anchors.variable.nunique()),
        duplicate_patient_targets=int(anchors.duplicated(['uniquepid','variable']).sum()))
    out.write_text(json.dumps(dict(summary=summary,target_counts=counts,bindings=bindings,
        private_queries_sha256=sha256(path),private_index_sha256=sha256(index_path),
        sampling='Fixed MD5 priority of seed|patient|target|stay|timestamp, one per patient-target across all stays/sites; no values/outcomes/model errors, no redraw after later rejection.',
        roles_assigned=False,masked_history_accepted=False,runtime_modified=False,promotion_allowed=False,
        limitation='Candidate anchors only. Subsequent source-version conflict replay, target/sibling masking, independent nonempty history and fit-only vocabulary checks may reject anchors. Whole-stay insulin flags are not selection inputs. Checkpoint supplies schema only, no learned weights or scalers.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
