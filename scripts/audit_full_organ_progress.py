"""Snapshot all E25 extraction roles without treating partial work as efficacy."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from scripts.cache_full_organ_compartments import DEST, ROOT, RESERVE, validate_cache
from scripts.score_monitored_continuation_outer import verify_bound_files
from whole_body_jepa.multisource_intervals import sha256


def prefix_digest(db, count):
    rows = list(db.execute('SELECT digest FROM compartments ORDER BY patient,target LIMIT ?', (count,)))
    if len(rows) != count:
        raise ValueError('Verified prefix is missing')
    digest = hashlib.sha256()
    for value, in rows:
        digest.update(value.encode()+b'\n')
    return digest.hexdigest()


def progress_summary(rows, disk_free):
    if {(r['fold'], r['role']) for r in rows} != {(f, r) for f in range(5) for r in ('fit', 'selection')} or len(rows) != 10:
        raise ValueError('All ten extraction roles required')
    for r in rows:
        if not (0 <= r['stored'] <= r['expected']) or r['complete'] != (r['stored'] == r['expected']):
            raise ValueError('Invalid extraction completion status')
    total = sum(r['expected'] for r in rows)
    stored = sum(r['stored'] for r in rows)
    started = [r for r in rows if r['stored']]
    bytes_per_row = max((r['bytes']/r['stored'] for r in started), default=None)
    timings = [r for r in started if r.get('added', 0)]
    # The most recent measured batch includes startup/hash costs. This is a
    # rough planning estimate, not a representative timing confidence bound.
    seconds_per_row = max((r['wall_seconds']/r['added'] for r in timings), default=None)
    remaining_bytes = None if bytes_per_row is None else (total-stored)*bytes_per_row
    return dict(expected=total, stored=stored, remaining=total-stored,
        completed_roles=sum(r['complete'] for r in rows), complete=all(r['complete'] for r in rows),
        estimated_remaining_bytes=remaining_bytes,
        estimated_remaining_hours=None if seconds_per_row is None else (total-stored)*seconds_per_row/3600,
        disk_free_bytes=disk_free, reserve_bytes=RESERVE,
        estimated_space_sufficient=None if remaining_bytes is None else disk_free-remaining_bytes >= RESERVE,
        estimate_limitation='Largest observed role byte/time rate; batches are ordered prefixes, not representative samples. Runtime reserve remains enforced.')


def main():
    pilot_path = Path('artifacts/native_reconstruction_full_organs_pilot_verification_20261003.json')
    pilot = json.loads(pilot_path.read_text())
    checked = {}
    verify_bound_files(pilot['bindings'], checked)
    rows, snapshots = [], {}
    for fold in range(5):
        for role in ('fit', 'selection'):
            rp = Path(f'artifacts/native_reconstruction_full_organs_fold{fold}_{role}_20261003.json')
            reference_report = Path(f'artifacts/native_reconstruction_nonlinear_cache_fold{fold}_{role}_20261001.json')
            parent = json.loads(reference_report.read_text())
            expected = parent['examples']
            path = DEST/f'fold{fold}_{role}.sqlite'
            if not rp.exists():
                if path.exists():
                    raise ValueError('Unreported partial cache: resume its finite batch before snapshotting')
                rows.append(dict(fold=fold, role=role, expected=expected, stored=0, complete=False, bytes=0))
                continue
            report = json.loads(rp.read_text())
            contract = report['contract']
            if (contract['fold'], contract['role']) != (fold, role):
                raise ValueError('Wrong role report')
            verify_bound_files(contract['bindings'], checked)
            if sha256(path) != report['private_sha256']:
                raise ValueError('Cache has unreported modifications')
            reference = ROOT/f'nonlinear_probe_20261001/fold{fold}_{role}.sqlite'
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
                db.execute('ATTACH DATABASE ? AS reference', (reference.resolve().as_uri()+'?mode=ro',))
                stored_contract = json.loads(db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0])
                if stored_contract != contract:
                    raise ValueError('Report/cache contract mismatch')
                status = validate_cache(db, contract['organ_shape'])
                if status['expected'] != expected or any(status[k] != report[k] for k in status):
                    raise ValueError('Incorrect cache completeness report')
                if fold == 0 and prefix_digest(db, 512) != pilot['results'][role]['prefix_digest']:
                    raise ValueError('Independently verified pilot prefix changed')
                errors = db.execute('SELECT MAX(body_error),MAX(point_error) FROM compartments').fetchone()
                if errors != (report['body_max_error'], report['point_max_error']):
                    raise ValueError('Parity diagnostics changed')
            rows.append(dict(fold=fold, role=role, **status, bytes=path.stat().st_size,
                added=report['added'], wall_seconds=report['wall_seconds'],
                body_max_error=errors[0], point_max_error=errors[1]))
            snapshots[f'{fold}_{role}'] = report
    summary = progress_summary(rows, shutil.disk_usage(DEST).free)
    output = Path(f"artifacts/native_reconstruction_full_organs_progress_{summary['stored']}_20261003.json")
    if output.exists():
        raise ValueError('This progress snapshot already exists')
    output.write_text(json.dumps(dict(summary=summary, rows=rows, extraction_snapshots=snapshots,
        pilot_sha256=sha256(pilot_path), source_sha256=sha256(Path(__file__)), checked_files=len(checked),
        runtime_modified=False, promotion_allowed=False,
        limitation='Engineering progress snapshot only. Incomplete caches cannot train or score full E25. No new model efficacy claim.'),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
