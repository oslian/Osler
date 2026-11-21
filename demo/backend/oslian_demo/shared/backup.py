"""Backup of the two SQLite ledgers, and the check that a backup can be read back.

``python -m oslian_demo.shared.backup --to <folder>`` copies the clinical ledger and the access audit ledger with
SQLite's online backup, which is safe while the server is writing. Each copy is then opened and checked (SQLite's
integrity check, and the audit chain for the audit ledger), and a manifest beside the copies records every file's size
and SHA-256, its row counts, and the audit ledger's last hash.

``--check <manifest>`` repeats that on an existing backup: the restore test. To restore, stop the server and put the
copies where ``OSLIAN_STREAM_DB`` and ``OSLIAN_AUDIT_DB`` point.

The copies are not encrypted and hold what the ledgers hold. The folder must be on an encrypted volume, and a backup
kept on the same machine as the server protects against little.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from ..config import AppConfig
from .audit import verify_chain


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect(path: Path) -> dict:
    """Open a copy the way a restore would and report what it holds. Raises ``ValueError`` if it is damaged."""

    with closing(sqlite3.connect(f"file:{path}?immutable=1", uri=True)) as conn:   # reads it without touching it
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ValueError(f"{path.name}: SQLite integrity check says {result}")
        tables = [name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        entry = {"rows": {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in sorted(tables)}}
        if "access_audit" in entry["rows"]:
            count, head, bad = verify_chain(conn)
            if bad is not None:
                raise ValueError(f"{path.name}: access audit chain broken at row id {bad}")
            entry["audit_head"] = head
    return entry


def back_up(sources: list[Path], folder: Path) -> Path:
    """Copy each existing ledger into ``folder``; returns the manifest's path."""

    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    files = {}
    for source in sources:
        if not source.is_file():
            continue
        copy = folder / f"{source.stem}-{stamp}.sqlite3"
        with closing(sqlite3.connect(source)) as origin, closing(sqlite3.connect(copy)) as target:
            origin.backup(target)
        os.chmod(copy, 0o600)
        files[copy.name] = {"source": str(source), "bytes": copy.stat().st_size, "sha256": _sha256(copy), **_inspect(copy)}
    if not files:
        raise ValueError("no ledger to back up")
    manifest = folder / f"backup-{stamp}.json"
    manifest.write_text(json.dumps({"made_at": stamp, "files": files}, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest, 0o600)
    return manifest


def check(manifest: Path) -> dict:
    """The restore test: every file the manifest names is present, unchanged, readable and holds the same rows."""

    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    for name, entry in recorded["files"].items():
        copy = manifest.parent / name
        if not copy.is_file():
            raise ValueError(f"{name}: missing")
        if _sha256(copy) != entry["sha256"]:
            raise ValueError(f"{name}: the file differs from the one that was backed up")
        found = _inspect(copy)
        if any(found.get(key) != entry.get(key) for key in ("rows", "audit_head")):
            raise ValueError(f"{name}: the rows differ from the manifest")
    return recorded


def main(argv: list[str] | None = None, out=None) -> None:
    parser = argparse.ArgumentParser(description="Back up the ledgers, or check a backup.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--to", type=Path, help="folder to write a new backup into (on an encrypted volume)")
    action.add_argument("--check", type=Path, help="a backup's manifest to verify")
    args = parser.parse_args(argv)
    out = out or sys.stdout
    try:
        if args.check:
            recorded = check(args.check)
            out.write(f"ok {len(recorded['files'])} files from {recorded['made_at']}\n")
        else:
            config = AppConfig.from_env()
            out.write(f"{back_up([Path(config.stream_db), Path(config.audit_db)], args.to)}\n")
    except ValueError as error:
        sys.exit(f"backup: {error}")


if __name__ == "__main__":
    main()
