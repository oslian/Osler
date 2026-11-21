"""The access audit ledger: who asked for what, about which patient, when, and what the server answered.

One row per API request and per sign-in event, in its own SQLite file so it can be retained and reviewed apart from
the clinical ledger. Rows are append-only: the schema rejects UPDATE and DELETE. Request bodies and query strings are
never recorded; they can hold patient text or a key.

Each row carries a SHA-256 over its own fields and the row before it, so a changed, inserted or removed row breaks
the chain from that point on. That shows tampering inside the file; it does not stop someone who can rewrite the whole
file, which is why the backup records the last hash (``shared/backup.py``) and belongs on another machine.

Review, with ``python -m oslian_demo.shared.audit``:

    --patient <id> (or --actor, --since, --until)   the matching rows as CSV, e.g. who opened a patient's record
    --summary [--since --until]                     one line per user: requests, patients opened, refused requests
    --verify                                        recompute the chain; exit 1 and name the first row that does not fit

Rows are never deleted, so the ledger is as old as the file; keeping six years means keeping the file and its backups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
import threading

from ..config import AppConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS access_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    actor_name  TEXT,
    action      TEXT NOT NULL,
    path        TEXT NOT NULL,
    patient_id  TEXT,
    status      INTEGER NOT NULL,
    remote_addr TEXT,
    chain       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS access_audit_patient ON access_audit (patient_id, at);
CREATE INDEX IF NOT EXISTS access_audit_actor ON access_audit (actor, at);
CREATE TRIGGER IF NOT EXISTS access_audit_no_update BEFORE UPDATE ON access_audit
BEGIN SELECT RAISE(ABORT, 'access_audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS access_audit_no_delete BEFORE DELETE ON access_audit
BEGIN SELECT RAISE(ABORT, 'access_audit is append-only'); END;
"""


COLUMNS = ("id", "at", "actor", "actor_name", "action", "path", "patient_id", "status", "remote_addr")
SUMMARY_COLUMNS = ("actor", "actor_name", "requests", "patients", "refused", "first_at", "last_at")
REFUSED = (401, 403, 429)
GENESIS = "0" * 64


def _link(previous: str, fields: list) -> str:
    return hashlib.sha256((previous + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))).encode()).hexdigest()


def verify_chain(conn: sqlite3.Connection) -> tuple[int, str, int | None]:
    """Recompute the chain. Returns the number of rows that fit, the last hash, and the id of the first row that does
    not fit (``None`` when all do). Removing rows from the end leaves a valid, shorter chain: compare the last hash with
    the one a backup recorded."""

    chain, count = GENESIS, 0
    for row in conn.execute(f"SELECT {', '.join(COLUMNS)}, chain FROM access_audit ORDER BY id"):
        chain = _link(chain, list(row[1:-1]))
        if row[-1] != chain:
            return count, chain, row[0]
        count += 1
    return count, chain, None


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def record(self, *, actor: str, actor_name: str | None, action: str, path: str, patient_id: str | None,
               status: int, remote_addr: str | None) -> None:
        fields = [datetime.now(timezone.utc).isoformat(timespec="milliseconds"), actor, actor_name, action, path,
                  patient_id, status, remote_addr]
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")               # another process may append to the same file
            try:
                last = self._conn.execute("SELECT chain FROM access_audit ORDER BY id DESC LIMIT 1").fetchone()
                self._conn.execute(
                    "INSERT INTO access_audit (at, actor, actor_name, action, path, patient_id, status, remote_addr, chain) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (*fields, _link(last["chain"] if last else GENESIS, fields)))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def rows(self, *, patient_id: str | None = None, actor: str | None = None, since: str | None = None,
             until: str | None = None) -> list[dict]:
        """For review, oldest first. ``since`` (inclusive) and ``until`` (exclusive) are UTC ISO times or dates."""

        where, values = [], []
        for clause, value in (("patient_id = ?", patient_id), ("actor = ?", actor), ("at >= ?", since), ("at < ?", until)):
            if value is not None:
                where.append(clause)
                values.append(value)
        sql = "SELECT * FROM access_audit" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id"
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, values)]

    def summary(self, *, since: str | None = None, until: str | None = None) -> list[dict]:
        """For the regular review: one line per user, busiest first."""

        sql = ("SELECT actor, MAX(actor_name) AS actor_name, COUNT(*) AS requests, COUNT(DISTINCT patient_id) AS patients, "
               f"SUM(status IN ({', '.join(map(str, REFUSED))})) AS refused, MIN(at) AS first_at, MAX(at) AS last_at "
               "FROM access_audit WHERE at >= ? AND at < ? GROUP BY actor ORDER BY requests DESC, actor")
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, (since or "", until or "9999"))]

    def verify(self) -> tuple[int, str, int | None]:
        with self._lock:
            return verify_chain(self._conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()




def main(argv: list[str] | None = None, out=None) -> None:
    parser = argparse.ArgumentParser(description="Review the access audit ledger: rows as CSV, a per-user summary, or a chain check.")
    parser.add_argument("--db", type=Path, default=None, help="the audit ledger (default: OSLIAN_AUDIT_DB)")
    parser.add_argument("--patient")
    parser.add_argument("--actor")
    parser.add_argument("--since", help="UTC ISO time or date, inclusive")
    parser.add_argument("--until", help="UTC ISO time or date, exclusive")
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--summary", action="store_true", help="one line per user instead of one per request")
    what.add_argument("--verify", action="store_true", help="check that no row was changed, inserted or removed")
    args = parser.parse_args(argv)
    path = args.db or AppConfig.from_env().audit_db
    if not Path(path).is_file():
        parser.error(f"no audit ledger at {path}")
    out = out or sys.stdout
    log = AuditLog(path)
    try:
        if args.verify:
            count, head, bad = log.verify()
            if bad is not None:
                sys.exit(f"access audit chain broken at row id {bad} (after {count} good rows)")
            out.write(f"ok rows={count} head={head}\n")
        elif args.summary:
            writer = csv.DictWriter(out, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(log.summary(since=args.since, until=args.until))
        else:
            writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({name: row[name] for name in COLUMNS}
                             for row in log.rows(patient_id=args.patient, actor=args.actor, since=args.since, until=args.until))
    finally:
        log.close()


if __name__ == "__main__":
    main()
