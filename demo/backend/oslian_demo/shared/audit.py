"""The access audit ledger: who asked for what, about which patient, when, and what the server answered.

One row per API request and per sign-in event, in its own SQLite file so it can be retained and reviewed apart from
the clinical ledger. Rows are append-only: the schema rejects UPDATE and DELETE. Request bodies and query strings are
never recorded; they can hold patient text or a key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading

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
    remote_addr TEXT
);
CREATE INDEX IF NOT EXISTS access_audit_patient ON access_audit (patient_id, at);
CREATE INDEX IF NOT EXISTS access_audit_actor ON access_audit (actor, at);
CREATE TRIGGER IF NOT EXISTS access_audit_no_update BEFORE UPDATE ON access_audit
BEGIN SELECT RAISE(ABORT, 'access_audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS access_audit_no_delete BEFORE DELETE ON access_audit
BEGIN SELECT RAISE(ABORT, 'access_audit is append-only'); END;
"""


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
        at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO access_audit (at, actor, actor_name, action, path, patient_id, status, remote_addr) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (at, actor, actor_name, action, path, patient_id, status, remote_addr))

    def rows(self, *, patient_id: str | None = None, actor: str | None = None) -> list[dict]:
        """For review and tests, oldest first."""

        where, values = [], []
        for column, value in (("patient_id", patient_id), ("actor", actor)):
            if value is not None:
                where.append(f"{column} = ?")
                values.append(value)
        sql = "SELECT * FROM access_audit" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id"
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, values)]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
