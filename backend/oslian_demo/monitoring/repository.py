"""SQLite (WAL) repository for the streaming demo.

All SQL lives here. The raw FHIR ledger is append-only and unique on
(source_system, resource_type, resource_id, version_id); corrected resources
arrive as new versions and never overwrite history. Playback sessions record
*deliveries* of ledger rows per epoch, so a stream reset replays the feed
without deleting or duplicating ledger history.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    resource_json TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS encounters (
    encounter_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    period_start TEXT NOT NULL,
    resource_json TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_fhir_resources (
    raw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_system TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    patient_id TEXT,
    source_last_updated TEXT,
    first_received_at TEXT NOT NULL,
    first_event_id TEXT,
    resource_json TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    UNIQUE (source_system, resource_type, resource_id, version_id)
);
CREATE TABLE IF NOT EXISTS canonical_observations (
    canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id INTEGER NOT NULL REFERENCES raw_fhir_resources(raw_id),
    patient_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    observation_status TEXT NOT NULL,
    variable TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    effective_time TEXT NOT NULL,
    issued TEXT,
    source_last_updated TEXT,
    received_at TEXT NOT NULL,
    mapping_version TEXT NOT NULL,
    rule_key TEXT NOT NULL,
    input_value REAL NOT NULL,
    input_unit TEXT NOT NULL,
    derivation TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_canonical_patient_time ON canonical_observations(patient_id, effective_time);
CREATE TABLE IF NOT EXISTS mapping_failures (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id INTEGER NOT NULL REFERENCES raw_fhir_resources(raw_id),
    patient_id TEXT,
    reason TEXT NOT NULL,
    detail TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stream_sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    fixture_version TEXT NOT NULL,
    status TEXT NOT NULL,
    speed REAL NOT NULL,
    epoch INTEGER NOT NULL,
    cursor INTEGER NOT NULL,
    sim_elapsed_seconds REAL NOT NULL,
    last_event_sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_cursors (
    session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
    epoch INTEGER NOT NULL,
    source_system TEXT NOT NULL,
    source_sequence INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, epoch)
);
CREATE TABLE IF NOT EXISTS session_deliveries (
    session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
    epoch INTEGER NOT NULL,
    raw_id INTEGER NOT NULL REFERENCES raw_fhir_resources(raw_id),
    source_sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (session_id, epoch, raw_id)
);
CREATE TABLE IF NOT EXISTS stream_events (
    session_id TEXT NOT NULL REFERENCES stream_sessions(session_id),
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    emitted_at TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence)
);
CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    patient_id TEXT NOT NULL,
    status TEXT NOT NULL,
    runtime_mode TEXT NOT NULL,
    model_commit TEXT,
    registry_sha256 TEXT,
    release_date TEXT,
    input_cutoff TEXT NOT NULL,
    input_watermark TEXT,
    input_resource_count INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    input_refs_json TEXT NOT NULL,
    trigger_event_ids_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER,
    error TEXT,
    summary_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_patient ON model_runs(patient_id, started_at);
CREATE TABLE IF NOT EXISTS forecast_cells (
    run_id TEXT NOT NULL REFERENCES model_runs(run_id),
    cell TEXT NOT NULL,
    target TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL,
    unit TEXT,
    observed_value REAL,
    measurement_age_hours REAL,
    observations INTEGER NOT NULL,
    forecast_point REAL,
    source_model TEXT NOT NULL,
    status TEXT NOT NULL,
    fallback_reason TEXT,
    interval_status TEXT NOT NULL,
    lower REAL,
    upper REAL,
    causal_claim_allowed INTEGER NOT NULL,
    PRIMARY KEY (run_id, cell)
);
"""


@dataclass(frozen=True)
class RawInsert:
    raw_id: int
    created: bool


class Repository:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # One shared connection guarded by a re-entrant lock: SQLite serialises
        # writers anyway and this keeps ":memory:" databases usable in tests.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(SCHEMA)

    # ---- generic helpers -------------------------------------------------
    @staticmethod
    def _bind(params: Iterable[Any] | dict[str, Any]) -> tuple[Any, ...] | dict[str, Any]:
        return params if isinstance(params, dict) else tuple(params)

    def _rows(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, self._bind(params)).fetchall()

    def _one(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, self._bind(params)).fetchone()

    def _exec(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, self._bind(params))

    def transaction(self):
        return _Transaction(self)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- patients / encounters ------------------------------------------
    def upsert_patient(self, patient_id: str, source_system: str, resource: dict, received_at: str) -> None:
        self._exec(
            "INSERT INTO patients(patient_id, source_system, resource_json, received_at) VALUES (?,?,?,?) "
            "ON CONFLICT(patient_id) DO UPDATE SET resource_json=excluded.resource_json",
            (patient_id, source_system, json.dumps(resource, sort_keys=True), received_at),
        )

    def upsert_encounter(self, encounter_id: str, patient_id: str, period_start: str, resource: dict,
                         received_at: str) -> None:
        self._exec(
            "INSERT INTO encounters(encounter_id, patient_id, period_start, resource_json, received_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(encounter_id) DO UPDATE SET resource_json=excluded.resource_json, "
            "period_start=excluded.period_start",
            (encounter_id, patient_id, period_start, json.dumps(resource, sort_keys=True), received_at),
        )

    def patient(self, patient_id: str) -> dict | None:
        row = self._one("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
        return {**dict(row), "resource": json.loads(row["resource_json"])} if row else None

    def patients(self) -> list[dict]:
        return [{**dict(r), "resource": json.loads(r["resource_json"])}
                for r in self._rows("SELECT * FROM patients ORDER BY patient_id")]

    def encounter_for_patient(self, patient_id: str) -> dict | None:
        row = self._one("SELECT * FROM encounters WHERE patient_id=? ORDER BY period_start LIMIT 1", (patient_id,))
        return {**dict(row), "resource": json.loads(row["resource_json"])} if row else None

    # ---- raw ledger --------------------------------------------------------
    def insert_raw(self, *, source_system: str, resource_type: str, resource_id: str, version_id: str,
                   patient_id: str | None, source_last_updated: str | None, received_at: str,
                   event_id: str | None, resource: dict, validation_status: str) -> RawInsert:
        with self._lock:
            existing = self._one(
                "SELECT raw_id FROM raw_fhir_resources WHERE source_system=? AND resource_type=? "
                "AND resource_id=? AND version_id=?",
                (source_system, resource_type, resource_id, version_id),
            )
            if existing:
                return RawInsert(int(existing["raw_id"]), False)
            cursor = self._exec(
                "INSERT INTO raw_fhir_resources(source_system, resource_type, resource_id, version_id, patient_id, "
                "source_last_updated, first_received_at, first_event_id, resource_json, validation_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (source_system, resource_type, resource_id, version_id, patient_id, source_last_updated,
                 received_at, event_id, json.dumps(resource, sort_keys=True), validation_status),
            )
            return RawInsert(int(cursor.lastrowid), True)

    def raw(self, raw_id: int) -> dict | None:
        row = self._one("SELECT * FROM raw_fhir_resources WHERE raw_id=?", (raw_id,))
        return {**dict(row), "resource": json.loads(row["resource_json"])} if row else None

    def raw_by_reference(self, source_system: str, resource_type: str, resource_id: str,
                         version_id: str) -> dict | None:
        row = self._one(
            "SELECT * FROM raw_fhir_resources WHERE source_system=? AND resource_type=? AND resource_id=? "
            "AND version_id=?", (source_system, resource_type, resource_id, version_id))
        return {**dict(row), "resource": json.loads(row["resource_json"])} if row else None

    def raw_count(self) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM raw_fhir_resources")["n"])

    # ---- canonical / failures ---------------------------------------------
    def insert_canonical(self, raw_id: int, patient_id: str, resource_id: str, version_id: str,
                         observation_status: str, *, variable: str, value: float, unit: str,
                         effective_time: str, issued: str | None, source_last_updated: str | None,
                         received_at: str, mapping_version: str, rule_key: str, input_value: float,
                         input_unit: str, derivation: str) -> int:
        cursor = self._exec(
            "INSERT INTO canonical_observations(raw_id, patient_id, resource_id, version_id, observation_status, "
            "variable, value, unit, effective_time, issued, source_last_updated, received_at, mapping_version, "
            "rule_key, input_value, input_unit, derivation) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (raw_id, patient_id, resource_id, version_id, observation_status, variable, value, unit,
             effective_time, issued, source_last_updated, received_at, mapping_version, rule_key,
             input_value, input_unit, derivation),
        )
        return int(cursor.lastrowid)

    def canonical_for_raw(self, raw_id: int) -> list[dict]:
        return [dict(r) for r in self._rows("SELECT * FROM canonical_observations WHERE raw_id=?", (raw_id,))]

    def insert_mapping_failure(self, raw_id: int, patient_id: str | None, reason: str, detail: str,
                               recorded_at: str) -> None:
        self._exec(
            "INSERT INTO mapping_failures(raw_id, patient_id, reason, detail, recorded_at) VALUES (?,?,?,?,?)",
            (raw_id, patient_id, reason, detail, recorded_at),
        )

    def mapping_failures(self, patient_id: str) -> list[dict]:
        return [dict(r) for r in self._rows(
            "SELECT f.*, r.resource_type, r.resource_id, r.version_id FROM mapping_failures f "
            "JOIN raw_fhir_resources r ON r.raw_id=f.raw_id WHERE f.patient_id=? ORDER BY f.failure_id",
            (patient_id,))]

    # ---- sessions -----------------------------------------------------------
    def create_session(self, session: dict) -> None:
        self._exec(
            "INSERT INTO stream_sessions(session_id, patient_id, source_system, fixture_version, status, speed, "
            "epoch, cursor, sim_elapsed_seconds, last_event_sequence, created_at, updated_at) "
            "VALUES (:session_id,:patient_id,:source_system,:fixture_version,:status,:speed,:epoch,:cursor,"
            ":sim_elapsed_seconds,:last_event_sequence,:created_at,:updated_at)",
            session,
        )

    def update_session(self, session_id: str, **fields: Any) -> None:
        assignments = ", ".join(f"{key}=:{key}" for key in fields)
        self._exec(f"UPDATE stream_sessions SET {assignments} WHERE session_id=:session_id",
                   {**fields, "session_id": session_id})

    def session(self, session_id: str) -> dict | None:
        row = self._one("SELECT * FROM stream_sessions WHERE session_id=?", (session_id,))
        return dict(row) if row else None

    def sessions(self) -> list[dict]:
        return [dict(r) for r in self._rows("SELECT * FROM stream_sessions ORDER BY created_at")]

    def session_for_patient(self, patient_id: str, source_system: str) -> dict | None:
        row = self._one(
            "SELECT * FROM stream_sessions WHERE patient_id=? AND source_system=? ORDER BY created_at DESC LIMIT 1",
            (patient_id, source_system))
        return dict(row) if row else None

    def set_ingestion_cursor(self, session_id: str, epoch: int, source_system: str, source_sequence: int,
                             updated_at: str) -> None:
        self._exec(
            "INSERT INTO ingestion_cursors(session_id, epoch, source_system, source_sequence, updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(session_id, epoch) DO UPDATE SET source_sequence=excluded.source_sequence, "
            "updated_at=excluded.updated_at",
            (session_id, epoch, source_system, source_sequence, updated_at),
        )

    def record_delivery(self, session_id: str, epoch: int, raw_id: int, source_sequence: int, event_id: str,
                        received_at: str) -> bool:
        with self._lock:
            existing = self._one(
                "SELECT 1 FROM session_deliveries WHERE session_id=? AND epoch=? AND raw_id=?",
                (session_id, epoch, raw_id))
            if existing:
                return False
            self._exec(
                "INSERT INTO session_deliveries(session_id, epoch, raw_id, source_sequence, event_id, received_at) "
                "VALUES (?,?,?,?,?,?)", (session_id, epoch, raw_id, source_sequence, event_id, received_at))
            return True

    def delivered_count(self, session_id: str, epoch: int) -> int:
        return int(self._one("SELECT COUNT(*) AS n FROM session_deliveries WHERE session_id=? AND epoch=?",
                             (session_id, epoch))["n"])

    # ---- events -------------------------------------------------------------
    def append_event(self, session_id: str, event_id: str, event_type: str, payload: dict, emitted_at: str) -> int:
        with self._lock:
            row = self._one("SELECT last_event_sequence FROM stream_sessions WHERE session_id=?", (session_id,))
            if row is None:
                raise KeyError(session_id)
            sequence = int(row["last_event_sequence"]) + 1
            self._exec(
                "INSERT INTO stream_events(session_id, sequence, event_id, event_type, payload_json, emitted_at) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, sequence, event_id, event_type, json.dumps(payload, sort_keys=True), emitted_at))
            self._exec("UPDATE stream_sessions SET last_event_sequence=? WHERE session_id=?", (sequence, session_id))
            return sequence

    def events_after(self, session_id: str, sequence: int, limit: int = 5000) -> list[dict]:
        return [
            {"sequence": int(r["sequence"]), "event_id": r["event_id"], "event_type": r["event_type"],
             "payload": json.loads(r["payload_json"]), "emitted_at": r["emitted_at"]}
            for r in self._rows(
                "SELECT * FROM stream_events WHERE session_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                (session_id, sequence, limit))
        ]

    # ---- timeline -------------------------------------------------------------
    def delivered_observations(self, session_id: str, epoch: int, cutoff: str | None = None) -> list[dict]:
        """Canonical rows delivered in this epoch, newest ledger version per resource id.

        Rows whose latest delivered version is entered-in-error carry no canonical
        row and therefore disappear from the timeline; earlier versions of a
        corrected resource are superseded by the newest delivered version.
        """
        params: list[Any] = [session_id, epoch]
        cutoff_sql = ""
        if cutoff is not None:
            cutoff_sql = " AND c.effective_time <= ?"
            params.append(cutoff)
        rows = self._rows(
            "WITH delivered AS ("
            "  SELECT r.raw_id, r.resource_id, r.version_id, r.resource_type, d.received_at AS delivered_at, "
            "         d.source_sequence"
            "  FROM session_deliveries d JOIN raw_fhir_resources r ON r.raw_id=d.raw_id"
            "  WHERE d.session_id=? AND d.epoch=? AND r.resource_type='Observation'"
            "), latest AS ("
            "  SELECT resource_id, MAX(source_sequence) AS max_seq FROM delivered GROUP BY resource_id"
            ")"
            " SELECT c.*, dv.delivered_at, dv.source_sequence FROM delivered dv"
            " JOIN latest l ON l.resource_id=dv.resource_id AND l.max_seq=dv.source_sequence"
            " JOIN canonical_observations c ON c.raw_id=dv.raw_id"
            " WHERE 1=1" + cutoff_sql +
            " ORDER BY c.effective_time, c.variable, c.canonical_id",
            params,
        )
        return [dict(r) for r in rows]

    def latest_delivered_effective_time(self, session_id: str, epoch: int) -> str | None:
        rows = self.delivered_observations(session_id, epoch)
        return rows[-1]["effective_time"] if rows else None

    # ---- runs -------------------------------------------------------------------
    def create_run(self, run: dict) -> None:
        self._exec(
            "INSERT INTO model_runs(run_id, session_id, epoch, patient_id, status, runtime_mode, model_commit, "
            "registry_sha256, release_date, input_cutoff, input_watermark, input_resource_count, input_hash, "
            "input_refs_json, trigger_event_ids_json, started_at) VALUES (:run_id,:session_id,:epoch,:patient_id,"
            ":status,:runtime_mode,:model_commit,:registry_sha256,:release_date,:input_cutoff,:input_watermark,"
            ":input_resource_count,:input_hash,:input_refs_json,:trigger_event_ids_json,:started_at)",
            run,
        )

    def finish_run(self, run_id: str, *, status: str, completed_at: str, duration_ms: int,
                   error: str | None, summary: dict | None) -> None:
        self._exec(
            "UPDATE model_runs SET status=?, completed_at=?, duration_ms=?, error=?, summary_json=? WHERE run_id=?",
            (status, completed_at, duration_ms, error, json.dumps(summary, sort_keys=True) if summary else None,
             run_id))

    def insert_cells(self, run_id: str, cells: list[dict]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO forecast_cells(run_id, cell, target, horizon_hours, unit, observed_value, "
                "measurement_age_hours, observations, forecast_point, source_model, status, fallback_reason, "
                "interval_status, lower, upper, causal_claim_allowed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, c["cell"], c["target"], int(c["horizon_hours"]), c.get("unit"), c.get("observed_value"),
                  c.get("measurement_age_hours"), int(c.get("observations") or 0), c.get("forecast_point"),
                  c["source_model"], c["status"], c.get("fallback_reason"), c["interval_status"],
                  c.get("lower"), c.get("upper"), int(bool(c.get("causal_claim_allowed"))))
                 for c in cells],
            )

    def observations_for_refs(self, session_id: str, epoch: int, refs: list[str]) -> list[dict]:
        """Canonical rows for exactly these ``Observation/<id>/_history/<version>`` refs.

        A model run records the refs it actually consumed. Reading them back by ref -
        rather than re-filtering the current timeline by cutoff - is what keeps a
        historical view honest: a resource delivered after the run, or an amendment
        that superseded one of its inputs, is simply not in the set.
        """
        pairs = []
        for ref in refs:
            parts = ref.split("/")
            if len(parts) == 4 and parts[0] == "Observation" and parts[2] == "_history":
                pairs.append((parts[1], parts[3]))
        if not pairs:
            return []
        placeholders = ",".join("(?,?)" for _ in pairs)
        params: list[Any] = [session_id, epoch]
        for resource_id, version_id in pairs:
            params.extend((resource_id, version_id))
        rows = self._rows(
            "SELECT c.*, d.received_at AS delivered_at, d.source_sequence"
            " FROM session_deliveries d"
            " JOIN raw_fhir_resources r ON r.raw_id=d.raw_id"
            " JOIN canonical_observations c ON c.raw_id=d.raw_id"
            " WHERE d.session_id=? AND d.epoch=? AND r.resource_type='Observation'"
            f" AND (c.resource_id, c.version_id) IN (VALUES {placeholders})"
            " ORDER BY c.effective_time, c.variable, c.canonical_id",
            tuple(params))
        return [dict(r) for r in rows]

    def run(self, run_id: str) -> dict | None:
        row = self._one("SELECT * FROM model_runs WHERE run_id=?", (run_id,))
        return self._run_dict(row) if row else None

    def runs_for_patient(self, patient_id: str, limit: int = 200) -> list[dict]:
        return [self._run_dict(r) for r in self._rows(
            "SELECT * FROM model_runs WHERE patient_id=? ORDER BY started_at DESC, run_id DESC LIMIT ?",
            (patient_id, limit))]

    def latest_completed_run(self, session_id: str, epoch: int) -> dict | None:
        row = self._one(
            "SELECT * FROM model_runs WHERE session_id=? AND epoch=? AND status='completed' "
            "ORDER BY started_at DESC, run_id DESC LIMIT 1", (session_id, epoch))
        return self._run_dict(row) if row else None

    def latest_run(self, session_id: str, epoch: int) -> dict | None:
        row = self._one(
            "SELECT * FROM model_runs WHERE session_id=? AND epoch=? ORDER BY started_at DESC, run_id DESC LIMIT 1",
            (session_id, epoch))
        return self._run_dict(row) if row else None

    def cells_for_run(self, run_id: str) -> list[dict]:
        cells = []
        for r in self._rows("SELECT * FROM forecast_cells WHERE run_id=? ORDER BY horizon_hours, target", (run_id,)):
            cell = dict(r)
            cell["causal_claim_allowed"] = bool(cell["causal_claim_allowed"])
            cells.append(cell)
        return cells

    def cell_history(self, patient_id: str, cell: str, limit: int = 200) -> list[dict]:
        return [dict(r) for r in self._rows(
            "SELECT m.run_id, m.epoch, m.session_id, m.status AS run_status, m.input_cutoff, m.completed_at, "
            "f.forecast_point, f.observed_value, f.status, f.source_model FROM forecast_cells f "
            "JOIN model_runs m ON m.run_id=f.run_id WHERE m.patient_id=? AND f.cell=? "
            "ORDER BY m.started_at DESC LIMIT ?", (patient_id, cell, limit))]

    @staticmethod
    def _run_dict(row: sqlite3.Row) -> dict:
        run = dict(row)
        run["input_refs"] = json.loads(run.pop("input_refs_json") or "[]")
        run["trigger_event_ids"] = json.loads(run.pop("trigger_event_ids_json") or "[]")
        summary = run.pop("summary_json")
        run["summary"] = json.loads(summary) if summary else None
        return run


class _Transaction:
    def __init__(self, repo: Repository):
        self.repo = repo

    def __enter__(self):
        self.repo._lock.acquire()
        self.repo._conn.execute("BEGIN IMMEDIATE")
        return self.repo

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.repo._conn.execute("COMMIT")
            else:
                self.repo._conn.execute("ROLLBACK")
        finally:
            self.repo._lock.release()
        return False
