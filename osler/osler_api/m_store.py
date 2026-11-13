"""Durable per-hospital state of M in SQLite (standard library), one transaction per request.

Tables: ``stays`` (each hospital's stays in arrival order, for the warm-up), ``cells`` (the online state of each
hospital x training sample x cell, as JSON) and ``forecasts`` (issued forecasts awaiting outcomes: anchor time,
forecast time per cell, outcomes received).  Stay keys are whatever the caller sends; nothing else about the patient
is stored.  Forecasts older than ``EXPIRE_DAYS`` are dropped with their pending state.  A single lock serializes
requests, so the update order is the order requests arrive.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXPIRE_DAYS = 7
SCHEMA = """
CREATE TABLE IF NOT EXISTS stays (hospital TEXT NOT NULL, stay TEXT NOT NULL, arrival INTEGER NOT NULL,
                                  PRIMARY KEY (hospital, stay));
CREATE TABLE IF NOT EXISTS cells (hospital TEXT NOT NULL, sample INTEGER NOT NULL, cell TEXT NOT NULL, state TEXT NOT NULL,
                                  PRIMARY KEY (hospital, sample, cell));
CREATE TABLE IF NOT EXISTS forecasts (id TEXT PRIMARY KEY, hospital TEXT NOT NULL, stay TEXT NOT NULL,
                                      anchor_time TEXT NOT NULL, created TEXT NOT NULL, issued TEXT NOT NULL,
                                      outcomes TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS forecasts_by_hospital ON forecasts (hospital, created);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.connection.executescript(SCHEMA)
        self.lock = threading.Lock()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield Transaction(self.connection)
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            self.connection.execute("COMMIT")


class Transaction:
    def __init__(self, connection: sqlite3.Connection):
        self.db = connection

    def arrival(self, hospital: str, stay: str) -> tuple[int, int]:
        """The stay's arrival index at the hospital (0 = first) and how many stays the hospital has seen."""

        row = self.db.execute("SELECT arrival FROM stays WHERE hospital = ? AND stay = ?", (hospital, stay)).fetchone()
        seen = self.db.execute("SELECT COUNT(*) FROM stays WHERE hospital = ?", (hospital,)).fetchone()[0]
        if row is not None:
            return int(row[0]), int(seen)
        self.db.execute("INSERT INTO stays VALUES (?, ?, ?)", (hospital, stay, seen))
        return int(seen), int(seen) + 1

    def cell_state(self, hospital: str, sample: int, cell: str) -> dict | None:
        row = self.db.execute("SELECT state FROM cells WHERE hospital = ? AND sample = ? AND cell = ?",
                              (hospital, sample, cell)).fetchone()
        return None if row is None else json.loads(row[0])

    def save_state(self, hospital: str, sample: int, cell: str, state: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cells VALUES (?, ?, ?, ?)", (hospital, sample, cell, json.dumps(state)))

    def add_forecast(self, forecast_id: str, hospital: str, stay: str, anchor_time: str, issued: dict[str, str]) -> None:
        self.db.execute("INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (forecast_id, hospital, stay, anchor_time, now(), json.dumps(issued), "{}"))

    def latest_anchor(self, hospital: str, stay: str) -> str | None:
        row = self.db.execute("SELECT MAX(anchor_time) FROM forecasts WHERE hospital = ? AND stay = ?", (hospital, stay)).fetchone()
        return row[0]

    def forecast(self, forecast_id: str) -> dict | None:
        row = self.db.execute("SELECT hospital, stay, anchor_time, issued, outcomes FROM forecasts WHERE id = ?",
                              (forecast_id,)).fetchone()
        if row is None:
            return None
        return {"hospital": row[0], "stay": row[1], "anchor_time": row[2], "issued": json.loads(row[3]),
                "outcomes": json.loads(row[4])}

    def record_outcome(self, forecast_id: str, outcomes: dict) -> None:
        self.db.execute("UPDATE forecasts SET outcomes = ? WHERE id = ?", (json.dumps(outcomes), forecast_id))

    def expired(self, hospital: str) -> set[str]:
        """Remove and return the hospital's forecasts older than ``EXPIRE_DAYS``."""

        cutoff = (datetime.now(timezone.utc) - timedelta(days=EXPIRE_DAYS)).isoformat()
        ids = {r[0] for r in self.db.execute("SELECT id FROM forecasts WHERE hospital = ? AND created < ?", (hospital, cutoff))}
        self.db.execute("DELETE FROM forecasts WHERE hospital = ? AND created < ?", (hospital, cutoff))
        return ids


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
