"""The single WholeBody runtime adapter.

Two modes, chosen by ``OSLIAN_WHOLEBODY_RUNTIME``:

``fixture`` (default)
    The pinned release is *not* attached. The 41-cell catalog, model commit,
    registry hash and units come from the versioned contract fixture
    (``fixtures/wholebody_release_contract_v1.json``, generated from the real
    release). Every run completes with ``status: degraded`` cells and a null
    ``forecast_point`` — nothing is fabricated and the UI must say the runtime
    is not attached.

``release``
    Runs the pinned release in a worker subprocess (``worker.py``) so its own
    modules and pinned scikit-learn load. Manifest checksums and the pinned
    commit are verified in the worker; any failure fails closed.

Cell contract (every mode)::

    cell, target, unit, horizon_hours, observed_value, measurement_age_hours,
    observations, forecast_point, source_model, status (model | persistence |
    missing | degraded), fallback_reason, interval_status, lower, upper (null
    unless an exact patient-specific conformal artifact exists — none does),
    causal_claim_allowed (always false)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from ..config import RELEASE_CONTRACT_FIXTURE as CONTRACT_FIXTURE, RESEARCH_DIR, load_model_lock
from .timeline import ModelInput

MODE_FIXTURE = "fixture"
MODE_RELEASE = "release"
PERSISTENCE_SOURCE = "patient_persistence_anchor"
STATUS_MODEL, STATUS_PERSISTENCE, STATUS_MISSING, STATUS_DEGRADED = "model", "persistence", "missing", "degraded"
INTERVAL_STATUS_NONE = "no_patient_specific_conformal_artifact"
# User decision (2026-09-01): the 2026-08-22 release may produce real outputs in an explicit demo
# launch profile, as a *bounded research-demo runtime* only. It is not the formal
# adjacent-three-level architecture required by AGENTS.md, not a production runtime, and not a
# clinically validated model; every fail-closed check and every output boundary stays as is.
BOUNDED_DEMO_APPROVAL = {
    "approved_scope": "bounded research-demo runtime",
    "approved_on": "2026-09-01",
    "launch_profile": "OSLIAN_WHOLEBODY_RUNTIME=release with a pinned LF checkout (see README.md)",
    "required_ui_wording": ["bounded research demo", "research-only", "point forecasts only", "no clinical alert"],
    "must_not_be_called": ["formal adjacent-three-level model", "production runtime", "clinically validated model"],
    "formal_architecture_rule_satisfied": False,
    "promotion_registry_entry": False,
    "fail_closed_checks_unchanged": True,
    "output_boundaries_unchanged": ["persistence baseline", "null interval", "causal_claim_allowed: false"],
    "safe_default_mode": MODE_FIXTURE,
}


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    release_dir: Path | None
    python: str
    pythonpath: str | None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if env is None else env
        mode = env.get("OSLIAN_WHOLEBODY_RUNTIME", MODE_FIXTURE).strip().lower()
        if mode not in {MODE_FIXTURE, MODE_RELEASE}:
            raise ValueError("OSLIAN_WHOLEBODY_RUNTIME must be 'fixture' or 'release'")
        # the submodule is the default checkout; an override is for a working copy of it
        release_dir = env.get("OSLIAN_WHOLEBODY_RELEASE_DIR") or RESEARCH_DIR
        return cls(
            mode=mode,
            release_dir=Path(release_dir).resolve() if release_dir else None,
            python=env.get("OSLIAN_WHOLEBODY_PYTHON") or sys.executable,
            pythonpath=env.get("OSLIAN_WHOLEBODY_PYTHONPATH") or None,
        )


@dataclass(frozen=True)
class RuntimeResult:
    cells: list[dict[str, Any]]
    summary: dict[str, Any]
    provenance: dict[str, Any]
    duration_ms: int


class RuntimeUnavailable(RuntimeError):
    """The runtime cannot produce a trustworthy result; the run must fail closed."""


def load_contract_fixture(path: Path = CONTRACT_FIXTURE) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "oslian.wholebody-release-contract.v1":
        raise ValueError("unsupported WholeBody contract fixture schema")
    return document


def normalize_release_cells(release_cells: list[dict[str, Any]], state: dict[str, dict[str, Any]],
                            catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Map raw release cells onto the Oslian cell contract.

    ``state`` is the delivered-timeline variable state (observed value, age,
    count) so observed values come from our ledger, not from the CSV echo.
    """
    out = []
    for raw in release_cells:
        cell = str(raw["cell"])
        if cell not in catalog:
            raise RuntimeUnavailable(f"release returned a cell outside the pinned registry: {cell}")
        target = str(raw["target"])
        state_variable = "o2sat" if target == "hypoxemia_event" else target
        observed = state.get(state_variable)
        point = raw.get("forecast_point")
        source = str(raw.get("source_model") or "unknown")
        if point is None:
            status = STATUS_MISSING
        elif source == PERSISTENCE_SOURCE:
            status = STATUS_PERSISTENCE
        elif raw.get("can_move"):
            status = STATUS_MODEL
        else:
            status = STATUS_PERSISTENCE
        if raw.get("lower") is not None or raw.get("upper") is not None:
            raise RuntimeUnavailable(f"{cell}: release emitted an interval but no patient-specific artifact exists")
        out.append({
            "cell": cell,
            "target": target,
            "unit": raw.get("unit") or catalog[cell].get("unit"),
            "horizon_hours": int(raw["horizon_hours"]),
            "observed_value": raw.get("observed_value"),
            "measurement_age_hours": raw.get("measurement_age_hours"),
            "observations": int(raw.get("observations") or 0),
            "forecast_point": float(point) if point is not None else None,
            "source_model": source,
            "status": status,
            "fallback_reason": raw.get("fallback_reason"),
            "interval_status": str(raw.get("interval_status") or INTERVAL_STATUS_NONE),
            "lower": None,
            "upper": None,
            "causal_claim_allowed": False,
            "ledger_observation_count": int(observed["count"]) if observed else 0,
        })
    if len(out) != len(catalog):
        raise RuntimeUnavailable(f"release returned {len(out)} cells; registry has {len(catalog)}")
    return out


class ReleaseWorker:
    """Owns the worker subprocess; one request at a time."""

    def __init__(self, config: RuntimeConfig, expected_commit: str):
        self.config = config
        self.expected_commit = expected_commit
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self.startup: dict[str, Any] | None = None
        self.error: str | None = None

    def ensure_started(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None and self.startup:
                return self.startup
            self._start_locked()
            return self.startup or {"ready": False, "checks": [], "error": self.error}

    def _start_locked(self) -> None:
        self.startup = None
        self.error = None
        if self.config.release_dir is None or not (self.config.release_dir / "models").is_dir():
            self.error = "OSLIAN_WHOLEBODY_RELEASE_DIR is not set or has no models/ directory"
            return
        env = dict(os.environ)
        paths = [str(self.config.release_dir)]
        if self.config.pythonpath:
            paths.append(self.config.pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        worker = Path(__file__).resolve().parent / "worker.py"
        try:
            self._process = subprocess.Popen(
                [self.config.python, str(worker), "--release-dir", str(self.config.release_dir),
                 "--expected-commit", self.expected_commit],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", env=env, cwd=str(self.config.release_dir),
            )
            line = self._process.stdout.readline() if self._process.stdout else ""
            if not line:
                self.error = "worker exited before reporting readiness"
                return
            self.startup = json.loads(line)
            if not self.startup.get("ready"):
                self.error = "release validation failed: " + "; ".join(
                    f"{c['check']} {c['detail']}" for c in self.startup.get("checks", []) if c["status"] != "passed")
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None and bool(self.startup and self.startup.get("ready"))

    def predict(self, csv_text: str) -> dict[str, Any]:
        with self._lock:
            if not self.alive():
                self._start_locked()
                if not self.alive():
                    raise RuntimeUnavailable(self.error or "worker unavailable")
            assert self._process and self._process.stdin and self._process.stdout
            request_id = uuid.uuid4().hex
            self._process.stdin.write(json.dumps({"op": "predict", "request_id": request_id, "csv": csv_text}) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                self.startup = None
                raise RuntimeUnavailable("worker terminated during inference")
            reply = json.loads(line)
            if reply.get("request_id") != request_id or not reply.get("ok"):
                raise RuntimeUnavailable(str(reply.get("error") or "worker returned an invalid reply"))
            return reply

    def stop(self) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    if self._process.stdin:
                        self._process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                        self._process.stdin.flush()
                    self._process.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    self._process.kill()
            self._process = None


class WholeBodyRuntime:
    def __init__(self, config: RuntimeConfig | None = None, contract_path: Path = CONTRACT_FIXTURE):
        self.config = config or RuntimeConfig.from_env()
        self.contract = load_contract_fixture(contract_path)
        self.catalog: dict[str, dict[str, Any]] = {c["cell"]: c for c in self.contract["catalog"]["cells"]}
        # model-lock.json is the authority on which release may be loaded. The contract
        # fixture was generated from that same release, so the two must agree; if they
        # ever drift, the demo refuses to start rather than run an unapproved model.
        self.lock = load_model_lock()
        if self.lock["validated_commit"] != self.contract["model_commit"]:
            raise RuntimeUnavailable(
                f"model-lock.json approves {self.lock['validated_commit']} but the runtime contract "
                f"fixture was generated from {self.contract['model_commit']}")
        self.worker = ReleaseWorker(self.config, self.lock["validated_commit"]) if self.config.mode == MODE_RELEASE else None

    # ---- discovery ------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.config.mode

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.mode,
            "runtime_attached": self.attached(),
            "model_repository": self.contract["model_repository"],
            "model_commit": self.contract["model_commit"],
            "release_date": self.contract["release_date"],
            "registry_sha256": self.contract["registry_sha256"],
            "registry_schema": self.contract["catalog"]["schema"],
            "runtime_scope": "bounded research demo" if self.mode == MODE_RELEASE else "gated fixture (safe default)",
            "approval": BOUNDED_DEMO_APPROVAL,
        }

    def attached(self) -> bool:
        return bool(self.worker and self.worker.alive())

    def capabilities(self) -> dict[str, Any]:
        cells = list(self.contract["catalog"]["cells"])
        move = sum(1 for c in cells if c["can_move"])
        return {
            **self.provenance(),
            "catalog_cells": len(cells),
            "model_cells": move,
            "persistence_cells": len(cells) - move,
            "variables": self.contract["variables"],
            "variable_units": self.contract["variable_units"],
            "modules": self.contract["modules"],
            "modules_note": "functional modules from the checkpoint; they are not anatomical organs",
            "forward_fill_max_age_hours": self.contract["forward_fill_max_age_hours"],
            "cells": [
                {"cell": c["cell"], "target": c["target"], "horizon_hours": c["horizon_hours"],
                 "unit": c.get("unit"), "source_model": c["source_model"], "can_move": c["can_move"],
                 "fallback_reason": c.get("fallback_reason")}
                for c in cells
            ],
            "interval_policy": ("lower/upper are null for every cell: the release bundles no "
                                "patient-specific conformal interval artifact"),
            "persistence_policy": "persistence is a baseline fallback, not a model prediction",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "clinical_alerts_supported": False,
            "formal_architecture_gate": self.contract["formal_architecture_gate"],
        }

    def health(self) -> dict[str, Any]:
        base = {**self.provenance(), "strict_manifest_verification": True}
        if self.mode == MODE_FIXTURE:
            return {
                **base, "status": "runtime_not_attached", "ready": False,
                "reason": "WholeBody release runtime is gated pending approval; runs return degraded cells",
                "checks": [{"check": "contract_fixture", "status": "passed",
                            "detail": f"{len(self.catalog)} catalog cells from {self.contract['generated_from']}"}],
            }
        assert self.worker is not None
        startup = self.worker.ensure_started()
        ready = self.worker.alive()
        return {
            **base, "status": "healthy" if ready else "degraded", "ready": ready,
            "runtime_attached": ready,
            "reason": None if ready else self.worker.error,
            "worker_commit": startup.get("commit") if startup else None,
            "worker_registry_sha256": startup.get("registry_sha256") if startup else None,
            "scikit_learn": startup.get("scikit_learn") if startup else None,
            "checks": startup.get("checks", []) if startup else [],
        }

    # ---- inference -------------------------------------------------------------
    def run(self, model_input: ModelInput, state: dict[str, dict[str, Any]]) -> RuntimeResult:
        started = time.perf_counter()
        if self.mode == MODE_FIXTURE:
            cells = self._degraded_cells(state, model_input, "wholebody_runtime_not_attached")
            return RuntimeResult(cells, self._summary(cells), self.provenance(),
                                 int((time.perf_counter() - started) * 1000))
        assert self.worker is not None
        reply = self.worker.predict(model_input.csv_text)  # raises RuntimeUnavailable -> run fails closed
        startup = self.worker.startup or {}
        if startup.get("commit") != self.contract["model_commit"]:
            raise RuntimeUnavailable("worker commit differs from the pinned contract commit")
        if startup.get("registry_sha256") != self.contract["registry_sha256"]:
            raise RuntimeUnavailable("worker registry hash differs from the pinned contract registry")
        cells = normalize_release_cells(reply["result"]["cells"], state, self.catalog)
        return RuntimeResult(cells, self._summary(cells), self.provenance(),
                             int((time.perf_counter() - started) * 1000))

    def _degraded_cells(self, state: dict[str, dict[str, Any]], model_input: ModelInput, reason: str) -> list[dict]:
        cells = []
        for spec in self.contract["catalog"]["cells"]:
            target = spec["target"]
            variable = "o2sat" if target == "hypoxemia_event" else target
            observed = state.get(variable)
            cells.append({
                "cell": spec["cell"], "target": target, "unit": spec.get("unit"),
                "horizon_hours": int(spec["horizon_hours"]),
                "observed_value": (float(observed["value"] <= 92.0) if target == "hypoxemia_event" else observed["value"]) if observed else None,
                "measurement_age_hours": observed["age_hours"] if observed else None,
                "observations": int(observed["count"]) if observed else 0,
                "forecast_point": None, "source_model": "none", "status": STATUS_DEGRADED,
                "fallback_reason": reason, "interval_status": INTERVAL_STATUS_NONE,
                "lower": None, "upper": None, "causal_claim_allowed": False,
                "ledger_observation_count": int(observed["count"]) if observed else 0,
            })
        return cells

    @staticmethod
    def _summary(cells: list[dict]) -> dict[str, int]:
        return {
            "catalog_cells": len(cells),
            "model_cells": sum(1 for c in cells if c["status"] == STATUS_MODEL),
            "persistence_cells": sum(1 for c in cells if c["status"] == STATUS_PERSISTENCE),
            "missing_cells": sum(1 for c in cells if c["status"] == STATUS_MISSING),
            "degraded_cells": sum(1 for c in cells if c["status"] == STATUS_DEGRADED),
        }

    def close(self) -> None:
        if self.worker:
            self.worker.stop()
