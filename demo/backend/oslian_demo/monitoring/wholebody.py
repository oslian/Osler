"""The single WholeBody runtime adapter.

Two modes, chosen by ``OSLIAN_WHOLEBODY_RUNTIME``:

``fixture`` (default)
    The pinned release is *not* attached. The 41-cell catalog, model commit,
    registry and selector hashes and units come from the versioned contract fixture
    (``fixtures/wholebody_release_contract_v2.json``, generated from the real
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
    missing | degraded), fallback_reason, interval_status, lower, upper,
    causal_claim_allowed (always false)

``lower``/``upper`` are accepted only from a model cell the catalog marks
``interval_capable``, with one of the lock's calibrated ``interval_status``
values, finite bounds and ``lower <= forecast_point <= upper``. Anything else
that carries a bound fails the run closed. Every other cell is null.

``hypoxemia_event`` cells carry a calibrated event probability in [0, 1] as their
``forecast_point``; they never carry an interval.

``status`` is decided per prediction, not per cell. The release reports a head
that declined to move as ``patient_persistence_anchor`` with
``fallback_reason: continuous_head_abstained``; that is a baseline, not a model
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
# every release source that is a model prediction when the release says the cell moved
MODEL_SOURCES = frozenset({
    "frozen_whole_body_jepa", "unified_shared_jepa_target_head", "multisource_shared_jepa_target_head",
    "calibrated_event_risk_head",
})
ABSTENTION_REASON = "continuous_head_abstained"
EVENT_TARGET = "hypoxemia_event"
STATUS_MODEL, STATUS_PERSISTENCE, STATUS_MISSING, STATUS_DEGRADED = "model", "persistence", "missing", "degraded"
INTERVAL_STATUS_NOT_ATTACHED = "runtime_not_attached"
# User decisions. 2026-09-01: the release may produce real outputs in an explicit demo launch
# profile, as a *bounded research-demo runtime* only. 2026-11-12: the demo moves to the
# oslian/Model 2026-09-02 release and shows its calibrated 90% intervals and event probabilities
# where - and only where - the release emits them from a cell's own calibrator. It is still not the
# formal adjacent-three-level architecture, not a production runtime and not a clinically
# validated model; every fail-closed check stays.
BOUNDED_DEMO_APPROVAL = {
    "approved_scope": "bounded research-demo runtime",
    "approved_on": "2026-09-01",
    "release_adopted_on": "2026-11-12",
    "launch_profile": "OSLIAN_WHOLEBODY_RUNTIME=release with a pinned LF checkout (see README.md)",
    "required_ui_wording": ["bounded research demo", "research-only", "no clinical alert"],
    "must_not_be_called": ["formal adjacent-three-level model", "production runtime", "clinically validated model"],
    "formal_architecture_rule_satisfied": False,
    "promotion_registry_entry": False,
    "fail_closed_checks_unchanged": True,
    "output_boundaries": ["persistence baseline", "interval only from the cell's own release calibrator",
                          "event cells are calibrated probabilities", "causal_claim_allowed: false"],
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
        # model/ in this repository is the default; an override is for another working copy of it
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
    if document.get("schema") != "oslian.wholebody-release-contract.v2":
        raise ValueError("unsupported WholeBody contract fixture schema")
    return document


def _bounds(cell: str, raw: dict[str, Any], status: str, point: float | None, spec: dict[str, Any],
            bounded_statuses: frozenset[str]) -> tuple[float | None, float | None]:
    """The interval a cell may show, or a refusal. Never a repaired or guessed bound."""
    lower, upper = raw.get("lower"), raw.get("upper")
    interval_status = str(raw.get("interval_status") or "")
    if lower is None and upper is None:
        if interval_status in bounded_statuses:
            raise RuntimeUnavailable(f"{cell}: release claims {interval_status} but emitted no bounds")
        return None, None
    if status != STATUS_MODEL or not spec.get("interval_capable") or interval_status not in bounded_statuses:
        raise RuntimeUnavailable(f"{cell}: release emitted an interval the pinned catalog does not allow "
                                 f"(status={status}, interval_status={interval_status!r})")
    try:
        lo, hi = float(lower), float(upper)
    except (TypeError, ValueError) as exc:
        raise RuntimeUnavailable(f"{cell}: release emitted a non-numeric interval") from exc
    if not (math.isfinite(lo) and math.isfinite(hi) and point is not None and lo <= point <= hi):
        raise RuntimeUnavailable(f"{cell}: release interval [{lo}, {hi}] does not contain its point {point}")
    return lo, hi


def normalize_release_cells(release_cells: list[dict[str, Any]], state: dict[str, dict[str, Any]],
                            catalog: dict[str, dict[str, Any]],
                            bounded_statuses: frozenset[str]) -> list[dict[str, Any]]:
    """Map raw release cells onto the Oslian cell contract.

    ``state`` is the delivered-timeline variable state (observed value, age,
    count) so observed values come from our ledger, not from the CSV echo.
    """
    out = []
    for raw in release_cells:
        cell = str(raw["cell"])
        if cell not in catalog:
            raise RuntimeUnavailable(f"release returned a cell outside the pinned registry: {cell}")
        spec = catalog[cell]
        target = str(raw["target"])
        state_variable = "o2sat" if target == EVENT_TARGET else target
        observed = state.get(state_variable)
        point = raw.get("forecast_point")
        source = str(raw.get("source_model") or "unknown")
        if point is None:
            status = STATUS_MISSING
        elif source == PERSISTENCE_SOURCE:
            status = STATUS_PERSISTENCE
        elif source in MODEL_SOURCES and raw.get("can_move"):
            status = STATUS_MODEL
        else:
            raise RuntimeUnavailable(f"{cell}: release returned a point from an unrecognised source {source!r}")
        point = float(point) if point is not None else None
        if target == EVENT_TARGET and status == STATUS_MODEL and not 0.0 <= point <= 1.0:
            raise RuntimeUnavailable(f"{cell}: event probability {point} is outside [0, 1]")
        lower, upper = _bounds(cell, raw, status, point, spec, bounded_statuses)
        out.append({
            "cell": cell,
            "target": target,
            "unit": raw.get("unit") or spec.get("unit"),
            "horizon_hours": int(raw["horizon_hours"]),
            "observed_value": raw.get("observed_value"),
            "measurement_age_hours": raw.get("measurement_age_hours"),
            "observations": int(raw.get("observations") or 0),
            "forecast_point": point,
            "source_model": source,
            "status": status,
            "fallback_reason": raw.get("fallback_reason"),
            "interval_status": str(raw.get("interval_status") or "unknown"),
            "lower": lower,
            "upper": upper,
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

    def status(self) -> dict[str, Any]:
        """The startup verdict, without ever waiting for an inference to finish.

        ``self._lock`` is held for the whole of a ``predict`` round trip, so
        asking for it here would make ``/api/model/health`` - and anything else
        that reads the runtime - queue behind the model. The startup record and
        the process handle are written once at start and only read afterwards, so
        a live worker can be reported from them directly. The lock is taken only
        when the worker genuinely has to be started, which happens at most once
        per process and never while inference is in flight.
        """
        if self._process is not None and self._process.poll() is None and self.startup:
            return self.startup
        return self.ensure_started()

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
        for key in ("registry_sha256", "selector_sha256"):
            if self.lock[key] != self.contract[key]:
                raise RuntimeUnavailable(f"model-lock.json and the runtime contract fixture disagree on {key}")
        self.bounded_statuses = frozenset(self.lock["output_boundaries"]["interval_statuses_with_bounds"])
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
            "selector_sha256": self.contract["selector_sha256"],
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
            "interval_cells": sum(1 for c in cells if c["interval_capable"]),
            "event_risk_cells": sum(1 for c in cells if c["target"] == EVENT_TARGET and c["can_move"]),
            "variables": self.contract["variables"],
            "variable_units": self.contract["variable_units"],
            "modules": self.contract["modules"],
            "modules_note": "functional modules from the checkpoint; they are not anatomical organs",
            "forward_fill_max_age_hours": self.contract["forward_fill_max_age_hours"],
            "cells": [
                {"cell": c["cell"], "target": c["target"], "horizon_hours": c["horizon_hours"],
                 "unit": c.get("unit"), "source_model": c["source_model"], "can_move": c["can_move"],
                 "interval_capable": c["interval_capable"], "fallback_reason": c.get("fallback_reason")}
                for c in cells
            ],
            "interval_policy": self.lock["output_boundaries"]["interval_policy"],
            "event_policy": self.lock["output_boundaries"]["event_cells"],
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
        startup = self.worker.status()
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
        if startup.get("selector_sha256") != self.contract["selector_sha256"]:
            raise RuntimeUnavailable("worker source selector hash differs from the pinned contract selector")
        cells = normalize_release_cells(reply["result"]["cells"], state, self.catalog, self.bounded_statuses)
        return RuntimeResult(cells, self._summary(cells), self.provenance(),
                             int((time.perf_counter() - started) * 1000))

    def _degraded_cells(self, state: dict[str, dict[str, Any]], model_input: ModelInput, reason: str) -> list[dict]:
        cells = []
        for spec in self.contract["catalog"]["cells"]:
            target = spec["target"]
            variable = "o2sat" if target == EVENT_TARGET else target
            observed = state.get(variable)
            cells.append({
                "cell": spec["cell"], "target": target, "unit": spec.get("unit"),
                "horizon_hours": int(spec["horizon_hours"]),
                "observed_value": (float(observed["value"] <= 92.0) if target == EVENT_TARGET else observed["value"]) if observed else None,
                "measurement_age_hours": observed["age_hours"] if observed else None,
                "observations": int(observed["count"]) if observed else 0,
                "forecast_point": None, "source_model": "none", "status": STATUS_DEGRADED,
                "fallback_reason": reason, "interval_status": INTERVAL_STATUS_NOT_ATTACHED,
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
