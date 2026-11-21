"""The single forecast runtime adapter: the validated expert cells, over HTTP.

The demo owns no model. Forecasts come from the forecast API in ``api/`` of this
repository (``POST /v1/expert/forecast``): 45 target x horizon cells that were
frozen and then replicated on external hospital databases. The demo shows 42 of
them; ``NOT_SHOWN`` names the three it leaves out and why. Two modes, chosen by
``OSLIAN_FORECAST_RUNTIME``:

``detached`` (default)
    No API is called. The catalog comes from the pinned contract
    (``fixtures/forecast_api_contract_v1.json``, generated from the API). Every
    run completes with ``status: degraded`` cells and a null ``forecast_point``;
    nothing is fabricated and the UI must say the runtime is not attached.

``api``
    Calls the API at ``OSLIAN_FORECAST_API_URL``. Before the first forecast the
    API's own cell list and registry hash are compared with the pinned contract,
    and every reply is checked again; any difference fails the run closed.

Cell contract (every mode)::

    cell, target, unit, horizon_hours, observed_value, measurement_age_hours,
    observations, forecast_point, source_model, status (model | missing |
    degraded), fallback_reason, interval_status, lower, upper,
    causal_claim_allowed (always false)

``lower``/``upper`` are always null: the validated interval needs each
hospital's own outcomes, which a demo stream does not have. A forecast is issued
at an anchor (every 6 hours from the encounter start), so a run's input cutoff is
an anchor time and ``horizon_hours`` counts from it. Past 72 hours of stay the
cells were never fitted, and the run returns degraded cells that say so.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
import urllib.error
from urllib.parse import urlsplit
import urllib.request

from ..config import FORECAST_CONTRACT_FIXTURE as CONTRACT_FIXTURE, is_loopback
from .fhir import parse_instant
from .timeline import EPISODE_MAX_HOURS, ModelInput

MODE_DETACHED = "detached"
MODE_API = "api"
STATUS_MODEL, STATUS_MISSING, STATUS_DEGRADED = "model", "missing", "degraded"
INTERVAL_STATUS = "not_available_without_site_online_state"
INTERVAL_STATUS_NOT_ATTACHED = "runtime_not_attached"
INTERVAL_POLICY = ("no interval is shown: the validated interval is fitted online on each hospital's own outcomes, "
                   "which this demo stream does not have")
REASON_NOT_ATTACHED = "forecast_api_not_attached"
REASON_OUTSIDE_EPISODE = "outside_fitted_episode_window"
REQUEST_TIMEOUT_SECONDS = 30.0
# Cells the API serves and the demo does not show. Each passed its validation on average error
# and was then found to say little for the patients it is most often asked about. They are left
# out of the catalog and never requested; the capabilities document lists them with the reason.
NOT_SHOWN: dict[str, str] = {
    "urine_output@3h": "the forecast is close to 80 mL/hour whatever the current rate is; below a current rate of "
                       "100 mL/hour it is further from what is measured than the current rate itself",
    "urine_output@6h": "the forecast is close to 80 mL/hour whatever the current rate is; below a current rate of "
                       "100 mL/hour it is further from what is measured than the current rate itself",
    "creatinine@48h": "the forecast predicts a rise at normal creatinine values that the data do not show",
}


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    api_url: str
    api_token: str = ""

    def __post_init__(self) -> None:
        # Requests carry patient measurements: beyond this computer they go over TLS, to an API that asks for a token.
        parts = urlsplit(self.api_url)
        if self.mode == MODE_API and not is_loopback(parts.hostname or ""):
            if parts.scheme != "https":
                raise ValueError("OSLIAN_FORECAST_API_URL must be https unless the API runs on this computer")
            if not self.api_token:
                raise ValueError("OSLIAN_FORECAST_API_TOKEN is required unless the API runs on this computer")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if env is None else env
        mode = env.get("OSLIAN_FORECAST_RUNTIME", MODE_DETACHED).strip().lower()
        if mode not in {MODE_DETACHED, MODE_API}:
            raise ValueError("OSLIAN_FORECAST_RUNTIME must be 'detached' or 'api'")
        return cls(mode=mode, api_url=env.get("OSLIAN_FORECAST_API_URL", "http://127.0.0.1:8000").rstrip("/"),
                   api_token=env.get("OSLIAN_FORECAST_API_TOKEN", ""))


@dataclass(frozen=True)
class RuntimeResult:
    cells: list[dict[str, Any]]
    summary: dict[str, Any]
    provenance: dict[str, Any]
    duration_ms: int


class RuntimeUnavailable(RuntimeError):
    """The runtime cannot produce a trustworthy result; the run must fail closed."""


def load_contract_fixture(path=CONTRACT_FIXTURE) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "oslian.forecast-api-contract.v1":
        raise ValueError("unsupported forecast contract fixture schema")
    return document


class ForecastRuntime:
    def __init__(self, config: RuntimeConfig | None = None, contract_path=CONTRACT_FIXTURE):
        self.config = config or RuntimeConfig.from_env()
        self.contract = load_contract_fixture(contract_path)
        served = {c["cell"] for c in self.contract["cells"]}
        if not set(NOT_SHOWN) <= served:
            raise ValueError(f"cells marked not shown are not in the contract: {sorted(set(NOT_SHOWN) - served)}")
        self.cells: list[dict[str, Any]] = [c for c in self.contract["cells"] if c["cell"] not in NOT_SHOWN]
        self.catalog: dict[str, dict[str, Any]] = {c["cell"]: c for c in self.cells}
        self._verified = False
        self.error: str | None = None

    # ---- the API ----------------------------------------------------------------
    def _call(self, path: str, body: dict | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.api_url + path, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.config.api_token}"} if self.config.api_token else {})})
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeUnavailable(f"forecast API refused {path}: {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeUnavailable(f"forecast API unreachable at {self.config.api_url}: {exc}") from exc

    def _check_model(self, model: dict[str, Any]) -> None:
        if model.get("registry_sha256") != self.contract["registry_sha256"]:
            raise RuntimeUnavailable("the forecast API serves a registry other than the pinned contract's")

    def verify(self) -> None:
        """The API's cells and registry are the pinned contract's, or the runtime is not attached."""
        if self._verified:
            return
        try:
            served = self._call("/v1/expert/cells")
            self._check_model(served.get("model") or {})
            cells = {(c["cell"], c["target"], int(c["horizon_hours"])) for c in served["cells"]}
            if cells != {(c["cell"], c["target"], int(c["horizon_hours"])) for c in self.contract["cells"]}:
                raise RuntimeUnavailable("the forecast API's cells differ from the pinned contract's")
        except RuntimeUnavailable as exc:
            self.error = str(exc)
            raise
        self.error, self._verified = None, True

    # ---- discovery ------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self.config.mode

    def attached(self) -> bool:
        return self.mode == MODE_API and self._verified

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.mode,
            "runtime_attached": self.attached(),
            "model_repository": self.contract["model_repository"],
            "model_commit": self.contract["api_commit"],
            "release_date": self.contract["release_date"],
            "registry": self.contract["registry"],
            "registry_sha256": self.contract["registry_sha256"],
            "weights": self.contract["weights"],
            "runtime_scope": "validated expert cells, day-one points" if self.mode == MODE_API else "detached (safe default)",
            "evidence": self.contract["evidence"],
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            **self.provenance(),
            "catalog_cells": len(self.cells),
            "model_cells": len(self.cells),
            "interval_cells": 0,
            "anchor_spacing_hours": self.contract["anchor_spacing_hours"],
            "episode_max_hours": EPISODE_MAX_HOURS,
            "cells": [dict(c) for c in self.cells],
            "served_cells": len(self.contract["cells"]),
            "not_shown": [{"cell": cell, "reason": reason} for cell, reason in sorted(NOT_SHOWN.items())],
            "interval_policy": INTERVAL_POLICY,
            "boundary": self.contract["boundary"],
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "clinical_alerts_supported": False,
        }

    def health(self) -> dict[str, Any]:
        base = self.provenance()
        contract_check = {"check": "contract_fixture", "status": "passed",
                          "detail": f"{len(self.cells)} catalog cells from {self.contract['generated_from']}"}
        if self.mode == MODE_DETACHED:
            return {**base, "status": "runtime_not_attached", "ready": False,
                    "reason": "the forecast API is not attached; runs return degraded cells", "checks": [contract_check]}
        try:
            self.verify()
        except RuntimeUnavailable:
            pass
        ready = self.attached()
        return {**base, "runtime_attached": ready, "status": "healthy" if ready else "degraded", "ready": ready,
                "reason": None if ready else self.error,
                "checks": [contract_check, {"check": "api_cells_and_registry", "status": "passed" if ready else "failed",
                                            "detail": self.config.api_url if ready else str(self.error)}]}

    # ---- inference -------------------------------------------------------------
    def run(self, model_input: ModelInput, state: dict[str, dict[str, Any]]) -> RuntimeResult:
        started = time.perf_counter()
        done = lambda cells: RuntimeResult(cells, self._summary(cells), self.provenance(),
                                           int((time.perf_counter() - started) * 1000))
        if self.mode == MODE_DETACHED:
            return done(self._cells(state, {}, REASON_NOT_ATTACHED))
        hours = (parse_instant(model_input.cutoff, "cutoff") - parse_instant(model_input.onset, "onset")).total_seconds() / 3600.0
        if hours > EPISODE_MAX_HOURS:
            return done(self._cells(state, {}, REASON_OUTSIDE_EPISODE))
        self.verify()                                             # raises RuntimeUnavailable -> the run fails closed
        reply = self._call("/v1/expert/forecast", {"anchors": list(model_input.anchors), "cells": sorted(self.catalog)})
        self._check_model(reply.get("model") or {})
        if parse_instant(reply["anchor_time"] + "Z", "anchor_time") != parse_instant(model_input.cutoff, "cutoff"):
            raise RuntimeUnavailable("the forecast API answered for an anchor other than the run's input cutoff")
        forecasts = {item["cell"]: item for item in reply["forecasts"]}
        if set(forecasts) != set(self.catalog):
            raise RuntimeUnavailable(f"the forecast API returned {len(forecasts)} cells; the contract has {len(self.catalog)}")
        return done(self._cells(state, forecasts, None))

    def _cells(self, state: dict[str, dict[str, Any]], forecasts: dict[str, dict[str, Any]], reason: str | None) -> list[dict]:
        cells = []
        for spec in self.cells:
            observed = state.get(spec["target"])
            forecast = forecasts.get(spec["cell"])
            point = forecast["point"] if forecast else None
            if forecast is None:
                status, fallback = STATUS_DEGRADED, reason
            elif forecast["status"] == "ok" and point is not None:
                status, fallback = STATUS_MODEL, None
            elif forecast["status"] == "current_value_missing" and point is None:
                status, fallback = STATUS_MISSING, "current_value_missing"
            else:
                raise RuntimeUnavailable(f"{spec['cell']}: the forecast API returned an unrecognised status {forecast['status']!r}")
            cells.append({
                "cell": spec["cell"], "target": spec["target"], "unit": spec["unit"],
                "horizon_hours": int(spec["horizon_hours"]),
                "observed_value": observed["value"] if observed else None,
                "measurement_age_hours": observed["age_hours"] if observed else None,
                "observations": int(observed["count"]) if observed else 0,
                "forecast_point": float(point) if point is not None else None,
                "source_model": spec["source_model"] if forecast else "none", "status": status,
                "fallback_reason": fallback,
                "interval_status": INTERVAL_STATUS if forecast else INTERVAL_STATUS_NOT_ATTACHED,
                "lower": None, "upper": None, "causal_claim_allowed": False,
                "ledger_observation_count": int(observed["count"]) if observed else 0,
            })
        return cells

    @staticmethod
    def _summary(cells: list[dict]) -> dict[str, int]:
        return {
            "catalog_cells": len(cells),
            "model_cells": sum(1 for c in cells if c["status"] == STATUS_MODEL),
            "missing_cells": sum(1 for c in cells if c["status"] == STATUS_MISSING),
            "degraded_cells": sum(1 for c in cells if c["status"] == STATUS_DEGRADED),
        }

    def close(self) -> None:
        """Nothing to release: the API is a separate process the demo does not own."""
