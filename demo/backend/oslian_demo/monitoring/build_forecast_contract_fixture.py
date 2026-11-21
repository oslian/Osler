"""Regenerate ``fixtures/forecast_api_contract_v1.json`` from a running forecast API.

    python -m oslian_demo.monitoring.build_forecast_contract_fixture --api-url http://127.0.0.1:8000 --api-commit <sha>

``--api-commit`` is the last commit of this repository that changed ``api/``
(``git log -1 --format=%H -- api``). It is recorded as provenance; what the
runtime verifies on every start is the registry hash and the cell list.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from ..config import FORECAST_CONTRACT_FIXTURE
from .mapping import CANONICAL_UNITS
from .timeline import ANCHOR_SPACING_HOURS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-commit", required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.api_url.rstrip("/") + "/v1/expert/cells", timeout=30) as response:
        served = json.loads(response.read().decode("utf-8"))
    if float(served["anchor_spacing_hours"]) != ANCHOR_SPACING_HOURS:
        raise SystemExit("the API's anchor spacing differs from the demo's anchor builder")
    model = served["model"]
    document = {
        "schema": "oslian.forecast-api-contract.v1",
        "generated_from": f"GET /v1/expert/cells of api/ at commit {args.api_commit}",
        "model_repository": "https://github.com/oslian/Osler.git",
        "api_commit": args.api_commit,
        "release_date": "2026-11-02",
        "registry": model["registry"],
        "registry_sha256": model["registry_sha256"],
        "weights": model["weights"],
        "training_database": model["training_database"],
        "anchor_spacing_hours": ANCHOR_SPACING_HOURS,
        "evidence": "each cell was frozen and then replicated on external hospital databases under one-shot protocols; "
                    "research-only, not clinically validated",
        "boundary": served["boundary"],
        "cells": [{"cell": c["cell"], "target": c["target"], "horizon_hours": int(c["horizon_hours"]),
                   "unit": CANONICAL_UNITS[c["target"]], "source_model": c["recipe"], "flags": list(c["flags"])}
                  for c in sorted(served["cells"], key=lambda c: (c["target"], int(c["horizon_hours"])))],
    }
    FORECAST_CONTRACT_FIXTURE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{len(document['cells'])} cells -> {FORECAST_CONTRACT_FIXTURE}")


if __name__ == "__main__":
    main()
