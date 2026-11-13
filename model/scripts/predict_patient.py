#!/usr/bin/env python3
"""Run the frozen release on one longitudinal patient CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whole_body_jepa import predict_patient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="patient CSV")
    parser.add_argument("--output", type=Path, required=True, help="output JSON")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    result = predict_patient(
        args.input.read_text(encoding="utf-8"),
        filename=args.input.name,
        model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
