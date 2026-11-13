#!/usr/bin/env python3
"""Print every accepted measurement and every forecast-cell status."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from whole_body_jepa.contracts import VARIABLE_UNITS


def main() -> None:
    checkpoint = torch.load(
        Path("models/whole_body_joint_jepa.pt"),
        map_location="cpu",
        weights_only=False,
    )
    registry = json.loads(Path("models/runtime_registry.json").read_text())
    print("INPUT MEASUREMENTS")
    for variable in checkpoint["variables"]:
        print(f"{variable:24} {VARIABLE_UNITS[variable]}")
    print("\nFORECAST CELLS")
    for cell in registry["cells"]:
        status = "MODEL" if cell["can_move"] else "PERSISTENCE"
        print(f"{cell['cell']:28} {status:11} {cell['source_model']}")


if __name__ == "__main__":
    main()
