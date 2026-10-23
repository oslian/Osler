"""Merge per-horizon external_value_conditional_interval_audit runs into one report.

Runs are split by horizon only to bound memory; methods and gates are those of
the scoring script.  Counts are recomputed from the merged cell list.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from external_value_conditional_interval_audit import write_markdown


def merge(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    databases = {run["external_database"] for run in runs}
    if len(databases) != 1:
        raise ValueError(f"cannot merge different external databases: {sorted(databases)}")
    cells = sorted(
        (cell for run in runs for cell in run["cells"]),
        key=lambda cell: (int(cell["horizon_hours"]), cell["target"]),
    )
    evaluable = [cell for cell in cells if cell["evaluable"]]
    continuous = [cell for cell in evaluable if "fixed" in cell["metrics"]]
    merged = {key: value for key, value in runs[0].items() if key != "cells"}
    merged.update({
        "merged_from": [str(path) for path in paths],
        "mimic_data_sha256": {k: v for run in runs for k, v in run["mimic_data_sha256"].items()},
        "catalog_cells": len(cells),
        "evaluable_cells": len(evaluable),
        "not_evaluable_reasons": dict(Counter(cell["reason"] for cell in cells if not cell["evaluable"])),
        "point_pass_cells": sum(cell["metrics"]["point_pass"] for cell in evaluable),
        "fixed_interval_pass_cells": sum(cell["metrics"]["fixed"]["pass"] for cell in continuous),
        "value_conditional_interval_pass_cells": sum(
            cell["metrics"]["value_conditional"]["pass"] for cell in continuous
        ),
        "fixed_full_pass_cells": sum(cell["metrics"]["fixed_full_pass"] for cell in evaluable),
        "value_conditional_full_pass_cells": sum(
            cell["metrics"]["value_conditional_full_pass"] for cell in evaluable
        ),
        "cells": cells,
    })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    args = parser.parse_args()
    merged = merge(args.runs)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    write_markdown(merged, args.findings)


if __name__ == "__main__":
    main()
