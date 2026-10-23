"""Build the canonical eICU measurement cache used by every eICU cohort extractor.

Laboratory values (``bedside glucose`` mapped to ``glucose_poc``), nurse-charted
vital signs and interval-normalized urine output are read once for all stays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eicu_sepsis_transition_extract import MEASUREMENT_CACHE, build_measurement_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=MEASUREMENT_CACHE)
    args = parser.parse_args()
    print(json.dumps(build_measurement_cache(args.data_root, args.output), indent=2))


if __name__ == "__main__":
    main()
