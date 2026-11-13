"""Record a lockbox extraction addendum before any extracted label is opened.

Claims the one-shot step ``addendum_<database>`` of a frozen protocol and writes
its receipt with the SHA-256 of the extractor, its mapping files and every
extraction output (``osler_jepa.frozen_protocol``).  Lockbox scoring
(``site_adaptive_external_audit.py``) refuses to run without this receipt or if
any recorded file changed.  Before the addendum only support counts and
timestamps of the extraction may be inspected, never target values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_value_conditional_interval_audit import verify_protocol
from osler_jepa.frozen_protocol import claim_step, record_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--database", choices=("SICdb", "HiRID"), required=True)
    parser.add_argument("--files", nargs="+", required=True, help="extractor, mapping files and extraction outputs")
    args = parser.parse_args()
    verify_protocol(args.protocol)
    missing = [name for name in args.files if not Path(name).exists()]
    if missing:
        raise FileNotFoundError(missing)
    step = f"addendum_{args.database}"
    claim_step(args.protocol, step)
    print(json.dumps(record_step(args.protocol, step, args.files, []), indent=2))


if __name__ == "__main__":
    main()
