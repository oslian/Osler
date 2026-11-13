"""Shared preflight and result writing of every one-shot lockbox scorer (continuous, binary event and threshold event
parts).

A lockbox protocol scores its cells in parts, one per task type: ``parts[task_type]`` holds the part's ``step``,
``scored_cells``, the ``deployment_form_arguments`` of its scorer, ``external_reads`` (each read cell ``target@<h>h``
-> the extraction file it must come from) and its ``output`` path.  Every scorer resolves its configuration before
calling ``claim_scoring_step``, which checks, in order, that:

- a protocol and an external template are given and the run is from the project folder (where the protocol's
  relative paths resolve);
- every hashed code, configuration and training file is unchanged (``verify_protocol``);
- the database is one of the protocol's ``databases``; the protocol scores this part, with exactly these cells, this
  exact invocation, these exact external files per read cell, and this output path, which does not exist yet;
- the Python and package versions equal the protocol's ``environment``;
- every project module loaded in this process, every configuration file of the part (and a previous protocol whose
  extraction is reused) is hashed, and the MIMIC training file of every scored horizon is hashed;
- the database's addendum receipt exists, its recorded files are unchanged, it records every frozen read of the
  part, and every frozen read holds rows of this database only (``hospital_identities``; an empty, mixed or
  unidentified file fails here, before the claim, not at the read); where the protocol declares ``reused_extractions[database]``, those files are byte-identical to that previous
  protocol's addendum.

Just before the claim it takes an exclusive operating-system lock on the protocol's scoring (``hold_scoring_lock``),
so the parts run one at a time.  The claim then sets ``args.claimed_step``; reads go through ``external_cohort``, which admits only the part's frozen
reads of addendum-recorded files, and only while the step is open (``osler_jepa.frozen_protocol.authorize_lockbox_read``).
``write_result`` refuses a receipt if a project module the protocol does not hash was loaded after the claim, writes
the output exclusively and records the step.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections.abc import Callable
from pathlib import Path

from external_value_conditional_interval_audit import verify_protocol
from osler_jepa import frozen_protocol
from osler_jepa.frozen_protocol import claim_step, record_step, sha256, step_record, write_exclusive
from osler_jepa.lockbox_guard import hospital_identities
from osler_jepa.project_sources import PROJECT, environment, loaded_project_files

STEP_KIND = {"continuous_value": "continuous", "binary_event_probability": "binary",
             "threshold_event_probability": "threshold", "own_trajectory_point": "own_trajectory",
             "single_event_probability": "single_event", "joint_dependence": "dependence"}
def external_reads(rows: list[dict], template: str) -> dict[str, str]:
    """Each read cell ``target@<h>h`` of a part's contracts -> the external file of its horizon."""

    return {f"{row['target']}@{int(row['horizon_hours'])}h": template.format(horizon=int(row["horizon_hours"])) for row in rows}


def claim_scoring_step(args: argparse.Namespace, *, task_type: str, contracts: Callable[[], list[dict]],
                       config_files: list[Path], form: dict, reads: dict[str, str] | None = None,
                       training_files: list[str] | None = None) -> str:
    """Run every preflight check of one lockbox part, then claim its one-shot step and return it.

    ``contracts`` returns the part's cell contracts (with ``target`` and ``horizon_hours``); it is called only after
    the run is known to be in the project folder, where the catalog's relative paths resolve.  A part whose external
    files are not one per horizon (the anchored examples of each MIMIC sample) passes its ``reads`` (read key -> file)
    and its MIMIC ``training_files``; otherwise both follow from the templates."""

    if args.protocol is None:
        raise ValueError(f"{args.database} is a lockbox: --protocol is required")
    if not args.external_template:
        raise ValueError("--external-template is required for a lockbox database")
    if Path.cwd().resolve() != PROJECT:
        raise RuntimeError(f"lockbox scoring runs from the project folder {PROJECT}, where the protocol's paths resolve")
    rows = contracts()
    cells, horizons = [row["cell"] for row in rows], sorted({int(row["horizon_hours"]) for row in rows})
    protocol = verify_protocol(args.protocol)
    if args.database not in protocol.get("databases", []):
        raise RuntimeError(f"{args.database} is not a database of this protocol")
    part = protocol.get("parts", {}).get(task_type)
    if part is None:
        raise RuntimeError(f"{task_type} cells are outside this protocol's lockbox scope")
    step = f"{STEP_KIND[task_type]}_{args.database}"
    if part.get("step") != step:
        raise RuntimeError(f"the protocol's {task_type} part is not step {step}")
    if sorted(cells) != sorted(part["scored_cells"]):
        raise RuntimeError(f"{task_type} cells differ from the protocol's scored_cells")
    if form != part["deployment_form_arguments"]:
        raise RuntimeError("lockbox invocation differs from the protocol's frozen deployment form")
    reads = external_reads(rows, args.external_template) if reads is None else reads
    if reads != part.get("external_reads"):
        raise RuntimeError("external files differ from the protocol's frozen external_reads")
    if str(args.output) != part.get("output"):
        raise RuntimeError(f"the output of this part is frozen as {part.get('output')}")
    if Path(args.output).exists():
        raise FileExistsError(f"{args.output} exists; a one-shot output is never overwritten")
    if environment() != protocol.get("environment"):
        raise RuntimeError(f"interpreter or package versions differ from the protocol: {environment()}")
    reused = protocol.get("reused_extractions", {}).get(args.database)
    required = loaded_project_files() | {str(path) for path in config_files} | ({reused} if reused else set())
    unhashed = sorted(required - set(protocol["frozen_code_sha256"]))
    if unhashed:
        raise RuntimeError(f"files used by the scoring are not hashed by the protocol: {unhashed}")
    training = {args.mimic_template.format(horizon=h) for h in horizons} if training_files is None else set(training_files)
    untrained = sorted(training - set(protocol.get("frozen_training_data_sha256", {})))
    if untrained:
        raise RuntimeError(f"MIMIC training files not hashed by the protocol: {untrained}")
    addendum = step_record(args.protocol, f"addendum_{args.database}")["inputs_sha256"]     # raises without an addendum
    changed = sorted(name for name, digest in addendum.items() if sha256(Path(name)) != digest)
    if changed:
        raise RuntimeError(f"extraction files changed after the addendum: {changed}")
    unrecorded = sorted(set(reads.values()) - set(addendum))
    if unrecorded:
        raise RuntimeError(f"external transitions not recorded in the addendum: {unrecorded}")
    foreign = sorted(name for name in set(reads.values()) if hospital_identities(name) != {args.database})
    if foreign:
        raise RuntimeError(f"frozen reads not exclusively {args.database} rows (empty, mixed or another source): {foreign}")
    if reused:
        prior = step_record(Path(reused), f"addendum_{args.database}")["inputs_sha256"]
        differing = sorted(name for name in set(reads.values()) if prior.get(name) != addendum[name])
        if differing:
            raise RuntimeError(f"extraction files differ from the previous protocol's addendum: {differing}")
    args.scoring_lock = hold_scoring_lock(args.protocol)
    claim_step(args.protocol, step)
    args.claimed_step = step                     # external_cohort reads lockbox files only inside this open step
    return step


def hold_scoring_lock(protocol_path: Path) -> int:
    """An exclusive operating-system lock on this protocol's scoring, held until ``write_result`` or process exit (a
    crash releases it too), so parts run one at a time: a second part is refused before its claim."""

    folder = frozen_protocol.LEDGER / sha256(Path(protocol_path))
    folder.mkdir(parents=True, exist_ok=True)
    handle = os.open(folder / "scoring.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(handle)
        raise RuntimeError("another part of this protocol is being scored; parts run one at a time") from None
    return handle


def write_result(args: argparse.Namespace, step: str | None, output: dict) -> None:
    """Write a scorer's output.  For a claimed lockbox step: refuse if a project module the protocol does not hash was
    loaded after the claim (no output, no receipt), write the output exclusively, then record the step with the part's
    frozen reads as inputs."""

    text = json.dumps(output, indent=2)
    if step is None:
        Path(args.output).write_text(text, encoding="utf-8")
        return
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    late = sorted(loaded_project_files() - set(protocol["frozen_code_sha256"]))
    if late:
        raise RuntimeError(f"project modules loaded after the claim are not hashed by the protocol: {late}; no receipt")
    write_exclusive(Path(args.output), text)
    part = next(part for part in protocol["parts"].values() if part.get("step") == step)
    record_step(args.protocol, step, sorted(set(part["external_reads"].values())), [str(args.output)])
    os.close(args.scoring_lock)
