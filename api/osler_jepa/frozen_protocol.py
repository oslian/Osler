"""Bind one-shot analysis steps to a frozen protocol.

A protocol JSON names the spec it freezes (``"spec"``) and hashes files
(``"frozen_sha256"``).  ``load_frozen`` reads the spec only from the protocol,
refuses to run unless the spec, the spec's design table and every file the spec
lists are hashed and unchanged, and returns both.

Execution identity is the protocol's digest, not its filename: every step of a
protocol is claimed atomically in the project's ``causal_executions/<protocol
sha256>/`` (a byte-identical copy under another name or directory shares the ledger), completed by an
exclusively written receipt, and read back only after the claim, the receipt's
step and its protocol digest are checked.  Approval is never part of the
immutable protocol: ``reviewed`` checks a separate review record naming the
exact protocol and result digests.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path


LEDGER = Path(__file__).resolve().parents[1] / "causal_executions"     # one ledger per project, whatever the working directory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen(protocol_path: Path) -> tuple[dict, dict]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    frozen = protocol.get("frozen_sha256", {})
    if protocol.get("spec") not in frozen:
        raise RuntimeError("protocol does not hash its spec")
    if sha256(Path(protocol["spec"])) != frozen[protocol["spec"]]:
        raise RuntimeError(f"files changed after protocol freeze: {[protocol['spec']]}")
    spec = json.loads(Path(protocol["spec"]).read_text(encoding="utf-8"))
    missing = sorted({spec["design_table"], *spec["frozen_files"]} - set(frozen))
    if missing:
        raise RuntimeError(f"protocol does not hash required files: {missing}")
    changed = sorted(name for name, digest in frozen.items() if sha256(Path(name)) != digest)
    if changed:
        raise RuntimeError(f"files changed after protocol freeze: {changed}")
    return protocol, spec


def _ledger(protocol_path: Path) -> Path:
    return LEDGER / sha256(Path(protocol_path))


def write_exclusive(path: Path, text: str) -> None:
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(f"{path} exists; a one-shot step is never repeated") from None
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        stream.write(text)


def claim_step(protocol_path: Path, step: str) -> None:
    """Consume ``step`` of this protocol once; a second or concurrent claim raises."""

    folder = _ledger(protocol_path)
    folder.mkdir(parents=True, exist_ok=True)
    write_exclusive(folder / f"{step}.claimed", datetime.datetime.now(datetime.timezone.utc).isoformat())


def is_claimed(protocol_path: Path, step: str) -> bool:
    return (_ledger(protocol_path) / f"{step}.claimed").exists()


SCORING_STEP_KINDS = ("continuous", "binary", "threshold", "own_trajectory", "single_event", "dependence")


def authorize_lockbox_read(protocol_path: Path, step: str | None, database: str, cell: str, path: Path) -> None:
    """Allow reading ``path`` of a lockbox for ``cell`` (``target@<h>h``) only inside an open scoring step of that
    database, only as that step's part froze it (``parts[...]["external_reads"][cell] == path``), and only a file the
    database's addendum recorded, unchanged.  An addendum step, another database's step, an unclaimed step, a completed
    step (receipt written), another part's cell or another file does not authorize the read."""

    if step not in {f"{kind}_{database}" for kind in SCORING_STEP_KINDS}:
        raise RuntimeError(f"{database} is a lockbox: its data are read only inside a claimed scoring step of {database}")
    if not is_claimed(protocol_path, step):
        raise RuntimeError(f"{database} is a lockbox: step {step!r} was not claimed")
    if (_ledger(protocol_path) / f"{step}.json").exists():
        raise RuntimeError(f"{database} is a lockbox: step {step!r} is completed")
    parts = [part for part in json.loads(Path(protocol_path).read_text(encoding="utf-8")).get("parts", {}).values()
             if part.get("step") == step]
    if len(parts) != 1 or parts[0].get("external_reads", {}).get(cell) != str(path):
        raise RuntimeError(f"{path} for {cell} is not a frozen read of step {step!r}")
    recorded = step_record(protocol_path, f"addendum_{database}")["inputs_sha256"]
    if recorded.get(str(path)) is None or sha256(Path(path)) != recorded[str(path)]:
        raise RuntimeError(f"{path} is not an unchanged file recorded in the {database} addendum")


def record_step(protocol_path: Path, step: str, inputs: list[str], outputs: list[str]) -> dict:
    """Exclusive completion receipt of a claimed step, with input and output digests."""

    folder = _ledger(protocol_path)
    if not (folder / f"{step}.claimed").exists():
        raise RuntimeError(f"step {step!r} was not claimed")
    record = {"protocol_sha256": sha256(Path(protocol_path)), "step": step,
              "inputs_sha256": {name: sha256(Path(name)) for name in inputs},
              "outputs_sha256": {name: sha256(Path(name)) for name in outputs}}
    write_exclusive(folder / f"{step}.json", json.dumps(record, indent=2))
    return record


def step_record(protocol_path: Path, step: str) -> dict:
    """The completion receipt of a claimed step of exactly this protocol."""

    folder = _ledger(protocol_path)
    if not (folder / f"{step}.claimed").exists():
        raise RuntimeError(f"step {step!r} was not claimed")
    record = json.loads((folder / f"{step}.json").read_text(encoding="utf-8"))
    if record.get("step") != step or record.get("protocol_sha256") != sha256(Path(protocol_path)):
        raise RuntimeError(f"receipt of step {step!r} does not belong to this protocol")
    return record


def write_once(path: Path) -> Path:
    if Path(path).exists():
        raise FileExistsError(f"{path} exists; a one-shot output is never overwritten")
    return Path(path)


def reviewed(review_path: Path, protocol_path: Path, result_path: Path) -> bool:
    """True only for an explicit approval of exactly this protocol and this result."""

    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    return (review.get("decision") == "approved" and bool(review.get("reviewer"))
            and review.get("protocol_sha256") == sha256(Path(protocol_path))
            and review.get("result_sha256") == sha256(Path(result_path)))
