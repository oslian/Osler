"""Model release worker: runs *inside* the pinned release directory (model/).

Launched by ``wholebody.ReleaseWorker`` as a separate interpreter so that the
release's own ``osler_jepa``/``whole_body_joint_jepa`` modules and its pinned
scikit-learn are the ones loaded (the web repo has same-named packages).

Protocol (one JSON object per line on stdin/stdout):
    startup  -> {"ready": bool, "checks": [...], "commit": ..., "registry_sha256": ...,
                 "selector_sha256": ..., ...}
    request  {"op": "predict", "request_id": "...", "csv": "..."}
    reply    {"request_id": "...", "ok": true, "result": {...}} | {"ok": false, "error": "..."}

Manifest verification is never bypassed. A CRLF checkout makes the JSON files
fail their SHA-256; the fix is an LF checkout (``core.autocrlf=false`` or
``.gitattributes``), not a weaker check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _git_head(release_dir: Path) -> str | None:
    """The last commit that changed the release directory - its pinned identity in a monorepo."""
    try:
        return subprocess.run(["git", "-C", str(release_dir), "log", "-1", "--format=%H", "--", "."],
                              capture_output=True, text=True, check=True, timeout=60).stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    release_dir = Path(args.release_dir).resolve()

    # sys.path hygiene: the release checkout must shadow every same-named module.
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in {script_dir, ""}]
    sys.path.insert(0, str(release_dir))
    os.chdir(release_dir)

    checks: list[dict] = []
    ready = True

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal ready
        checks.append({"check": name, "status": "passed" if ok else "failed", "detail": detail})
        if not ok:
            ready = False

    head = _git_head(release_dir)
    check("pinned_commit", head == args.expected_commit, f"last model commit={head} expected={args.expected_commit}")

    manifest_path = release_dir / "models" / "MODEL_MANIFEST.json"
    manifest: dict = {}
    registry_sha = None
    selector_sha = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            path = release_dir / "models" / name
            actual = _sha256(path) if path.is_file() else None
            ok = actual == expected["sha256"]
            hint = ""
            if not ok and name.endswith(".json") and path.is_file() and b"\r\n" in path.read_bytes():
                hint = " (file has CRLF line endings; re-checkout with core.autocrlf=false or add .gitattributes)"
            check(f"sha256:{name}", ok, f"actual={actual}{hint}")
            if name == "runtime_registry.json" and ok:
                registry_sha = actual
            if name == "unified_source_selector.json" and ok:
                selector_sha = actual
    except Exception as exc:  # noqa: BLE001
        check("manifest", False, f"{type(exc).__name__}: {exc}")

    assets = None
    catalog = None
    if ready:
        try:
            import osler_jepa  # noqa: WPS433
            import whole_body_joint_jepa  # noqa: WPS433
            for module in (osler_jepa, whole_body_joint_jepa):
                origin = Path(module.__file__).resolve()
                inside = str(origin).startswith(str(release_dir))
                # report a release-relative path so no local checkout path leaks into fixtures or logs
                shown = origin.relative_to(release_dir).as_posix() if inside else "outside release checkout"
                check(f"module_origin:{module.__name__}", inside, shown)
            import sklearn  # noqa: WPS433
            check("scikit_learn_pin", sklearn.__version__ == "1.7.2", f"installed={sklearn.__version__}")
            from whole_body_jepa import load_release, predict_patient  # noqa: WPS433
            assets = load_release(release_dir / "models")
            catalog = assets.registry
            check("registry_cells", len(catalog["cells"]) == int(manifest.get("catalog_cells", -1)),
                  f"cells={len(catalog['cells'])}")
            smoke = predict_patient((release_dir / "examples" / "patient_1.csv").read_text(encoding="utf-8"),
                                    filename="patient_1.csv", model_dir=release_dir / "models")
            check("smoke_predict", smoke["summary"]["catalog_cells"] == len(catalog["cells"]),
                  json.dumps(smoke["summary"]))
        except Exception as exc:  # noqa: BLE001
            check("load_release", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)

    _emit({
        "ready": ready,
        "checks": checks,
        "commit": head,
        "registry_sha256": registry_sha,
        "selector_sha256": selector_sha,
        "release_date": manifest.get("release_date"),
        "manifest": manifest,
        "scikit_learn": __import__("sklearn").__version__ if ready else None,
    })
    if not ready:
        return 2

    from whole_body_jepa import predict_patient  # noqa: WPS433

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _emit({"ok": False, "error": "malformed request"})
            continue
        request_id = request.get("request_id")
        if request.get("op") == "shutdown":
            _emit({"request_id": request_id, "ok": True})
            return 0
        if request.get("op") != "predict":
            _emit({"request_id": request_id, "ok": False, "error": "unsupported op"})
            continue
        started = time.perf_counter()
        try:
            result = predict_patient(str(request["csv"]), filename="stream.csv", model_dir=release_dir / "models")
            _emit({"request_id": request_id, "ok": True, "result": result,
                   "duration_ms": int((time.perf_counter() - started) * 1000)})
        except Exception as exc:  # noqa: BLE001
            _emit({"request_id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
