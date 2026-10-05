"""Verify real expanded source stores without touching evaluation labels."""
import json
from pathlib import Path

import torch

from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.multisource_intervals import sha256


def main():
    out = Path("artifacts/native_reconstruction_source_store_20260918.json")
    if out.exists():
        raise ValueError("Verification already recorded")
    root = Path("data/research_recovery/broad_source_20260917")
    reports = [Path("artifacts/native_reconstruction_joint_source_manifest_20260917.json"),
               Path("artifacts/native_reconstruction_full_broad_samples_20260918.json")]
    for report in reports:
        for p, expected in json.loads(report.read_text())["bindings"].items():
            if sha256(Path(p)) != expected:
                raise ValueError("Source binding changed")
    paths = [root / "joint_source_manifest.pt", root / "full_samples/index.pt",
             Path("data/research_recovery/reconstruction_nested_selection_20260917.pt")]
    joint, expanded, split = [torch.load(p, weights_only=False) for p in paths]
    columns = [joint["variables"].index(t) for t in joint["targets"]]
    result = {}
    for mode in ("small", "expanded"):
        shards = source_shards(joint, expanded, mode, split["roles"])
        result[mode] = {}
        for group in ("selected", "broad"):
            for role in ("fit", "encoder_tune"):
                subset = [s for s in shards if s["group"] == group and s["role"] == role]
                store = SourceStore(subset, columns)
                # Independently access one complete target batch through the lazy path.
                for c, keys in store.groups.items():
                    if not keys:
                        raise ValueError("Missing trained target")
                    if store[keys[0]]["truth_normalized"] != store.labels[keys[0]]:
                        raise ValueError("Lazy label mismatch")
                result[mode][f"{group}/{role}"] = dict(
                    patients=len(subset), pairs=len(store.labels),
                    support={joint["variables"][c]: len(keys) for c, keys in store.groups.items()},
                    cache_capacity=store._cached_read.cache_info().maxsize)
                print(mode, group, role, len(subset), len(store.labels), flush=True)
    dependencies = paths + reports + [Path(__file__), Path("scripts/reconstruction_source_store.py"),
                            Path("RECONSTRUCTION_FULL_EXPOSURE_PROTOCOL_20260918.md")]
    out.write_text(json.dumps(dict(complete=True, stores=result,
        bindings={str(p): sha256(p) for p in dependencies},
        training_started=False, runtime_modified=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
