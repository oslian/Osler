"""Frozen six-arm paired evaluation on the two already-inspected cohorts."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_reconstruction_source_fit import durable, BACKUP
from scripts.audit_broad_prior_increment import load_rows
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def main():
    root = Path("data/research_recovery/broad_source_20260917")
    out = Path("artifacts/native_reconstruction_source_exposure_evaluation_20260918.json")
    private = root / "source_exposure_evaluation"
    if out.exists() or private.exists():
        raise ValueError("Existing evaluation")
    tp = Path("artifacts/native_reconstruction_source_exposure_pair_20260918.json")
    training = json.loads(tp.read_text())
    for p, h in training["bindings"].items():
        if sha256(Path(p)) != h:
            raise ValueError("Training changed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assets = deepcopy(load_release(Path("models")))
    variables = assets.checkpoint["variables"]
    expand_history_projection(assets.model, 6)
    assets.model.belief_input_mode = "observed_steps"
    template = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ("map",))).eval()
    paths = [Path(__file__), tp]
    models, medians = {}, {}
    for mode in ("small", "expanded"):
        for arm in ("real", "process", "placebo"):
            p = root / "source_exposure_pair" / mode / f"{arm}.pt"
            ck = torch.load(p, weights_only=False)
            if ck["epoch"] != 8 or ck["bindings"] != training["bindings"]:
                raise ValueError("Checkpoint mismatch")
            model = deepcopy(template)
            model.load_state_dict(ck["best_state_dict"])
            models[mode, arm] = model
            medians[mode] = ck["medians"]
            paths.append(p)
    bi, op = root / "matched_evaluation/index.pt", root / "retention.pt"
    refs = {"broad": load_rows(bi), "selected": {
        (r["patient"], r["target"]): r for r in torch.load(op, weights_only=False)}}
    indices = {"broad": Path("data/research_recovery/expanded_confirmation_20260917/samples/index.pt"),
               "selected": BACKUP / "map_native_anchor_support/expanded_reconstruction_20260913/index.pt"}
    sp = Path("artifacts/native_reconstruction_source_store_20260918.json")
    supports = json.loads(sp.read_text())["stores"]
    catalog = list(json.loads(Path("artifacts/native_reconstruction_matched_broad_evaluation_20260917.json").read_text())["targets"])
    paths.extend([bi, op, sp, *indices.values()])
    private.mkdir()
    results = {}
    for group, ip in indices.items():
        patients = {k[0] for k in refs[group]}
        rows, seen = [], set()
        with torch.inference_mode():
            for i, shard in enumerate(torch.load(ip, weights_only=False)["shards"]):
                if shard["patient"] not in patients:
                    continue
                p = durable(shard["path"])
                if sha256(p) != shard["sha256"]:
                    raise ValueError("Sample changed")
                s = torch.load(p, weights_only=False)
                for c, e in s["samples"].items():
                    key = (s["patient"], variables[c])
                    if key not in refs[group]:
                        continue
                    ref = refs[group][key]
                    if key in seen or ref["truth"] != e["truth_normalized"]:
                        raise ValueError("Pair mismatch")
                    seen.add(key)
                    row = dict(patient=key[0], target=key[1], site=ref["site"], truth=ref["truth"], old=ref["old"])
                    for mode in medians:
                        row[f"{mode}_population"] = medians[mode][c]
                    for (mode, arm), model in models.items():
                        row[f"{mode}_{arm}"] = float(model(input_tokens(e, arm, len(variables)),
                            e["event_hours"], e["anchor_hour"], key[1])[0])
                    rows.append(row)
                if (i + 1) % 500 == 0:
                    print(group, "patients traversed", i + 1, flush=True)
        if seen != set(refs[group]):
            raise ValueError("Incomplete evaluation")
        p = private / f"{group}.pt"
        torch.save(rows, p)
        paths.append(p)
        per = {}
        names = ["old"] + [f"{m}_{a}" for m in medians for a in ("population", "real", "process", "placebo")]
        for t in catalog:
            rr = [r for r in rows if r["target"] == t]
            if not rr:
                per[t] = dict(status="no_trained_source_head")
                continue
            y = np.array([r["truth"] for r in rr])
            errors = {name: abs(np.array([r[name] for r in rr]) - y) for name in names}
            sites = [r["site"] for r in rr]
            checks = {name: evidence(errors["expanded_real"] - errors[name], sites)
                      for name in ("old", "small_real", "expanded_population", "expanded_process", "expanded_placebo")}
            supported = all(d["support"][t] >= 30 for d in supports["expanded"].values())
            per[t] = dict(patients=len(rr), mae={k: float(v.mean()) for k, v in errors.items()},
                comparisons=checks, source_supported=supported,
                development_gate=supported and confirmed(checks, len(rr)))
        results[group] = per
    out.write_text(json.dumps(dict(complete=True, groups=results,
        bindings={str(p): sha256(p) for p in paths}, runtime_modified=False, promotion_allowed=False,
        limitation="Reused cohorts; current-value masked reconstruction only. Equal updates, changed broad patient support and source prior."), indent=2) + "\n")
    print("Paired evaluation complete", flush=True)


if __name__ == "__main__":
    main()
