"""Matched-budget small/full source comparison with persistent patient queues."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from scripts.preflight_full_source_exposure import draw_queue
from scripts.reconstruction_source_store import SourceStore, source_shards
from scripts.train_joint_source_reconstruction import equal_group_median
from scripts.train_reconstruction_replication_v2 import input_tokens
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_jepa.runtime import load_release


def epoch_plan(groups, queues, rng, draws=536):
    schedules = {}
    for group, targets in groups.items():
        selected = {c: draw_queue(keys, queues[group][c], rng, draws)
                    for c, keys in targets.items()}
        schedules[group] = [selected[c][i] for i in range(draws)
                            for c in rng.permutation(sorted(targets))]
    lengths = {len(s) for s in schedules.values()}
    if len(lengths) != 1:
        raise ValueError("Unequal group budgets")
    return [(g, schedules[g][i]) for i in range(next(iter(lengths)))
            for g in rng.permutation(list(groups))]


def shuffled_labels(store):
    rng = np.random.default_rng(20260913)
    result = {}
    for c, keys in sorted(store.groups.items()):
        if len(keys) < 2:
            raise ValueError("Placebo needs distinct patients")
        order = [keys[i] for i in rng.permutation(len(keys))]
        for receiver, donor in zip(order, order[1:] + order[:1]):
            if receiver[0] == donor[0]:
                raise ValueError("Placebo self-pair")
            result[receiver] = store.labels[donor]
    return result


def main():
    root = Path("data/research_recovery/broad_source_20260917")
    out = Path("artifacts/native_reconstruction_source_exposure_pair_20260918.json")
    if out.exists():
        raise ValueError("Completed experiment exists")
    verification = Path("artifacts/native_reconstruction_source_store_20260918.json")
    for p, h in json.loads(verification.read_text())["bindings"].items():
        if sha256(Path(p)) != h:
            raise ValueError("Verified source changed")
    joint = torch.load(root / "joint_source_manifest.pt", weights_only=False)
    expanded = torch.load(root / "full_samples/index.pt", weights_only=False)
    split = torch.load("data/research_recovery/reconstruction_nested_selection_20260917.pt", weights_only=False)
    variables, targets = joint["variables"], joint["targets"]
    columns = [variables.index(t) for t in targets]
    dependencies = [Path(__file__), verification,
        Path("scripts/preflight_full_source_exposure.py"), Path("scripts/reconstruction_source_store.py"),
        Path("scripts/train_joint_source_reconstruction.py"), Path("scripts/train_reconstruction_replication_v2.py"),
        Path("RECONSTRUCTION_FULL_EXPOSURE_PROTOCOL_20260918.md"), Path("models/whole_body_joint_jepa.pt")]
    dependencies += list(Path("whole_body_jepa").glob("*.py"))
    bindings = {str(p): sha256(p) for p in dependencies}
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    summary = {}
    for mode in ("small", "expanded"):
        shards = source_shards(joint, expanded, mode, split["roles"])
        data = {g: {r: SourceStore([s for s in shards if s["group"] == g and s["role"] == r], columns)
                    for r in ("fit", "encoder_tune")} for g in ("selected", "broad")}
        groups = {g: d["fit"].groups for g, d in data.items()}
        medians = {c: equal_group_median([np.array([data[g]["fit"].labels[k] for k in groups[g][c]])
                                       for g in data]) for c in columns}
        torch.manual_seed(19)
        assets = deepcopy(load_release(Path("models")))
        if assets.checkpoint["variables"] != variables:
            raise ValueError("Schema mismatch")
        expand_history_projection(assets.model, 6)
        assets.model.belief_input_mode = "observed_steps"
        initial = NativeBodyReconstruction(NativeOrganStateResidual(assets.model, variables, ("map",))).eval()
        with torch.no_grad():
            initial.readout.weight.zero_()
            initial.readout.bias.zero_()
            for c, median in medians.items():
                initial.readout.bias[c] = median
        placebo = {g: shuffled_labels(d["fit"]) for g, d in data.items()}

        def score(model, arm):
            result = {}
            with torch.inference_mode():
                for g, d in data.items():
                    errors = {c: [] for c in columns}
                    for key, truth in d["encoder_tune"].labels.items():
                        e = d["encoder_tune"][key]
                        prediction = float(model(input_tokens(e, arm, len(variables)),
                            e["event_hours"], e["anchor_hour"], variables[key[1]])[0])
                        errors[key[1]].append(abs(prediction - truth))
                    result[g] = {variables[c]: float(np.mean(v)) for c, v in errors.items()}
            value = float(np.mean([v for group in result.values() for v in group.values()]))
            if not np.isfinite(value):
                raise ValueError("Nonfinite selection score")
            return value, result

        private = root / "source_exposure_pair" / mode
        private.mkdir(parents=True, exist_ok=True)
        summary[mode] = {}
        for arm in ("real", "process", "placebo"):
            model = deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            rng = np.random.default_rng(19)
            queues = {g: {c: {} for c in columns} for g in data}
            seen = {g: {c: set() for c in columns} for g in data}
            path = private / f"{arm}.pt"
            if path.exists():
                ck = torch.load(path, weights_only=False)
                if ck["bindings"] != bindings:
                    raise ValueError("Resume contract changed")
                model.load_state_dict(ck["state_dict"])
                optimizer.load_state_dict(ck["optimizer"])
                rng.bit_generator.state = ck["rng"]
                torch.set_rng_state(ck["torch_rng"])
                queues, seen = ck["queues"], ck["seen"]
            else:
                value, per = score(model, arm)
                ck = dict(epoch=0, selected_epoch=0, best_tune=value,
                    best_state_dict=deepcopy(model.state_dict()), initial_scores=per, log=[], bindings=bindings)
            print(mode, arm, "start after epoch", ck["epoch"], flush=True)
            for epoch in range(ck["epoch"] + 1, 9):
                plan = epoch_plan(groups, queues, rng)
                loss_sum = 0.
                for group, key in plan:
                    e = data[group]["fit"][key]
                    seen[group][key[1]].add(key[0])
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(input_tokens(e, arm, len(variables)),
                        e["event_hours"], e["anchor_hour"], variables[key[1]])
                    truth = placebo[group][key] if arm == "placebo" else e["truth_normalized"]
                    loss = torch.nn.functional.smooth_l1_loss(prediction, prediction.new_tensor([truth]))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    loss_sum += float(loss.detach())
                value, per = score(model, arm)
                exposure = {g: {variables[c]: len(p) for c, p in d.items()} for g, d in seen.items()}
                for g in data:
                    for c in columns:
                        if len(seen[g][c]) != min(len(groups[g][c]), epoch * 536):
                            raise ValueError("Actual exposure differs from protocol")
                ck["log"].append(dict(epoch=epoch, tune_mae=value, group_target_scores=per,
                    updates=len(plan), draw_loss=loss_sum / len(plan), unique_exposure=exposure))
                if value < ck["best_tune"]:
                    ck.update(best_tune=value, selected_epoch=epoch, best_state_dict=deepcopy(model.state_dict()))
                ck.update(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), queues=queues, seen=seen,
                    variables=variables, targets=targets, medians=medians)
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, bindings=bindings),
                           private / f"{arm}_epoch{epoch}.pt")
                temp = path.with_suffix(".tmp")
                torch.save(ck, temp)
                temp.replace(path)
                print(mode, arm, epoch, value, "selected", ck["selected_epoch"], flush=True)
            summary[mode][arm] = {k: ck[k] for k in ("epoch", "selected_epoch", "best_tune", "initial_scores", "log")}
    out.write_text(json.dumps(dict(training_complete=True, modes=summary, bindings=bindings,
        source_targets=targets, runtime_modified=False, promotion_allowed=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
