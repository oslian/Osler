"""Train and audit a saved pure-JEPA DKA target-head checkpoint.

The v5 JEPA encoder remains frozen.  Real MIMIC rows train a shared neural
head encoder plus one independent head per physiological target.  No
hand-built belief features are allowed in this path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from dka_world_model import STATE_KEYS, load_checkpoint
from jepa_belief_adapter_candidate import build_examples
from real_world_improvement import ADAPTED_STATE_INDICES
from train_intervention_jepa import choose_device


SEEDS = (7, 11, 19, 23, 37, 53, 71)


class PureJEPADKATargetHeads(nn.Module):
    """Shared pure-JEPA feature encoder with target-specific residual heads."""

    def __init__(self, feature_dim: int, target_count: int, hidden: int = 128):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.target_count = int(target_count)
        self.shared = nn.Sequential(
            nn.Linear(feature_dim + target_count, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.target_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(target_count)
            ]
        )

    def forward(self, features: torch.Tensor, current_z: torch.Tensor) -> torch.Tensor:
        shared = self.shared(torch.cat([features, current_z], dim=-1))
        deltas = [0.75 * torch.tanh(head(shared)) for head in self.target_heads]
        return current_z + torch.cat(deltas, dim=-1)


def _split(groups, seed, fraction=0.30):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    test_groups = set(unique[: max(1, int(round(len(unique) * fraction)))].tolist())
    test = np.asarray([group in test_groups for group in groups], dtype=bool)
    return np.flatnonzero(~test), np.flatnonzero(test)


def _stats(examples, rows, target_indices):
    values = np.asarray(
        [[example["target_physical"][index] for index in target_indices] for example in examples],
        dtype=np.float32,
    )
    means = np.zeros(len(target_indices), dtype=np.float32)
    scales = np.ones(len(target_indices), dtype=np.float32)
    for column in range(len(target_indices)):
        finite = values[rows, column][np.isfinite(values[rows, column])]
        if len(finite):
            means[column] = float(np.median(finite))
            scales[column] = max(float(np.std(finite)), 1e-3)
    return means, scales


def _matrices(examples, rows, target_indices, means, scales):
    features = np.asarray([examples[index]["adapter_features"] for index in rows], dtype=np.float32)
    current = np.asarray(
        [[examples[index]["current_physical"][target] for target in target_indices] for index in rows],
        dtype=np.float32,
    )
    target = np.asarray(
        [[examples[index]["target_physical"][target] for target in target_indices] for index in rows],
        dtype=np.float32,
    )
    mask = np.isfinite(target).astype(np.float32)
    current_z = (np.nan_to_num(current, nan=means) - means) / scales
    target_z = (np.nan_to_num(target, nan=means) - means) / scales
    return features, current_z.astype(np.float32), target_z.astype(np.float32), mask


def _fit_one(examples, train_rows, test_rows, target_indices, epochs, batch_size, seed):
    torch.manual_seed(seed)
    means, scales = _stats(examples, train_rows, target_indices)
    x_train, current_train, y_train, mask_train = _matrices(
        examples, train_rows, target_indices, means, scales
    )
    x_test, current_test, y_test, mask_test = _matrices(
        examples, test_rows, target_indices, means, scales
    )
    model = PureJEPADKATargetHeads(x_train.shape[1], len(target_indices))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = generator.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            prediction = model(
                torch.from_numpy(x_train[batch]).float(),
                torch.from_numpy(current_train[batch]).float(),
            )
            target = torch.from_numpy(y_train[batch]).float()
            mask = torch.from_numpy(mask_train[batch]).float()
            loss_values = nn.functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=0.5
            )
            loss = (loss_values * mask).sum() / mask.sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_z = model(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(current_test).float(),
        ).numpy()
    pred = pred_z * scales + means
    truth = y_test * scales + means
    persistence = current_test * scales + means
    return model, means, scales, pred, truth, persistence, mask_test


def _bootstrap_ci(groups, delta, seed, samples=500):
    unique = np.unique(groups)
    by_group = np.asarray([delta[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(by_group, size=len(by_group), replace=True).mean() for _ in range(samples)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def run(args):
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    frame = pd.read_parquet(args.mimic)
    examples = build_examples(model, frame, device, include_beliefs=False)
    target_indices = tuple(ADAPTED_STATE_INDICES)
    groups = np.asarray([example["stay_id"] for example in examples])
    seed_reports = []
    for seed in SEEDS:
        train_rows, test_rows = _split(groups, seed)
        _, means, scales, prediction, truth, persistence, mask = _fit_one(
            examples,
            train_rows,
            test_rows,
            target_indices,
            args.epochs,
            args.batch_size,
            seed,
        )
        report = {}
        test_groups = groups[test_rows]
        for column, target_index in enumerate(target_indices):
            valid = mask[:, column].astype(bool)
            if valid.sum() < 20:
                continue
            candidate_error = np.abs(prediction[valid, column] - truth[valid, column])
            persistence_error = np.abs(persistence[valid, column] - truth[valid, column])
            delta = candidate_error - persistence_error
            ci = _bootstrap_ci(test_groups[valid], delta, seed + target_index)
            report[STATE_KEYS[target_index]] = {
                "n": int(valid.sum()),
                "candidate_mae": float(candidate_error.mean()),
                "persistence_mae": float(persistence_error.mean()),
                "delta_vs_persistence": float(delta.mean()),
                "bootstrap_95_ci_delta": ci,
                "significant_vs_persistence": bool(ci[1] < 0.0),
            }
        seed_reports.append(report)

    aggregate = {}
    for target in sorted({target for report in seed_reports for target in report}):
        rows = [report[target] for report in seed_reports if target in report]
        aggregate[target] = {
            "seeds_evaluated": len(rows),
            "significant_seeds": sum(row["significant_vs_persistence"] for row in rows),
            "mean_candidate_mae": float(np.mean([row["candidate_mae"] for row in rows])),
            "mean_persistence_mae": float(np.mean([row["persistence_mae"] for row in rows])),
            "validated_patient_gate": bool(
                len(rows) == len(SEEDS)
                and all(row["significant_vs_persistence"] for row in rows)
            ),
        }

    # Fit the saved checkpoint on every usable example. It is still candidate-only;
    # the held-out reports above are the evidence, not the final fit itself.
    all_rows = np.arange(len(examples))
    final_model, means, scales, _, _, _, _ = _fit_one(
        examples,
        all_rows,
        all_rows[:1],
        target_indices,
        args.epochs,
        args.batch_size,
        991,
    )
    checkpoint = Path(args.output_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "feature_dim": int(len(examples[0]["adapter_features"])),
            "target_indices": list(target_indices),
            "target_names": [STATE_KEYS[index] for index in target_indices],
            "target_medians": means.tolist(),
            "target_scales": scales.tolist(),
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "belief_features_used": False,
            "architecture": "frozen measurement-pure JEPA + physiology history + neural target-specific heads",
            "promotion_status": "candidate_only",
        },
        checkpoint,
    )
    result = {
        "test_type": "pure JEPA DKA target-head checkpoint audit",
        "examples": len(examples),
        "stays": int(len(np.unique(groups))),
        "seeds": list(SEEDS),
        "belief_features_used": False,
        "patient_heldout": seed_reports,
        "aggregate": aggregate,
        "checkpoint": str(checkpoint.resolve()),
        "hospital_holdout": {
            "status": "not_run",
            "reason": "transition artifact has no hospital identifier",
        },
        "boundary": {
            "factual_only": True,
            "causal_claims_allowed": False,
            "promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="/private/tmp/dka_pure_jepa_target_heads.json")
    parser.add_argument(
        "--output-checkpoint",
        default="/private/tmp/dka_pure_jepa_target_heads.pt",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

