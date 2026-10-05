#!/usr/bin/env python3
"""Cache fixed-width raw waveform tensors outside the repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_jepa.raw_waveform import read_raw_waveform_tensor


def run(args: argparse.Namespace) -> dict[str, object]:
    cohort = pd.read_parquet(args.cohort)
    past_scales = tuple(int(value) for value in args.past_scales.split(","))
    tensors = []
    row_indices = []
    failures = 0
    for position, row in cohort.iterrows():
        try:
            tensor = read_raw_waveform_tensor(
                row["segment_path"],
                int(row["anchor_sample"]),
                past_scales_seconds=past_scales,
                points_per_scale=args.points_per_scale,
                future_seconds=args.future_seconds,
            )
        except Exception:
            failures += 1
            continue
        tensors.append(tensor)
        row_indices.append(int(position))
        if len(row_indices) % 100 == 0:
            print(
                f"cached={len(row_indices)} scanned={position + 1} failures={failures}",
                flush=True,
            )
    if not tensors:
        raise ValueError("no raw waveform tensors were cached")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        row_index=np.asarray(row_indices, dtype=np.int64),
        past=np.stack([item["past"] for item in tensors]),
        future=np.stack([item["future"] for item in tensors]),
        past_mask=np.stack([item["past_mask"] for item in tensors]),
        future_mask=np.stack([item["future_mask"] for item in tensors]),
        quality=np.stack([item["quality"] for item in tensors]),
    )
    quality = np.stack([item["quality"] for item in tensors])
    report = {
        "schema": "whole_body_jepa.raw_waveform_cache.v1",
        "source_rows": int(len(cohort)),
        "cached_rows": int(len(row_indices)),
        "patients": int(cohort.loc[row_indices, "subject_id"].nunique()),
        "failures": int(failures),
        "points_per_scale": int(args.points_per_scale),
        "past_scales_seconds": list(past_scales),
        "future_seconds": int(args.future_seconds),
        "past_shape": list(tensors[0]["past"].shape),
        "future_shape": list(tensors[0]["future"].shape),
        "median_valid_fraction": float(np.median(quality[..., 0])),
        "row_level_cache_committable": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--points-per-scale", type=int, default=256)
    parser.add_argument("--past-scales", default="60,300,900")
    parser.add_argument("--future-seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
