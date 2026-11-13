"""Train the pure-JEPA body-system candidate across available eICU modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_jepa_body_system import train_module


DEFAULT_MODULES = (
    "aki",
    "acute_neuro",
    "cardiac_injury",
    "cardiovascular_instability",
    "coagulopathy_heme",
    "electrolyte_acid_base",
    "endocrine_stress",
    "gi_pancreatic_nutrition",
    "hepatic_failure",
    "immune_inflammatory",
    "integumentary_skin_wound",
    "musculoskeletal_rhabdo",
    "respiratory",
    "sepsis",
    "toxic_metabolic",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-stays-per-horizon", type=int, default=1500)
    parser.add_argument("--output-dir", default="/private/tmp/pure_jepa_body_systems")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for module in [value.strip() for value in args.modules.split(",") if value.strip()]:
        print(f"\n=== pure JEPA module: {module} ===")
        report_path = output_dir / f"{module}.json"
        checkpoint_path = output_dir / f"{module}.pt"
        try:
            report = train_module(
                args.data_root,
                module,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_stays_per_horizon=args.max_stays_per_horizon,
                checkpoint=checkpoint_path,
            )
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            summaries[module] = {
                "status": "trained",
                "targets": len(report.get("variables", [])),
                "rows": report.get("rows"),
                "patient_heldout": report.get("patient_heldout", {}),
                "checkpoint": str(checkpoint_path),
            }
        except (FileNotFoundError, KeyError, ValueError) as exc:
            summaries[module] = {"status": "not_available", "error": str(exc)}
            print(f"  skipped: {exc}")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()

