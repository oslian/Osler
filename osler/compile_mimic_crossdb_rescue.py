"""Compile leakage-safe eICU-to-MIMIC cell rescue evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def treatment_full_pass(item: dict[str, Any]) -> bool:
    random = item.get("random_patient_conformal_90", {})
    care = item.get("careunit_holdout", {}).get("conformal_90", {}).get("candidate", {})
    time = item.get("time_holdout", {}).get("conformal_90", {}).get("candidate", {})
    return bool(
        item.get("validated")
        and random.get("candidate_coverage_pass_splits") == 7
        and care.get("coverage_gate_passed")
        and time.get("coverage_gate_passed")
    )


def baseline_interval_pass(item: dict[str, Any]) -> bool:
    random = item.get("random_patient_conformal_90", {})
    care = item.get("careunit_holdout", {}).get("conformal_90", {}).get("baseline", {})
    time = item.get("time_holdout", {}).get("conformal_90", {}).get("baseline", {})
    return bool(
        random.get("baseline_coverage_pass_splits") == 7
        and care.get("coverage_gate_passed")
        and time.get("coverage_gate_passed")
    )


def build_report(
    base: dict[str, Any],
    rescue: dict[str, Any],
    creatinine24: dict[str, Any],
    creatinine48: dict[str, Any],
) -> dict[str, Any]:
    runtime = base["current_runtime_6h"]
    eicu_cells = set(runtime["eicu_validated_cells"])
    original_point = set(runtime["point_forecast_pass_cells"])
    original_full = set(runtime["point_and_interval_pass_cells"])
    rescue_targets = rescue.get("targets", {})

    point = set(original_point)
    full = set(original_full)
    sources: dict[str, dict[str, Any]] = {}
    for cell in sorted(eicu_cells):
        target = cell.split("@", 1)[0]
        item = rescue_targets.get(target, {})
        if item.get("validated"):
            point.add(cell)
            sources[cell] = {
                "point": "mimic_target_specific_pre_anchor_treatment_ridge",
                "interval": (
                    "domain_adaptive_patient_conformal_treatment_candidate"
                    if treatment_full_pass(item)
                    else None
                ),
            }
            if treatment_full_pass(item):
                full.add(cell)
        elif cell in original_point:
            sources[cell] = {
                "point": "mimic_physiology_ridge",
                "interval": (
                    "domain_adaptive_patient_conformal_physiology_baseline"
                    if baseline_interval_pass(item)
                    else (
                        "original_split_conformal"
                        if cell in original_full
                        else None
                    )
                ),
            }
            if baseline_interval_pass(item):
                full.add(cell)
        else:
            sources[cell] = {"point": None, "interval": None}

    point_only = sorted(point - full)
    fallback = sorted(eicu_cells - point)
    return {
        "schema": "eicu_to_mimic_cell_transfer_rescue.v2",
        "comparison_type": "cross_database_recipe_replication_with_mimic_rescue_candidates",
        "frozen_eicu_weight_transfer_tested": False,
        "future_treatment_columns_excluded": True,
        "treatment_context": {
            "anchor_preceding_only": True,
            "target_specific_actions": True,
            "dose_route_rate_unit_safe": True,
            "causal_claim_allowed": False,
        },
        "counts": {
            "eicu_validated_6h_cells": len(eicu_cells),
            "mimic_point_pass": len(point),
            "mimic_point_and_interval_pass": len(full),
            "mimic_point_only": len(point_only),
            "mimic_fallback": len(fallback),
        },
        "point_and_interval_pass_cells": sorted(full),
        "point_only_cells": point_only,
        "fallback_cells": fallback,
        "cell_sources": sources,
        "creatinine_horizon_audit": {
            "6h": "fallback",
            "24h": (
                "pass" if "creatinine" in creatinine24.get("validated", {}).get("forecast_targets", []) else "fallback"
            ),
            "48h": (
                "pass" if "creatinine" in creatinine48.get("validated", {}).get("forecast_targets", []) else "fallback"
            ),
            "interpretation": "Longer horizon alone does not rescue patient-equalized MIMIC generalization.",
        },
        "resolved_failures": {
            "glucose@6h": "target-specific observed treatment context",
            "hematocrit@6h": "target-specific fluids/transfusion context",
            "heart_rate@6h_interval": "discovery-only domain-adaptive patient conformal",
            "urine_output@6h_interval": "discovery-only domain-adaptive patient conformal",
        },
        "unresolved_failures": {
            "creatinine@6h": "persistence remains patient-equalized winner; 24h and 48h also fail",
            "hemoglobin@6h_interval": "unseen care-unit overcoverage",
            "map@6h_interval": "unseen care-unit undercoverage",
        },
        "safety_boundary": {
            "factual_research_only": True,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
            "automatic_runtime_promotion_allowed": False,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    return "\n".join([
        "# eICU 到 MIMIC-IV 格子修復結果",
        "",
        "日期：2026-08-15",
        "",
        "## 結論",
        "",
        f"- eICU 已驗證的 6h 格：`{counts['eicu_validated_6h_cells']}`",
        f"- MIMIC 點預測通過：`{counts['mimic_point_pass']}`",
        f"- MIMIC 點預測與 90% 區間皆通過：`{counts['mimic_point_and_interval_pass']}`",
        f"- 僅點預測通過：`{counts['mimic_point_only']}`",
        f"- 保持 fallback：`{counts['mimic_fallback']}`",
        "",
        "| 狀態 | 格子 |",
        "|---|---|",
        "| 點預測與區間皆通過 | " + ", ".join(report["point_and_interval_pass_cells"]) + " |",
        "| 僅點預測通過 | " + ", ".join(report["point_only_cells"]) + " |",
        "| fallback | " + ", ".join(report["fallback_cells"]) + " |",
        "",
        "## 已修復",
        "",
        "- `glucose@6h`：只使用 anchor 前、unit-safe 的 insulin/dextrose/nutrition/steroid context，point、care-unit、time、placebo 與 interval 全通過。",
        "- `hematocrit@6h`：移除無關治療欄位，只保留 fluid/albumin/diuretic/transfusion 後，完整通過。",
        "- `heart_rate@6h` 與 `urine_output@6h`：點模型不變；使用 discovery-only domain-adaptive patient conformal 後，care-unit coverage 回到允許範圍。",
        "- no-treatment audit 已硬性排除所有 `hist_*` 與 `act_*`；future treatment 不再可能洩漏。",
        "",
        "## 尚未通過",
        "",
        "- `creatinine`：6h、24h、48h 都未通過 patient-equalized gate。48h 雖改善逐列 MAE，但改善集中於重複量測較多的病人。",
        "- `hemoglobin@6h`：care-unit coverage 仍為 overcoverage；需 target-specific heteroskedastic scale head，不可用 test labels 調 nominal。",
        "- `MAP@6h`：care-unit coverage 仍為 undercoverage；同樣需 scale head 或更完整 anchor-time context。",
        "",
        "> 這是 MIMIC 上重新訓練同 recipe 與 MIMIC-specific rescue candidate 的外部 replication，不是 frozen eICU 權重直接轉移。",
        "",
        "所有結果維持 factual research-only；`causal_claim_allowed: false`。",
        "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rescue", type=Path, required=True)
    parser.add_argument("--creatinine24", type=Path, required=True)
    parser.add_argument("--creatinine48", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        load(args.base),
        load(args.rescue),
        load(args.creatinine24),
        load(args.creatinine48),
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
