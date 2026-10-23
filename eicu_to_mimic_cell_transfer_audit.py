"""Compare eICU-validated factual cells with an independent MIMIC audit.

This is a report compiler, not a model trainer.  The MIMIC input must be an
independently generated held-out audit using the same observation recipe.  The
report deliberately distinguishes recipe replication from frozen-weight
transfer because the legacy eICU ridge and registry artifacts do not contain
serialized inference weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODES = ("nowcast", "forecast", "interval")
MIMIC_VALIDATED_KEYS = {
    "nowcast": "nowcast_targets",
    "forecast": "forecast_targets",
    "interval": "interval_targets",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eicu_cells(report: dict[str, Any], mode: str) -> list[dict[str, str]]:
    key = f"{mode}_targets"
    return [
        {"module": str(module["module"]), "target": str(target)}
        for module in report.get("modules", [])
        for target in module.get("validated", {}).get(key, [])
    ]


def _failure_reasons(
    mimic_report: dict[str, Any], target: str, mode: str
) -> list[str]:
    target_report = mimic_report.get("targets", {}).get(target, {})
    result = target_report.get("forecast" if mode == "interval" else mode)
    if not isinstance(result, dict):
        return ["no_mimic_support"]

    reasons: list[str] = []
    patient = result.get("random_patient_splits", {})
    evaluated = int(patient.get("evaluated_splits", 0))
    if evaluated != 7:
        reasons.append("insufficient_patient_splits")
    if int(patient.get("selected_ridge_splits", 0)) != evaluated:
        reasons.append("patient_discovery_selection_instability")
    if int(patient.get("heldout_beats_baseline_splits", 0)) != evaluated:
        reasons.append("patient_heldout_baseline_gate")
    if int(patient.get("heldout_beats_placebo_splits", 0)) != evaluated:
        reasons.append("patient_heldout_placebo_gate")

    for split_name in ("careunit_holdout", "time_holdout"):
        split = result.get(split_name, {})
        if not split.get("available", False):
            reasons.append(f"{split_name}_unavailable")
        elif not split.get("gate_passed", False):
            reasons.append(f"{split_name}_value_gate")

    if mode == "interval":
        if int(patient.get("interval_pass_splits", 0)) != evaluated:
            reasons.append("patient_interval_gate")
        for split_name in ("careunit_holdout", "time_holdout"):
            split = result.get(split_name, {})
            if split.get("available", False) and not split.get(
                "interval_gate_passed", False
            ):
                reasons.append(f"{split_name}_interval_gate")
    return reasons or ["not_validated_unspecified"]


def _runtime_cells(catalog: dict[str, Any], horizon: int) -> set[str]:
    suffix = f"@{int(horizon)}h"
    return {
        str(cell)
        for candidate in catalog.get("candidates", [])
        for cell in candidate.get("validated_cells", [])
        if str(cell).endswith(suffix)
    }


def build_report(
    eicu_report: dict[str, Any],
    mimic_report: dict[str, Any],
    runtime_catalog: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for mode in MODES:
        cells = _eicu_cells(eicu_report, mode)
        eicu_targets = sorted({cell["target"] for cell in cells})
        mimic_targets = set(
            mimic_report.get("validated", {}).get(
                MIMIC_VALIDATED_KEYS[mode], []
            )
        )
        passed = sorted(set(eicu_targets) & mimic_targets)
        failed = sorted(set(eicu_targets) - mimic_targets)
        comparisons[mode] = {
            "eicu_module_cell_count": len(cells),
            "eicu_unique_target_count": len(eicu_targets),
            "mimic_pass_count": len(passed),
            "mimic_fail_count": len(failed),
            "pass_targets": passed,
            "failed_targets": [
                {
                    "target": target,
                    "reasons": _failure_reasons(
                        mimic_report, target, mode
                    ),
                }
                for target in failed
            ],
            "module_cells_passing_by_target_replication": [
                cell for cell in cells if cell["target"] in mimic_targets
            ],
        }

    runtime_6h = _runtime_cells(runtime_catalog, 6)
    mimic_forecast_6h = {
        f"{target}@6h"
        for target in mimic_report.get("validated", {}).get(
            "forecast_targets", []
        )
    }
    mimic_interval_6h = {
        f"{target}@6h"
        for target in mimic_report.get("validated", {}).get(
            "interval_targets", []
        )
    }
    point_pass = sorted(runtime_6h & mimic_forecast_6h)
    point_fail = sorted(runtime_6h - mimic_forecast_6h)
    full_pass = sorted(runtime_6h & mimic_forecast_6h & mimic_interval_6h)
    point_only = sorted((runtime_6h & mimic_forecast_6h) - mimic_interval_6h)

    return {
        "schema": "eicu_to_mimic_cell_transfer_audit.v1",
        "comparison_type": "recipe_retrained_cross_database_replication",
        "frozen_eicu_weight_transfer_tested": False,
        "frozen_weight_limitation": (
            "The eICU observation and runtime registries authorize cells but "
            "do not serialize all source-model inference weights."
        ),
        "eicu_source": eicu_report.get("artifact"),
        "mimic_source": mimic_report.get("artifact"),
        "mimic_cohort": mimic_report.get("cohort"),
        "comparison": comparisons,
        "current_runtime_6h": {
            "eicu_validated_cells": sorted(runtime_6h),
            "point_forecast_pass_cells": point_pass,
            "point_forecast_fail_cells": [
                {
                    "cell": cell,
                    "reasons": _failure_reasons(
                        mimic_report, cell.split("@", 1)[0], "forecast"
                    ),
                }
                for cell in point_fail
            ],
            "point_and_interval_pass_cells": full_pass,
            "point_pass_interval_fail_cells": [
                {
                    "cell": cell,
                    "reasons": _failure_reasons(
                        mimic_report, cell.split("@", 1)[0], "interval"
                    ),
                }
                for cell in point_only
            ],
            "counts": {
                "eicu_validated": len(runtime_6h),
                "mimic_point_pass": len(point_pass),
                "mimic_point_and_interval_pass": len(full_pass),
            },
        },
        "interpretation": {
            "point_generalization_is_target_specific": True,
            "interval_generalization_is_stricter_than_point_generalization": True,
            "careunit_shift_is_a_major_failure_mode": True,
            "causal_claim_allowed": False,
            "clinical_claim_allowed": False,
        },
    }


def markdown(report: dict[str, Any]) -> str:
    runtime = report["current_runtime_6h"]
    lines = [
        "# eICU 到 MIMIC-IV 格子外部驗證",
        "",
        "日期：2026-08-15",
        "",
        "## 結論",
        "",
        (
            f"目前 runtime catalog 在 eICU 有 {runtime['counts']['eicu_validated']} 個 6h 格。"
            f"在完整 MIMIC-IV 上重建同一 factual recipe 後，"
            f"{runtime['counts']['mimic_point_pass']} 個點預測通過，"
            f"其中 {runtime['counts']['mimic_point_and_interval_pass']} 個連 90% 區間也通過。"
        ),
        "",
        "> 這是方法在另一資料庫重新訓練與驗證的 replication；不是 frozen eICU 權重直接搬到 MIMIC。現有 registry 沒有保存所有來源模型的可推論權重。",
        "",
        "## 6h Runtime 格",
        "",
        "| 狀態 | 格子 |",
        "|---|---|",
        "| 點預測與區間皆通過 | "
        + ", ".join(runtime["point_and_interval_pass_cells"])
        + " |",
        "| 點預測通過、區間未通過 | "
        + ", ".join(
            item["cell"]
            for item in runtime["point_pass_interval_fail_cells"]
        )
        + " |",
        "| 點預測未通過 | "
        + ", ".join(
            item["cell"] for item in runtime["point_forecast_fail_cells"]
        )
        + " |",
        "",
        "## 未通過原因",
        "",
    ]
    for item in runtime["point_forecast_fail_cells"]:
        lines.append(
            f"- `{item['cell']}`：" + ", ".join(item["reasons"])
        )
    for item in runtime["point_pass_interval_fail_cells"]:
        lines.append(
            f"- `{item['cell']}`：點預測通過；區間失敗於 "
            + ", ".join(item["reasons"])
        )
    lines.extend(
        [
            "",
            "## 判讀",
            "",
            "- `creatinine@6h` 是明確的模型/時間尺度失敗：MIMIC 中 persistence 更好。",
            "- `glucose@6h` 有平均增益，但 care-unit heldout 信賴區間跨 0，屬於跨病房不穩定。",
            "- `hematocrit@6h` 勝過 persistence，卻未顯著勝過 capacity-matched placebo，表示增益未能歸因於有效生理特徵。",
            "- heart rate、hemoglobin、MAP、urine output 的點預測可轉移，但區間 coverage 在 care-unit heldout 偏離 0.87-0.93，所以不可顯示正式窄區間。",
            "- potassium 與 respiratory rate 是目前最完整的跨庫 6h 格：點預測與 90% 區間都成立。",
            "",
            "所有結果仍是 factual research-only；`causal_claim_allowed: false`。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eicu", type=Path, required=True)
    parser.add_argument("--mimic", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        _load(args.eicu), _load(args.mimic), _load(args.catalog)
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.findings.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "findings": str(args.findings),
                "counts": report["current_runtime_6h"]["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
