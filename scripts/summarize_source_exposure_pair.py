"""Separate original core evidence from incremental source-expansion evidence."""
import json
from pathlib import Path
import numpy as np
from scripts.audit_confirmed_reconstruction_heads import confirmed
from whole_body_jepa.multisource_intervals import sha256


def main():
    source = Path("artifacts/native_reconstruction_source_exposure_evaluation_20260918.json")
    out = Path("artifacts/native_reconstruction_source_exposure_summary_20260919.json")
    if out.exists():
        raise ValueError("Summary exists")
    report = json.loads(source.read_text())
    result = {}
    for group, targets in report["groups"].items():
        supported = {t: r for t, r in targets.items() if "mae" in r}
        core, incremental = [], []
        for t, r in supported.items():
            checks = {k: v for k, v in r["comparisons"].items() if k != "small_real"}
            if r["source_supported"] and confirmed(checks, r["patients"]):
                core.append(t)
            if r["development_gate"]:
                incremental.append(t)
        macro = {k: float(np.mean([r["mae"][k] for r in supported.values()]))
                 for k in ("old", "small_real", "expanded_real", "expanded_population", "expanded_process", "expanded_placebo")}
        result[group] = dict(core_pass=core, core_plus_incremental_pass=incremental,
            macro=macro, better_small=sum(r["mae"]["expanded_real"] < r["mae"]["small_real"] for r in supported.values()),
            better_old=sum(r["mae"]["expanded_real"] < r["mae"]["old"] for r in supported.values()),
            relative_to_small=macro["expanded_real"] / macro["small_real"] - 1,
            relative_to_old=macro["expanded_real"] / macro["old"] - 1)
    out.write_text(json.dumps(dict(groups=result, bindings={str(p): sha256(p) for p in
        (source, Path(__file__), Path("scripts/audit_confirmed_reconstruction_heads.py"))},
        runtime_modified=False, promotion_allowed=False,
        limitation="Core excludes the added small-source comparator; both must be reported. Reused eICU cohorts, not external or future validation."), indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
