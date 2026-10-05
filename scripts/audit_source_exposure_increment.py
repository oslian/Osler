"""Paired source expansion: separate prior shift from conditional error change."""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_broad_prior_increment import decompose
from whole_body_jepa.multisource_intervals import sha256


def main():
    source = Path("artifacts/native_reconstruction_source_exposure_evaluation_20260918.json")
    out = Path("artifacts/native_reconstruction_source_exposure_increment_20260919.json")
    if out.exists():
        raise ValueError("Audit exists")
    report = json.loads(source.read_text())
    for p, h in report["bindings"].items():
        if sha256(Path(p)) != h:
            raise ValueError("Evaluation binding changed")
    results = {}
    for group, targets in report["groups"].items():
        path = Path("data/research_recovery/broad_source_20260917/source_exposure_evaluation") / f"{group}.pt"
        rows = torch.load(path, weights_only=False)
        if len({(r["patient"], r["target"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate patient-target")
        per = {}
        for target, summary in targets.items():
            rr = [r for r in rows if r["target"] == target]
            if not rr:
                per[target] = dict(status="no_trained_source_head")
                continue

            def score(part):
                return decompose(*[[r[k] for r in part] for k in
                    ("truth", "small_real", "expanded_real", "small_population", "expanded_population")])

            overall = score(rr)
            expected = summary["mae"]["expanded_real"] - summary["mae"]["small_real"]
            if not np.isclose(overall["total"], expected, rtol=0, atol=1e-10):
                raise ValueError("Paired error replay mismatch")
            sites = [score([r for r in rr if r["site"] == site]) for site in sorted({r["site"] for r in rr})]
            per[target] = dict(patients=len(rr), pooled=overall, sites=len(sites),
                equal_site={k: float(np.mean([s[k] for s in sites])) for k in overall},
                sites_conditional_gain=sum(s["incremental_change"] < 0 for s in sites),
                sites_beats_prior=sum(s["new_increment"] < 0 for s in sites))
        result = [r for r in per.values() if "pooled" in r]
        results[group] = dict(targets=per,
            macro={k: float(np.mean([r["pooled"][k] for r in result])) for k in result[0]["pooled"]},
            conditional_improves=sum(r["pooled"]["incremental_change"] < 0 for r in result),
            beats_own_prior=sum(r["pooled"]["new_increment"] < 0 for r in result))
    out.write_text(json.dumps(dict(groups=results, descriptive_only=True,
        bindings={str(p): sha256(p) for p in (source, Path(__file__), Path("scripts/audit_broad_prior_increment.py"))},
        runtime_modified=False, promotion_allowed=False,
        limitation="Exact error identity, not causal decomposition. Site means include sparse sites; no test-informed tuning."), indent=2) + "\n")
    for group, result in results.items():
        print(group, result["macro"], "conditional improves", result["conditional_improves"],
              "beats own prior", result["beats_own_prior"])
        for t in ("glucose", "bun", "creatinine", "alt", "ast", "lipase", "troponin_i"):
            print(t, result["targets"][t]["pooled"])


if __name__ == "__main__":
    main()
