import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from causal_iv_analysis import design, estimate
from external_value_conditional_interval_audit import sha256


def synthetic(folder: Path, effect: float) -> dict:
    rng = np.random.default_rng(0)
    n_clusters, per = 30, 300
    cluster = np.repeat(np.arange(n_clusters), per)
    preference = rng.uniform(0.05, 0.6, n_clusters)[cluster]
    severity = rng.normal(size=len(cluster))
    baseline = 105 + 2 * severity + rng.normal(size=len(cluster))
    treated = (rng.uniform(size=len(cluster)) < np.clip(preference + 0.1 * severity, 0, 1)).astype(int)
    day1 = baseline + effect * treated + 1.5 * severity + rng.normal(size=len(cluster))
    platelets = 200 + rng.normal(0, 30, len(cluster))
    frame = pd.DataFrame({"stay_key": np.arange(len(cluster)), "cluster_id": cluster.astype(str), "subject_id": np.arange(len(cluster)),
                          "treated_balanced": treated, "age": rng.normal(60, 10, len(cluster)), "gender": rng.choice(["F", "M"], len(cluster)),
                          "baseline_chloride": baseline, "baseline_platelets": platelets})
    total = frame.groupby("cluster_id")["treated_balanced"].transform("sum")
    count = frame.groupby("cluster_id")["treated_balanced"].transform("size")
    frame["instrument"] = (total - frame["treated_balanced"]) / (count - 1)
    frame.to_parquet(folder / "design.parquet")
    pd.DataFrame({"stay_key": frame["stay_key"], "day1_chloride": day1,
                  "day1_platelets": platelets + rng.normal(0, 10, len(cluster))}).to_parquet(folder / "outcomes.parquet")
    return {"design_table": str(folder / "design.parquet"), "outcome_table": str(folder / "outcomes.parquet"), "key": "stay_key",
            "cluster": "cluster_id", "sensitivity_cluster": "subject_id", "instrument": "instrument", "treatment": "treated_balanced",
            "numeric_covariates": ["age", "baseline_chloride", "baseline_platelets"], "categorical_covariates": ["gender"],
            "outcomes": ["chloride", "platelets"], "primary_outcome": "chloride", "negative_control_outcome": "platelets",
            "benchmark_range": [-8.0, -0.3], "bootstrap_replicates": 100}


class CausalIvAnalysisTests(unittest.TestCase):
    def run_case(self, effect: float) -> dict:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            spec = synthetic(folder, effect)
            report = design(spec)
            self.assertGreater(report["first_stage_f_adjusted"], 10)
            protocol = folder / "protocol.json"
            protocol.write_text(json.dumps({"frozen_code_sha256": {spec["design_table"]: sha256(Path(spec["design_table"]))}}))
            return estimate(spec, protocol)

    def test_known_negative_effect_passes_the_benchmark_but_still_needs_review(self):
        report = self.run_case(-3.0)
        self.assertTrue(report["readiness_checks"]["benchmark_reproduced"])
        self.assertTrue(report["readiness_checks"]["negative_control_null"])
        self.assertEqual(report["readiness_status"], "ready_for_external_review")
        self.assertFalse(report["causal_claim_allowed"])

    def test_null_effect_fails_the_benchmark(self):
        report = self.run_case(0.0)
        self.assertFalse(report["readiness_checks"]["benchmark_reproduced"])
        self.assertEqual(report["readiness_status"], "failed")


if __name__ == "__main__":
    unittest.main()
