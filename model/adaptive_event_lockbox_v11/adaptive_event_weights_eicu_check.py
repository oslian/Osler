"""Check the exported event bundles against the scorer's MIMIC + site fit on a development database (eICU)."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import site_adaptive_event_audit as audit
from adaptive_event_weights import predict
from adaptive_cell_weights import reproduces
from external_value_conditional_interval_audit import EXTERNAL_PREFIX, external_cohort
from mimic_to_eicu_49_cell_external_audit import normalize_cohort
from multi_sample_external_audit import mimic_sample
from site_adaptive_event_dev import static_model

sys.argv = ["x", "--database", "eICU", "--output", "/dev/null"]
args = audit.parse_args()
out = {}
for cell, horizon in (("hypoxemia_event@1h", 1), ("hypoxemia_event@3h", 3)):
    external, _ = external_cohort(args, "hypoxemia_event", horizon)
    full, _ = normalize_cohort(pd.read_parquet(args.mimic_template.format(horizon=horizon)), database="MIMIC-IV",
                               target="hypoxemia_event", horizon=horizon, max_rows=0)
    for sample in range(5):
        frame = pd.concat([mimic_sample(full, target="hypoxemia_event", horizon=horizon, sample=sample, rows=15000), external],
                          ignore_index=True, sort=False)
        frame.sort_values(["subject_id", "anchor_time"], inplace=True); frame.reset_index(drop=True, inplace=True)
        subject = frame["subject_id"].astype(str)
        scored = static_model(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                              np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX["eICU"])), cell)["probability"]
        bundle = joblib.load(f"adaptive_event_weights_20261029/{cell.replace('@', '_')}_s{sample}.joblib")
        got = predict(bundle, frame)[scored.index.to_numpy()]
        diff = float(np.max(np.abs(got - scored.to_numpy())))
        out[f"{cell}_s{sample}"] = {"rows": int(len(scored)), "max_abs_difference": diff, "reproduces": reproduces(got, scored.to_numpy())}
        print(json.dumps({f"{cell}_s{sample}": out[f"{cell}_s{sample}"]}), flush=True)
Path("adaptive_event_weights_20261029_eicu_check.json").write_text(json.dumps(out, indent=2))
