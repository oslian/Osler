#!/bin/zsh
# Re-evaluation of the binary event cells (hypoxemia_event@1h/3h) on the leak-fixed data
# (step 2 of the 2026-10-22 fix order covered continuous cells only).  Unchanged static
# recipe and gate (multi_sample_external_audit.py, robust = full pass in >= 80% of 5 samples).
# Development databases only (non-independent).  Strictly serial (17 GB RAM).
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
COMMON=(--mimic-template "mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet"
        --cells hypoxemia_event@1h,hypoxemia_event@3h --samples 5)
$PY multi_sample_external_audit.py --database eICU --eicu-root eicu_harmonized_20261022 $COMMON \
  --output multi_sample_eICU_binary_leakfix_20261028.json > multi_sample_eICU_binary_leakfix_20261028.log 2>&1
$PY multi_sample_external_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" $COMMON \
  --output multi_sample_NWICU_binary_leakfix_20261028.json > multi_sample_NWICU_binary_leakfix_20261028.log 2>&1
$PY multi_sample_external_audit.py --database Zigong --external-template "zigong_observation_transitions_{horizon}h_v2.parquet" $COMMON \
  --output multi_sample_Zigong_binary_leakfix_20261028.json > multi_sample_Zigong_binary_leakfix_20261028.log 2>&1
$PY multi_sample_external_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" $COMMON \
  --output multi_sample_Emory_binary_leakfix_20261028.json > multi_sample_Emory_binary_leakfix_20261028.log 2>&1
echo "done $(date)"
