#!/bin/zsh
# Step 2 of the Codex fix order: re-evaluate the frozen deployment form (same recipes and online configs as v8)
# on the rebuilt data with the leak-free replay.  Serial (17 GB machine).  Old results stay as history.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
$PY site_adaptive_external_audit.py --database eICU --output site_adaptive_reeval_eICU_20261022.json > site_adaptive_reeval_eICU_20261022.log 2>&1
echo "eICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" \
  --output site_adaptive_reeval_NWICU_20261022.json > site_adaptive_reeval_NWICU_20261022.log 2>&1
echo "NWICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database Zigong --external-template "zigong_observation_transitions_{horizon}h_v2.parquet" \
  --output site_adaptive_reeval_Zigong_20261022.json > site_adaptive_reeval_Zigong_20261022.log 2>&1
echo "Zigong exit $? $(date)"
$PY site_adaptive_external_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" \
  --output site_adaptive_reeval_Emory_20261022.json > site_adaptive_reeval_Emory_20261022.log 2>&1
echo "Emory exit $? $(date)"
echo REEVAL_DONE
