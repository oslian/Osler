#!/bin/zsh
# Step 2b: same frozen form, DtACI updated against current thresholds under delayed feedback (no other setting changed).
# on the rebuilt data with the leak-free replay.  Serial (17 GB machine).  Old results stay as history.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
$PY site_adaptive_external_audit.py --database eICU --output site_adaptive_reeval_eICU_dtacifix_20261023.json > site_adaptive_reeval_eICU_dtacifix_20261023.log 2>&1
echo "eICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" \
  --output site_adaptive_reeval_NWICU_dtacifix_20261023.json > site_adaptive_reeval_NWICU_dtacifix_20261023.log 2>&1
echo "NWICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database Zigong --external-template "zigong_observation_transitions_{horizon}h_v2.parquet" \
  --output site_adaptive_reeval_Zigong_dtacifix_20261023.json > site_adaptive_reeval_Zigong_dtacifix_20261023.log 2>&1
echo "Zigong exit $? $(date)"
$PY site_adaptive_external_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" \
  --output site_adaptive_reeval_Emory_dtacifix_20261023.json > site_adaptive_reeval_Emory_dtacifix_20261023.log 2>&1
echo "Emory exit $? $(date)"
echo REEVAL_DONE
