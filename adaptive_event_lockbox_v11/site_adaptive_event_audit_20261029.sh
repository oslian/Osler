#!/bin/zsh
# Development-database evaluation of the binary event cells with online Platt scaling
# (post-hoc amendment site_adaptive_event_amendment_20261028.json); v2 after the Codex pre-freeze review
# (diagnostics, invalid-prediction handling, lockbox guards).  Strictly serial.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
$PY site_adaptive_event_audit.py --database eICU --output site_adaptive_event_eICU_20261029.json > site_adaptive_event_eICU_20261029.log 2>&1
$PY site_adaptive_event_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" \
  --output site_adaptive_event_NWICU_20261029.json > site_adaptive_event_NWICU_20261029.log 2>&1
$PY site_adaptive_event_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" \
  --output site_adaptive_event_Emory_20261029.json > site_adaptive_event_Emory_20261029.log 2>&1
echo "done $(date)"
