#!/bin/zsh
# v11 one-shot lockbox scoring of the binary event cells (user-approved 2026-10-29). Serial: SICdb, then HiRID.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa || exit 1
P=adaptive_event_lockbox_protocol_v11_20261029.json
.venv/bin/python site_adaptive_event_audit.py --database SICdb --external-template "sicdb_observation_transitions_{horizon}h_v1.parquet" \
  --protocol $P --output site_adaptive_event_lockbox_SICdb_v11.json > site_adaptive_event_lockbox_SICdb_v11.log 2>&1
echo "SICdb exit $?"
.venv/bin/python site_adaptive_event_audit.py --database HiRID --external-template "hirid_observation_transitions_{horizon}h_v1.parquet" \
  --protocol $P --output site_adaptive_event_lockbox_HiRID_v11.json > site_adaptive_event_lockbox_HiRID_v11.log 2>&1
echo "HiRID exit $?"
