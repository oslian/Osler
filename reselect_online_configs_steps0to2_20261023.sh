#!/bin/zsh
# Online-config re-selection steps 0-2 (online_config_reselection_protocol_20261023.json), strictly serial.
# Step 0 was started separately; steps 1-2 use the fixed-step tracker hard-coded as FROZEN_CONFIG,
# so they run only if step 0 re-selected that same tracker.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
while pgrep -f "site_adaptive_interval_dev.py" > /dev/null; do sleep 30; done
selected=$($PY -c "import json; print(json.load(open('site_adaptive_interval_dev_20261023.json'))['selected'])") || { echo "step 0 failed"; exit 1; }
echo "step 0 selected $selected $(date)"
[[ "$selected" == "weight_site_unit_0.02" ]] || { echo "STOP: tracker changed; update FROZEN_CONFIG before steps 1-2"; exit 2; }
$PY site_adaptive_point_dev.py --output site_adaptive_point_dev_20261023.json > site_adaptive_point_dev_20261023.log 2>&1 || { echo "step 1 failed"; exit 1; }
echo "step 1 selected $($PY -c "import json; print(json.load(open('site_adaptive_point_dev_20261023.json'))['selected'])") $(date)"
$PY site_adaptive_dtaci_dev.py --point-dev site_adaptive_point_dev_20261023.json --output site_adaptive_dtaci_dev_20261023.json > site_adaptive_dtaci_dev_20261023.log 2>&1 || { echo "step 2 failed"; exit 1; }
echo "step 2 selected $($PY -c "import json; print(json.load(open('site_adaptive_dtaci_dev_20261023.json'))['selected'])") $(date)"
echo STEPS_0_2_DONE
