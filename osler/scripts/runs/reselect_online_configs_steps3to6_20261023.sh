#!/bin/zsh
# Online-config re-selection steps 3-5 (online_config_reselection_protocol_20261023.json) and the single
# re-evaluation of the re-selected form on the four development databases.  Strictly serial (17 GB RAM);
# waits for steps 0-2.  Worker count does not change results (seeded per cell and sample).
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
until grep -q "STEPS_0_2_DONE\|STOP\|failed" reselect_online_configs_steps0to2_20261023.log 2>/dev/null; do sleep 60; done
grep -q STEPS_0_2_DONE reselect_online_configs_steps0to2_20261023.log || { echo "steps 0-2 did not finish"; exit 1; }
POINT=site_adaptive_point_dev_20261023.json
DTACI=site_adaptive_dtaci_dev_20261023.json
PRIOR=site_adaptive_dtaci_prior_dev_20261023.json
NETWORK=site_adaptive_network_dev_20261023.json
RECIPES=deployment_recipes_20261023.json

# Step 3: recipes (catalog cells, then the glucose_poc extension), merged into the deployment recipes.
$PY external_recipe_selection_dev.py --point-dev $POINT --dtaci-dev $DTACI --workers 2 \
  --output external_recipe_selection_dev_20261023.json > external_recipe_selection_dev_20261023.log 2>&1 || { echo "step 3 failed"; exit 1; }
$PY external_recipe_selection_dev.py --development-report glucose_poc_catalog_extension_20261020.json --point-dev $POINT --dtaci-dev $DTACI \
  --workers 2 --output glucose_poc_recipe_selection_dev_20261023.json > glucose_poc_recipe_selection_dev_20261023.log 2>&1 || { echo "step 3 poc failed"; exit 1; }
$PY -c "
import json
sources = ['external_recipe_selection_dev_20261023.json', 'glucose_poc_recipe_selection_dev_20261023.json']
recipes = {cell: recipe for path in sources for cell, recipe in json.load(open(path))['recipes'].items()}
json.dump({'schema': 'deployment_recipes.v1', 'sources': sources, 'recipes': recipes}, open('$RECIPES', 'w'), indent=2)
print(len(recipes))" || exit 1
echo "step 3 done $(date)"

# Step 4: theta0 (leave-one-database-out), step 5: network-shared DtACI (eICU halves).
$PY site_adaptive_dtaci_prior_dev.py --recipes external_recipe_selection_dev_20261023.json --point-dev $POINT --dtaci-dev $DTACI \
  --workers 2 --output $PRIOR > site_adaptive_dtaci_prior_dev_20261023.log 2>&1 || { echo "step 4 failed"; exit 1; }
echo "step 4 theta0 $($PY -c "import json; print(json.load(open('$PRIOR'))['selected_theta0'])") $(date)"
$PY site_adaptive_network_dev.py --recipes external_recipe_selection_dev_20261023.json --point-dev $POINT --dtaci-dev $DTACI \
  --dtaci-prior $PRIOR --workers 2 --output $NETWORK > site_adaptive_network_dev_20261023.log 2>&1 || { echo "step 5 failed"; exit 1; }
echo "step 5 network $($PY -c "import json; print(json.load(open('$NETWORK'))['selected'])") $(date)"

# One re-evaluation of the re-selected form.
FORM=(--point-dev $POINT --dtaci-dev $DTACI --dtaci-prior $PRIOR --network-dev $NETWORK --recipes $RECIPES)
$PY site_adaptive_external_audit.py --database eICU $FORM --output site_adaptive_reselected_eICU_20261023.json > site_adaptive_reselected_eICU_20261023.log 2>&1
echo "eICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" $FORM \
  --output site_adaptive_reselected_NWICU_20261023.json > site_adaptive_reselected_NWICU_20261023.log 2>&1
echo "NWICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database Zigong --external-template "zigong_observation_transitions_{horizon}h_v2.parquet" $FORM \
  --output site_adaptive_reselected_Zigong_20261023.json > site_adaptive_reselected_Zigong_20261023.log 2>&1
echo "Zigong exit $? $(date)"
$PY site_adaptive_external_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" $FORM \
  --output site_adaptive_reselected_Emory_20261023.json > site_adaptive_reselected_Emory_20261023.log 2>&1
echo "Emory exit $? $(date)"
echo RESELECTION_DONE
