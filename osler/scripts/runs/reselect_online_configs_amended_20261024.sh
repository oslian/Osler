#!/bin/zsh
# Amended re-selection (online_config_reselection_protocol_20261023.json, amendment 2026-10-24, post hoc):
# point correction affine_0.005 (site_adaptive_point_amendment_20261024.json); steps 2-5 re-run with their
# unchanged rules, then one re-evaluation of the four development databases.  Strictly serial (17 GB RAM).
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
POINT=site_adaptive_point_amendment_20261024.json
DTACI=site_adaptive_dtaci_dev_20261024.json
PRIOR=site_adaptive_dtaci_prior_dev_20261024.json
NETWORK=site_adaptive_network_dev_20261024.json
RECIPES=deployment_recipes_20261024.json

# Step 2: DtACI meta rate (MIMIC only) under the amended point correction.
$PY site_adaptive_dtaci_dev.py --point-dev $POINT --output $DTACI > site_adaptive_dtaci_dev_20261024.log 2>&1 || { echo "step 2 failed"; exit 1; }
echo "step 2 selected $($PY -c "import json; print(json.load(open('$DTACI'))['selected'])") $(date)"

# Step 3: recipes (catalog cells, then the glucose_poc extension), merged into the deployment recipes.
$PY external_recipe_selection_dev.py --point-dev $POINT --dtaci-dev $DTACI --workers 2 \
  --output external_recipe_selection_dev_20261024.json > external_recipe_selection_dev_20261024.log 2>&1 || { echo "step 3 failed"; exit 1; }
$PY external_recipe_selection_dev.py --development-report glucose_poc_catalog_extension_20261020.json --point-dev $POINT --dtaci-dev $DTACI \
  --workers 2 --output glucose_poc_recipe_selection_dev_20261024.json > glucose_poc_recipe_selection_dev_20261024.log 2>&1 || { echo "step 3 poc failed"; exit 1; }
$PY -c "
import json
sources = ['external_recipe_selection_dev_20261024.json', 'glucose_poc_recipe_selection_dev_20261024.json']
recipes = {cell: recipe for path in sources for cell, recipe in json.load(open(path))['recipes'].items()}
json.dump({'schema': 'deployment_recipes.v1', 'sources': sources, 'recipes': recipes}, open('$RECIPES', 'w'), indent=2)
print(len(recipes))" || exit 1
echo "step 3 done $(date)"

# Step 4: theta0 (leave-one-database-out), step 5: network-shared DtACI (eICU halves).
$PY site_adaptive_dtaci_prior_dev.py --recipes external_recipe_selection_dev_20261024.json --point-dev $POINT --dtaci-dev $DTACI \
  --workers 2 --output $PRIOR > site_adaptive_dtaci_prior_dev_20261024.log 2>&1 || { echo "step 4 failed"; exit 1; }
echo "step 4 theta0 $($PY -c "import json; print(json.load(open('$PRIOR'))['selected_theta0'])") $(date)"
$PY site_adaptive_network_dev.py --recipes external_recipe_selection_dev_20261024.json --point-dev $POINT --dtaci-dev $DTACI \
  --dtaci-prior $PRIOR --workers 2 --output $NETWORK > site_adaptive_network_dev_20261024.log 2>&1 || { echo "step 5 failed"; exit 1; }
echo "step 5 network $($PY -c "import json; print(json.load(open('$NETWORK'))['selected'])") $(date)"

# One re-evaluation of the re-selected form.
FORM=(--point-dev $POINT --dtaci-dev $DTACI --dtaci-prior $PRIOR --network-dev $NETWORK --recipes $RECIPES)
$PY site_adaptive_external_audit.py --database eICU $FORM --output site_adaptive_amended_eICU_20261024.json > site_adaptive_amended_eICU_20261024.log 2>&1
echo "eICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database NWICU --external-template "nwicu_observation_transitions_{horizon}h_v5.parquet" $FORM \
  --output site_adaptive_amended_NWICU_20261024.json > site_adaptive_amended_NWICU_20261024.log 2>&1
echo "NWICU exit $? $(date)"
$PY site_adaptive_external_audit.py --database Zigong --external-template "zigong_observation_transitions_{horizon}h_v2.parquet" $FORM \
  --output site_adaptive_amended_Zigong_20261024.json > site_adaptive_amended_Zigong_20261024.log 2>&1
echo "Zigong exit $? $(date)"
$PY site_adaptive_external_audit.py --database Emory --external-template "emory_challenge2019_transitions_{horizon}h_v2.parquet" $FORM \
  --output site_adaptive_amended_Emory_20261024.json > site_adaptive_amended_Emory_20261024.log 2>&1
echo "Emory exit $? $(date)"
echo AMENDED_RESELECTION_DONE
