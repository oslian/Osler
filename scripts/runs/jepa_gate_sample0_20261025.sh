#!/bin/zsh
# JEPA under jepa_gate_rule_20261022.json (revised 2026-10-24), MIMIC sample 0 only (development run;
# promotion needs 5 samples).  Strictly serial (17 GB RAM): anchors -> 7-seed training (resumed after
# OOM kills) -> per-database gate.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
EXAMPLES=four_db_jepa_examples_harmonized_anchor_20261025.parquet
REGISTRY=expert_anchor_registry_core6_four_db_20261025.json
AUDIT=jepa_four_db_sample0_20261025.json
PREDICTIONS=jepa_four_db_sample0_predictions_20261025
MODULES=heart,blood_vessels,lungs,kidneys,liver,pancreas,stomach,intestines,thyroid,adrenal_glands,bone_marrow,spleen,brain,skeletal_muscle

if [[ ! -f $EXAMPLES ]]; then
  $PY build_expert_anchor_columns.py --examples four_db_jepa_examples_harmonized.parquet \
    --targets glucose,bicarbonate,creatinine,potassium,map,heart_rate \
    --output $EXAMPLES --registry $REGISTRY > expert_anchor_four_db_20261025.log 2>&1 || { echo "anchors failed"; exit 1; }
  echo "anchors done $(date)"
fi

for attempt in {1..30}; do
  $PY teacher_anchored_joint_jepa_audit.py --event-examples $EXAMPLES --modules $MODULES \
    --split-mode database --train-database mimiciv --test-database eicu,nwicu,zigong \
    --seeds 7,19,31,43,57,71,97 --max-stays 20000 --expert-anchor-registry $REGISTRY \
    --export-predictions $PREDICTIONS --output $AUDIT --resume >> jepa_four_db_sample0_20261025.log 2>&1
  rc=$?
  echo "training attempt $attempt exit $rc $(date)"
  [[ $rc -eq 0 ]] && break
  [[ $rc -eq 137 ]] || { echo "training failed"; exit 1; }
done
[[ $rc -eq 0 ]] || { echo "training did not finish"; exit 1; }

$PY jepa_gate.py --runs "$AUDIT:$PREDICTIONS" --output jepa_gate_sample0_20261025.json > jepa_gate_sample0_20261025.log 2>&1 || { echo "gate failed"; exit 1; }
echo "JEPA_SAMPLE0_DONE $(date)"
