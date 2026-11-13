#!/bin/zsh
# JEPA MIMIC samples 1-4 under jepa_mimic_samples_protocol_20261025.json, then the gate over samples 0-4.
# Strictly serial (17 GB RAM).  Per sample: MIMIC ledger -> event examples -> four-DB harmonized examples
# -> anchors (recipe sample k) -> 7-seed training (resumed after OOM kills).  Rebuildable intermediates are
# deleted once the anchored examples exist (disk is nearly full).
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
MIMIC_ROOT=$HOME/Downloads/mimic-iv-3.1
TARGETS=glucose,bicarbonate,creatinine,potassium,map,heart_rate
MODULES=heart,blood_vessels,lungs,kidneys,liver,pancreas,stomach,intestines,thyroid,adrenal_glands,bone_marrow,spleen,brain,skeletal_muscle
RUNS=(jepa_four_db_sample0_20261025.json:jepa_four_db_sample0_predictions_20261025)

for k in 1 2 3 4; do
  LEDGER=mimiciv_canonical_event_ledger_5000_s$k
  MIMIC_EX=mimiciv_canonical_event_examples_5000_s$k.parquet
  FOUR_DB=four_db_jepa_examples_harmonized_s$k.parquet
  EXAMPLES=four_db_jepa_examples_harmonized_anchor_s${k}_20261025.parquet
  REGISTRY=expert_anchor_registry_core6_four_db_s${k}_20261025.json
  AUDIT=jepa_four_db_sample${k}_20261025.json
  PREDICTIONS=jepa_four_db_sample${k}_predictions_20261025
  LOG=jepa_four_db_sample${k}_20261025.log

  if [[ ! -f $EXAMPLES ]]; then
    $PY canonical_event_ledger_extract.py --database mimiciv --data-root $MIMIC_ROOT --max-stays 5000 --sample $k \
      --output-dir $LEDGER --report ${LEDGER}_report.json > $LOG 2>&1 || { echo "sample $k ledger failed"; exit 1; }
    $PY canonical_event_examples.py --event-data $LEDGER --output $MIMIC_EX --report ${MIMIC_EX:r}_report.json >> $LOG 2>&1 \
      || { echo "sample $k examples failed"; exit 1; }
    $PY prepare_jepa_event_examples.py --inputs $MIMIC_EX eicu_canonical_event_examples_full_5000.parquet \
      nwicu_canonical_event_examples.parquet zigong_canonical_event_examples.parquet --output $FOUR_DB >> $LOG 2>&1 \
      || { echo "sample $k harmonize failed"; exit 1; }
    $PY build_expert_anchor_columns.py --examples $FOUR_DB --targets $TARGETS --sample $k \
      --output $EXAMPLES --registry $REGISTRY >> $LOG 2>&1 || { echo "sample $k anchors failed"; exit 1; }
    rm -rf $LEDGER $MIMIC_EX $FOUR_DB
    echo "sample $k anchors done $(date) free=$(df -h . | awk 'NR==2{print $4}')"
  fi

  for attempt in {1..30}; do
    $PY teacher_anchored_joint_jepa_audit.py --event-examples $EXAMPLES --modules $MODULES \
      --split-mode database --train-database mimiciv --test-database eicu,nwicu,zigong \
      --seeds 7,19,31,43,57,71,97 --max-stays 20000 --expert-anchor-registry $REGISTRY \
      --export-predictions $PREDICTIONS --output $AUDIT --resume >> $LOG 2>&1
    rc=$?
    echo "sample $k training attempt $attempt exit $rc $(date)"
    [[ $rc -eq 0 ]] && break
    [[ $rc -eq 137 ]] || { echo "sample $k training failed"; exit 1; }
  done
  [[ $rc -eq 0 ]] || { echo "sample $k training did not finish"; exit 1; }
  RUNS+=($AUDIT:$PREDICTIONS)
done

$PY jepa_gate.py --runs $RUNS --output jepa_gate_samples0to4_20261025.json > jepa_gate_samples0to4_20261025.log 2>&1 \
  || { echo "gate failed"; exit 1; }
echo "JEPA_SAMPLES_DONE $(date)"
