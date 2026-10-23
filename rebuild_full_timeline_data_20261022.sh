#!/bin/zsh
# Rebuild every transition dataset with the corrected contract (all anchors kept; label offsets recorded).
# Same arguments as the v4 / 20261015 builds; only the contract changed.  Serial to avoid memory kills.
cd /Users/chunyouchang/Desktop/llm_project/medical_jepa
PY=.venv/bin/python
# Strictly serial (the machine has 17 GB; one MIMIC horizon peaks near 9 GB): never run other large jobs alongside.
# Finished outputs are skipped, so the script can be restarted after an interruption.
for h in 1 3 6 12 24 48; do
  [ -f mimiciv_observation_transitions_${h}h_full_v31_treatment_unit_safe_v5.parquet ] && continue
  $PY mimiciv_observation_transition_extract.py --horizon-hours $h --anchor-step-hours 6 --include-treatment-context \
    --output mimiciv_observation_transitions_${h}h_full_v31_treatment_unit_safe_v5.parquet \
    --report mimiciv_observation_transitions_${h}h_full_v31_treatment_unit_safe_v5_report.json > rebuild_mimic_${h}h_v5.log 2>&1
  echo "mimic ${h}h exit $? $(date)"
done
for h in 1 3 6 12 24 48; do
  [ -f nwicu_observation_transitions_${h}h_v5.parquet ] || \
  $PY nwicu_observation_transition_extract.py --nwicu-dir /Users/chunyouchang/Downloads/northwestern-icu-nwicu-database-0.1.0/data --horizon-hours $h \
    --output nwicu_observation_transitions_${h}h_v5.parquet --report nwicu_observation_transitions_${h}h_v5_report.json > rebuild_nwicu_${h}h_v5.log 2>&1
  echo "nwicu ${h}h exit $? $(date)"
  [ -f zigong_observation_transitions_${h}h_v2.parquet ] || \
  $PY zigong_observation_transition_extract.py --root "/Users/chunyouchang/Downloads/critical-care-database-comprising-patients-with-infection-at-zigong-fourth-peoples-hospital-1.1/DataTables" \
    --horizon-hours $h --output zigong_observation_transitions_${h}h_v2.parquet --report zigong_observation_transitions_${h}h_v2.json > rebuild_zigong_${h}h_v2.log 2>&1
  echo "zigong ${h}h exit $? $(date)"
  [ -f emory_challenge2019_transitions_${h}h_v2.parquet ] || \
  $PY emory_challenge2019_transition_extract.py --root /Users/chunyouchang/mimic/physionet.org/files/challenge-2019/1.0.0/training/training_setB \
    --horizon-hours $h --output emory_challenge2019_transitions_${h}h_v2.parquet --report emory_challenge2019_transitions_${h}h_v2.json > rebuild_emory_${h}h_v2.log 2>&1
  echo "emory ${h}h exit $? $(date)"
done
for f in eicu_harmonized_20261015/eicu_*_transitions_*h.parquet; do
  base=$(basename $f .parquet); module=${base#eicu_}; module=${module%_transitions_*}; h=${base##*_}; h=${h%h}
  [ -f eicu_harmonized_20261022/${base}.parquet ] && continue
  out=(--horizon-hours $h --output eicu_harmonized_20261022/${base}.parquet --report eicu_harmonized_20261022/${base}_report.json)
  case $module in
    aki|sepsis|respiratory)   # dedicated disease extractors, all stays (as in the 20261015 build)
      $PY eicu_${module}_transition_extract.py --data-root /Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0 \
        $out > eicu_harmonized_20261022/${module}_${h}h.log 2>&1;;
    *)
      $PY eicu_body_system_transition_extract.py --data-root /Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0 \
        --disease $module --max-stays 2500 $out > eicu_harmonized_20261022/${module}_${h}h.log 2>&1;;
  esac
  echo "eicu $module ${h}h exit $? $(date)"
done
echo REBUILD_DONE
