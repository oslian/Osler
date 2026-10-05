# Availability-Safe JEPA Input Trial

Freeze before evaluation. Same queries/patients/labels as the availability
glucose audit; all unscored queries retained. Native ID, measurement time and
result availability are retained. Source canonical maps plus explicit units
and bounds determine accepted labs. Unverified vital/treatment channels stay
missing: full schema does not mean every channel is observed. No imputation.

Construct exact 48/24/12/8/6/4/3/2/1/0h historical snapshots using only results
available at each historical timestamp. Earlier missing history stays missing;
no repeated earliest snapshot. Latest measured known result is current, not
latest arriving old sample. Revisions become visible only when recorded.

Same frozen JEPA checkpoint, three input arms: glucose-only truthful masks/age;
all accepted labs truthful masks/age; all labs with existing all-one/zero-age
encoding (same safe numerical values). Compare frozen shared-head forecasts,
then refit equal-width311 ridge readouts, source-only alpha1/10/100 over seven
existing patient fit/tune seeds. Baseline persistence on identical rows.
No other architecture, no global online learning, no new interval claim.
eICU determines fits; MIMIC is already-opened validation, not a final lockbox.
Changing masks under frozen weights tests distribution compatibility, not
end-to-end observation-aware training. Readout gain is not cross-organ causal
evidence; full placebo/care-unit/time/conformal gates remain required.

Input repair v2: the first unit inventory revealed equivalent spellings
K/mcL, mcg/dL, mcU/ml and mOsm/kg H2O absent from the whitelist. Accept these
without changing numerical values. No model settings or gates change. Retain
v1 artifact; this is a documented post-inventory extraction repair, not a fresh
lockbox. Ionized calcium mg/dL and unitless INR remain explicitly unsupported
in this bounded adapter rather than silently guessing a conversion.
