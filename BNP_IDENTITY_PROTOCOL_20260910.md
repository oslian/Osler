# BNP Identity Repair Audit

MIMIC d_labitems identifies50963 as NTproBNP, Blood. Current code mislabeled it
bnp. Correct mapping to separate nt_pro_bnp; no BNP<->NT-proBNP conversion or
alias. The53-variable existing model has no NT-proBNP input, so that observation
does not fill BNP. Preserve raw NT-proBNP as a distinct analyte, not missing BNP.

Matched fixed supported candidates: seeds7/19/31, whole_unbounded and
whole_unbounded_balanced. Keep the previous source-support calcium repair in
both arms. Before/after differ ONLY in the verified cached BNP relabeling.
No retraining, external selection, new labels, or interval recalibration.
Report patient-equal MIMIC glucose6h MAE and paired bootstrap95% CI and
number of changed predictions. eICU values and inputs are unchanged by this
MIMIC-only item mapping fix. Negative results do not justify restoring an
incorrect identity. Preserve the old cache/report, generate a new corrected
cache. Repeated MIMIC validation is not a pristine external gate.
