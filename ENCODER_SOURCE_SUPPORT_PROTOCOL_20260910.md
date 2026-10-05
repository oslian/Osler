# Source-Only Support Intervention

Fixed existing whole_unbounded and whole_unbounded_balanced weights for seeds
7/19/31. Same availability-safe glucose6h inputs and actual5-7h labels.
No weight update or target-database selection. Use the identical source-fit
patient partition that trained each candidate. Count distinct fit patients
whose legal lag snapshots contain each variable. Predeclare two rules:
keep variables seen by at least1 patient; keep those seen by at least20.
Always require glucose meets20. Apply to all historical/current input slots,
values/masks/ages/organ presence consistently; removed values become unknown.
Do not retain removed-variable age as a process shortcut.

Compare full versus each restriction with identical weights, patient-equal MAE
and paired patient-bootstrap CI on eICU hospital and reused MIMIC validation.
Report all seeds/modes, including harm. This is an input intervention, not a
trained supported-only model, not a clinical causal effect or full gate.
Inference masking is distribution shift itself; improvement would implicate
unsupported-input sensitivity, not demonstrate cross-organ understanding.
Also report negative and extreme candidate outputs after removing tanh.
No release or calibrator changes; no patient-row output in version control.
