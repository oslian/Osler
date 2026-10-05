# Raw-history comparison before another encoder repair

Same three-way potassium source patients and measured labels. Fit Ridge residual
models on ten existing causal history snapshots, including values/masks/ages.
Compare own30features, full source-supported53-variable history, and full support
with only target numeric values. Source support derived from fitting patients.
No feature unit transform learned from tune/evaluation; existing inherited scaler.
Patient-equal sample weights normalized to fitting row count. Alpha grid fixed
1,10,100,1000,10000; select minimum tune patient-MAE separately by arm. Evaluation
unused until selection. All seven predefined source partitions.

Compare full with own, support, persistence and prior full JEPA on identical
evaluation rows, paired patient bootstrap1000. No additional external read,
capacity-equivalence claim, final lockbox or promotion. A negative linear readout
does not prove absence of nonlinear information. A positive result would justify
testing how encoder preserves raw signal, not automatically replace runtime.

Numerical execution correction: first float32 fit emitted ill-conditioned linear
solve warnings. Before interpreting its scores, rerun identical grid/partitions
using float64 feature matrices. Original output retained as float32 diagnostic;
float64 is authoritative. No feature/model/grid selected from evaluation.
