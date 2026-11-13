# Training-matched non-target support control

Use the completed three-way potassium protocol unchanged: same785fit/262tune/
262evaluation patients, seven seeds,64epochs, model capacity, warm start,
patient sampling, value and JEPA loss. Only remove non-potassium numerical
content from the whole arm at BOTH training and evaluation; retain masks,
ages,presence and actual potassium history. Normalized non-target values0 are
an explicit diagnostic ablation, not measured/imputed values for serving.

Rerun own control and require exact logs and saved predictions to reproduce
the prior three-way run. Compare newly trained support-only arm against prior
full-numeric whole, own and persistence on unchanged evaluation patients;
checkpoints still selected exclusively on tune. No setting sweep or external
read. Primary support-only vs full-numeric comparison, all seven seeds report.

This tests whether numerical cross-context adds value beyond support to this
trained model. Support encodes both clinical observation processes and possible
physiological correlates; it is NOT a pure hospital-workflow variable. A null
result cannot prove all numeric information absent. Reused source and warm
start forbid pristine-lockbox or full promotion claims. No runtime changes.
