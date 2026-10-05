# Source Patient Learning Curve

Frozen settings before fitting: use history-mask candidate contract,
original source support policy/scales,32epochs x16updates x32patient-balanced
samples, AdamW and source-tune selection including epoch0. Compare all four
own/whole bounded/identity arms, seven original seeds. Retain original136
tune patients per seed and all original hospital/MIMIC evaluation rows.

Existing250-fit history-mask checkpoints are the starting reference. Add750,
then2000 new independent source patients for totals1000 and2250. Select the
new eligible patients by a fixed learning-curve-v1 SHA256 order independent
of values and external results, nested between sizes. Their full unscored
histories are retained; only available5-7h pairs contribute supervised loss.
Do not repartition original tune patients after adding the new pool.

The first run is1000fit patients. It is a fixed-compute learning curve:
larger N receives fewer updates per person, not extra optimization steps.
Failure therefore cannot establish a converged data-size ceiling. The added
pool is broader than the original bounded creatinine cohort, so composition
and patient count are not perfectly isolated. Report both limitations.

Compare patient-paired MAE against the same250-patient arm and whole versus
own, persistence and existing descriptive strata. Save additional input
hashes and exact fit/tune counts. No settings selected on MIMIC/NWICU;
both are inspected validation, not pristine lockboxes. No intervals or
runtime promotion from this diagnostic, no causal claim.
