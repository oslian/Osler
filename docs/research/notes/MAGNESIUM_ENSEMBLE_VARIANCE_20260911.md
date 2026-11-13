# Fixed ensemble variance diagnostic

All seven seeds share exactly256hospital rows/103patients. No source-evaluation
averaging, which could mix training roles. Parent hashes verified. Float64
pointwise mean-member-MSE = ensemble-MSE + prediction-variance identity passes.

Own prediction variance .00099354; whole .00278031 (~2.80x).
Mean member MSE own .0524878,whole .0528717; ensemble MSE .0514942,.0500914.
Ensemble MAE own .181701,whole .182428. Paired MAE difference+.000727,
CI[-.011928,.013669]. Whole ensemble beats persistence, not own ensemble.

Across-fit sensitivity contributes measurable squared-error variation,
but averaging does not solve the MAE incremental gate. These seven members
vary partitions AND sampling seeds. Follow-up verifies all initial weights are
identical (`docs/research/notes/MAGNESIUM_INITIAL_STATE_20260911.md`); not initialization variance. MSE point
improvement is descriptive, not a substitute MAE gate or significance claim.
Not a noise-floor estimate or proof of irreducible biological uncertainty.
No runtime/calibration/promotion. Artifact magnesium_ensemble_variance_20260911.json.
