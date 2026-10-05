# E35: frozen-bridge output direction and objective diagnosis

## Fixed question

E34 trained only a recurrent bridge against real truth. Its backbone is fixed,
yet fitting gains reverse for some unseen-role means. Is final-output correction
harm already present in its infinitesimal direction, or added by crossing truth?
Does the SmoothL1(beta=1) training loss improve where evaluation MAE worsens?

This is NOT E9 teacher imitation, another blend fit, or a new model. It reuses
all E34 saved predictions, all five folds, three channels and three roles, and
the same binding-input zero-bridge `bound_start`. Preserve all 53 targets and
unsupported statuses. No fitting, alpha search, patient/target selection,
new inference, significance screen, promotion or clinical direction claim.

## Exact fixed diagnostic

For truth y, start b, prediction p, let r=y-b and c=p-b. For the final-output
line b+a*c, the MAE right derivative at a=0 is g=-sign(r)*c if r!=0 and |c|
otherwise. Its finite loss change is

    |r-c|-|r| = g + 2*max(sign(r)*c-|r|, 0)  (r != 0)
    |r-c|-|r| = |c|                         (r == 0).

The second term is nonnegative curvature caused by crossing truth. Its
aggregate is not gross harm: benefits and harms can cancel. Separately report
gross harm/benefit and away-versus-crossing harm using the existing E9 identity.
SmoothL1 derivative is -clip(r,-1,1)*c; finite beta=1 loss changes are reported
with the SAME patient/component weights as MAE. Also keep occurrence weighting
to distinguish evaluation reweighting from final fitting objective. This is the
saved endpoint, not the mean of losses recorded along optimizer updates and
does not include AdamW decay as a supervised loss.

Within each target, average repeated folds per patient, then patients or
provenance components equally. Target-equal summaries retain every supported
target; support differs across roles, so compare within role. Original MAEs,
identities, role sizes/disjointness and artifact hashes must reproduce.
Components are provenance groups, not assumed hospitals. No IDs in public JSON.

Convexity implies: if the fixed-weight mean g is strictly positive, no uniform
positive OUTPUT interpolation/extrapolation scale can reduce that empirical
MAE. If g<0 but the full change>0, some sufficiently small positive scale helps
on these hindsight labels, but is NOT a deployable or validated calibration.
Near-zero signs within 1e-12*(1+base_mae+prediction_mae) are inconclusive, not
used to select a model. Reducing a recurrent bridge's weight is NOT equivalent
to interpolating final outputs; no claim about that parameter path is allowed.

## Validation and stopping

Before real data: exact zero residual, wrong direction, beneficial and harmful
crossing, objective disagreement, duplicate/null identities, unsupported
targets, nonfinite inputs, patient and component weighting. Check derivatives
with finite differences and beta=1 loss against PyTorch. Independently compute
every real row and both aggregation levels with scalar math/fsum, and compare
against the vectorized report. Parent E34 verified artifact binds inputs.

Report all channels/roles and all target counts; no posthoc threshold or alpha
tuning. A SmoothL1/MAE conflict is an observed endpoint contrast, not proof that
changing the loss repairs generalization. A nonnegative direction rules out
only a uniform positive scale for these saved predictions and weights.
Data are repeatedly inspected development, not an untouched external lockbox.
Close the diagnostic after exact reproduction; any intervention requires a
separate fixed comparison justified by the result. Runtime stays unchanged.
