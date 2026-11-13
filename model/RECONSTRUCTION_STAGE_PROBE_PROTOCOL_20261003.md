# Event Encoding Versus Persistent Aggregation

Frozen epoch10, current masked reconstruction only. This is diagnostic,
not a new serving architecture, full-cohort efficacy trial or promotion.
Prior E16 used the final trained readout on intermediate states. Here each
representation gets its own fit-only probe, trained on exactly the same pairs.

All47 targets and5 existing source folds. For each role/target choose at most
512 fitting and256 selection patients by SHA256 of a fixed salt and patient ID;
the selection rule never sees outcome, error, history length or component.
All available patients below those limits remain. This bounds extraction cost,
not an opportunity to choose successful targets. Roles remain disjoint.

Same target and sibling mask throughout. Fixed trained encoder event proposals:
mean of all64-dimensional proposals, last64-dimensional proposal, final64 body.
Append the SAME159 raw process features (presence, lag, log count) to each.
Thus these three have equal223-dimensional probe inputs. Independent controls:
raw265 and process159. Different width for raw/process is explicit, not an
exact-capacity equivalence claim. No ground-truth target or own history enters.

Two prespecified readout families for every source: existing100-stage HGB
absolute-error settings; fit-only weighted StandardScaler + Ridge(alpha100,
solver cholesky). Existing bounded component weights, no parameter selection.
No calibration, outer or future labels. Hash-bind model, cache, roles, scalers,
source code and protocol. Check sampled raw summary and truth against original
cache for every example; reproduce original forward on first sample per role/
target. Validate cache payload digests. Save private identities/features only
under ignored data, aggregate results under artifacts.

After all235 target-fold fits finish: average normalized absolute loss over a
patient's repeated folds, then patient and component means. Report eight
contrasts (mean-body, mean-process, mean-raw, last-body for each readout family),
paired bootstrap5000, one-sided paired-unit p<.05/(47*8), CI upper<0, >=30
patients and >=3components. These are reused-development descriptive screens,
not untouched confirmation. Report support and all failures, no partial-fold
promotion. Sparse/poorly represented components limit interpretation.

If mean beats body for both families, this supports a usefulness loss across
aggregation for these probes, NOT irreversible information loss. One-family
gain indicates readout dependence. Mean failure cannot establish event encoder
failure because averaging itself discards order and detailed timing. Controls
retain identical process covariates; shortcut and coordinate confounds are not
entirely eliminated. No automatic new head search after the result.

One fixed run. Numerical, role, digest or original-forward mismatch stops the
run. Resume only artifacts matching exact contracts. No runtime changes.
