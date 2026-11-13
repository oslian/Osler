# Source-Only Observation Support Decision Probe

Freeze before execution. This follows the completed nonlinear source error
decomposition, not a new encoder or new clinical capability. All47 targets and
five existing selection folds remain; no tail/target/group deletion or outer
evaluation. The earlier aggregate support correlation and source conditional
residual correction studies remain negative evidence, not untried methods.

## Question

Can pre-query target/sibling-masked observation identity and actual measurement
age predict whether the fixed nonlinear body readout or native readout has
lower expected absolute error on a different source component?

## Fixed Comparison

Within EACH original fold, use only selection rows. Leave one whole component
out for the decision probe; never mix selector training rows across original
folds. Existing encoder/readout fitting components are disjoint from the entire
selection role. All preprocessing, weights and selector coefficients use only
the remaining selection components. The decision target is
`abs(body-truth)-abs(native-truth)`, not a single-example win indicator.

Probe arms:

1. Constant training-weighted mean expected difference.
2. Aggregate support: observed count, count aged<=1h, count aged<=6h, mean
   observed age. Missing values never enter age averages.
3. Variable-specific support:53 explicit masks plus53 log1p measurement ages;
   absent ages have zero feature value with a separate mask. No numeric lab
   values, truth, hospital names, target history or sibling history enter.
4. Same106-feature/capacity probe with whole feature vectors cyclically
   reassigned within fitting components. Preserve component marginals, derange
   all groups with>=2patients; report singleton exceptions. Query features
   stay real in both arms. One fixed null seed, not a permutation p-test.

Use the repository's bounded component weights [.5,2], weighted StandardScaler
and Ridge(alpha=100, solver=cholesky). Squared loss estimates expected *absolute
error difference*, not the physiologic value; no win-frequency optimization.
No parameter search or learned decision threshold. Choose body if estimated
difference<0, native otherwise (ties use native). Always-native, always-body
and the existing population reference remain visible. If fewer than two
training components or30training patients exist, retain explicit unsupported
status rather than dropping that query from the overall report.

Read only the verified observation-mask/measurement-age snapshot columns and
source prediction rows. Snapshot can contain other roles, but they are never
used to fit preprocessing, the probe or cutoffs. Join exact patient+target
identity; enforce target/sibling masking, prediction hashes and roles.

## Evaluation

Average paired losses over each patient's eligible original folds, then report
patient-equal and component-equal means. Overlapping models/queries are not
independent replications. Report chosen-body fraction, regret relative to each
constant arm, and held-component expected-difference MSE against constant.
Variable-specific versus native/body/constant/aggregate/shuffled are the five
fixed contrasts. Use existing paired summary bootstrap1000 for descriptive95%
CIs and its one-sided tests at0.05/(5*47) for a labeled diagnostic screen.
All components stay, including small ones. Incomplete probe support cannot pass.

This is development-only meta-crossvalidation, not a fresh external lockbox,
runtime promotion or calibrated interval. MIMIC/NWICU are not opened for this
experiment. No change to original weights, safety policies or accepted models.
Success would justify a subsequently frozen selector tested independently;
failure only rejects this fixed observation-support decision rule, not every
possible individual-state representation.
