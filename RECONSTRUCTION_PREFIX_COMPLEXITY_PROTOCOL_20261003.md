# Frozen Readout Prefix Complexity: Source-Only Intervention

Fixed before execution: all47 targets, all5 original folds, same frozen epoch10
encoder and saved HGB models. Evaluate stages [1,5,10,25,50,100] with sklearn's
public staged_predict, not truncation of private tree internals or refitting.
No added dependency, new encoder, treatment features or outer-label inspection.

Within each original source-selection fold and target, leave one entire source
component out. On the remaining components choose the stage minimizing equal-
component MAE; ties choose the earliest stage. Apply that single chosen stage
to all people in the held component. Choose body/raw/process/shuffled heads
independently under the same rule and six-stage search budget. No true value,
truth band or individual error from that held component may select its stage.
Require at least30 selection-training people and2 other components. Unsupported
groups remain explicitly unsupported, with stage100 only for bookkeeping; they
cannot qualify for the complete diagnostic screen. Do not drop small groups.

Reuse the exact existing selection identities and labels. Last-stage predictions
must reproduce all four original head predictions. Fit-stage body curves are
diagnostic only: same complete original fitting examples, all six stages, with
original native/population references. Report retrospective central/intermediate/
tail errors to locate troponin's conditional fitting tradeoff; never use these
bands as routing inputs. Verify the100-stage fit means against the completed
fit-gap audit. No source query becomes encoder/head fitting data.

Primary paired comparisons: component-selected body prefix against native,
fixed body100, population, selected process prefix and selected shuffled prefix.
Report patient-equal losses averaging each person's fold losses, component-equal
losses, paired bootstrap95%CIs, and the existing one-sided t screening statistic
with .05/(47*5) correction at BOTH levels, sufficient30people/3components and
complete selector support. Require both CI upper bounds below0 as well. These
are conditional diagnostic screens, NOT runtime promotion or fresh external
validation: source outcomes have been reused, fold models overlap, and bootstrap
does not include repeated model/stage-selection fitting uncertainty.

Raw prefix is a descriptive information control, not another promotion test.
All47 target means and all groups remain visible, including nonbenefits. Report
how often stages are shortened, so failure to use a shorter model is distinct
from shorter models failing to improve. Fixed-stage curves on queries are
retrospective diagnostics, not an oracle policy or external selection rule.

No runtime/model/interval change. A positive result must still undergo the
original independent model, interval and external gates. Reducing complexity
may remove real subgroup gains; no universal benefit is assumed.
