# Source-Only Original/Probe Selection

Same strict site split and frozen body cache from the raw/state comparison.
For every33supported targets, candidates are original readout, fit median, and
body StandardScaler+Ridge alpha1/10/100/1000/10000. Fit ridges only on fit rows;
choose lowest tune MAE, ties favor original then median then listed alphas.
No target shortlist or evaluation-specific threshold. Freeze choices before
scoring evaluation; original and all-probe versions remain controls.

Report per-target paired bootstrap2000(seed20260917), all33target retention,
and macro target-equal normalized MAE. Macro score does not override target
harms. Existing tune selected encoder and evaluation was already inspected:
development evidence only, no promotion or new lockbox claim.
