# Native history recency gradient probe

Frozen seven clock-corrected real checkpoints,32distinct source-fit patients
per seed, one random fit query per patient. No evaluation/hospital outcomes or
model selection. Raw gradients of normalized potassium output with respect to
normalized observed history values; aggregate absolute magnitude by active-event
distance from query. Report: `artifacts/native_memory_gradient_20260911.json`.

Fraction of absolute observed-value gradient from positions at least10active
updates back: potassium48.9%-75.9%, insulin17.9%-43.8% across seven probes.
Every probe has nonzero older-event gradients. This rejects a literal disconnected
old-history path or total recent-only erasure on these sampled inputs. It does
not establish useful long-term memory or comparable per-event importance:
older bins contain more observations; channels have different source scaling.

Insulin gradients remain smaller in total normalized-coordinate magnitude than
potassium in these probes. That is not a biological sensitivity comparison or
evidence to amplify insulin weights. Local derivatives do not measure forecast
utility, causal treatment effects, or what an alternative trained encoder could
retain. No saturation or vanishing-gradient root cause is established here.

Together with native-sequence and clock refits, this removes another simple
explanation: the model can depend on earlier events but has not shown stable
incremental use of treatment numbers. Next diagnostics must distinguish signal
learnability/conditioning from this demonstrated nonzero path; do not blindly
increase recurrence length. No training, runtime, interval or promotion change.
