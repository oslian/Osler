# Source-tune retention trajectory

Seven selected-retention32epoch fits replayed with added per-target source-tune
MAE at every epoch and initialization. All original glucose logs match exactly,
including selected epochs14,20,32,3,31,29,10. Extra evaluation did not change the
training trajectory. No new validation labels fed into optimizer or selection.

Each split has17-20 targets with >=30 fit and tune patients. At every positive
epoch, at least one target has numerically greater MAE than initialization.
Number of numerically worse targets at the least-harmful positive epoch:
7,2,2,1,4,4,3 respectively. At primary-selected epochs:13,13,6,6,11,12,8.
No all-target non-worsening positive epoch exists under this strict numerical
comparison. This is NOT statistical noninferiority or a clinical margin, and
does not establish significant deterioration for every tiny positive difference.

The missing curves are now filled. A switch of checkpoint alone cannot produce
literal all-target non-worsening on this trajectory. It remains possible that a
prespecified tolerance-based tradeoff has utility, but such a policy requires
appropriate margins and held-out testing; do not relax gates after viewing this.
This narrows the selection-only hypothesis, not all possible training objectives.

Initial forecasts are imperfect, so preserving every initial prediction is not
the ultimate goal. A useful joint model needs actual other-target supervision
and explicit tradeoff validation. Earlier auxiliary-loss experiments are not
erased by this result and must be consulted before another replay.

No runtime, weight deployment, calibrator, new lockbox or promotion change.
No claim of causal physiology. Aggregate curves only; raw data stays private.
Artifact: `artifacts/native_retention_curve_20260911.json`.
Runner: `python -m scripts.replay_retention_curve`.
