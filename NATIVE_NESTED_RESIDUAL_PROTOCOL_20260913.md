# Nested Patient-Excluded Residual

Reuse source64fit32tune and64+64heldout pilot queries exactly. Baseline is the
new nested own-history model excluding residual patients from fitting/selection.
Full-input numeric/support transforms recomputed from baseline-fit histories
only (MAP unchanged). Fullstate versus anchor-only same original state-input
control,8epochs, sameinit, optimizer, value+EMAJEPA losses. Source tune selection
only. Save transforms with candidate. Pretraining provenance caveat retained.
No runtime/promotion; comparison to old experiment is not isolated exposure
causation because baseline quality/selection and transforms also change.
