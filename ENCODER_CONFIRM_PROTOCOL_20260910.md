# Supported Decoder Confirmation

Fixed before evaluation. Glucose6h only, availability-safe5-7h labels and
unchanged patient roles. Seven source seeds7,19,31,43,59,71,83. Four matched
arms: own/whole history x tanh/identity direct decoder. All use source-fit
observed-variable support protection, including the bounded controls. Updated
BNP identity cache is used throughout. Same8x16x32 training budget and source
tune selection including epoch zero. JEPA weight1; no external mode selection.

Primary contrasts: identity vs tanh for each input view, whole vs own for each
decoder, all vs persistence. Patient-equal MAE and paired patient bootstrap
95% CI,1000 draws. Seven seeds reuse patients; not independent replications.
Report every contrast. Exploratory CIs are not multiplicity-adjusted promotion.

Persist local prediction arrays in scored-frame order and bind input hashes.
Follow-up descriptive care-unit/time strata must not change fitted weights.
This is point confirmation, not conformal calibration. Source tune outcomes
must not be reused as independent calibration. No runtime change, other-target
qualification, untouched-lockbox claim or causal claim is allowed.
