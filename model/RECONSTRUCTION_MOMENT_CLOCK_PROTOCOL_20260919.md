# Moment-Clock Sensitivity

Hold the original sqrtLR accumulated experiment fixed: seed19, initialization,
expanded source patients,90+90selection,33heads, eightepochs,528exampleblocks,
536AdamWsteps,rate1e-4*sqrt(528),decay.01,clipping and input/label contracts.
Change only betas(.9,.999) to(.9^528,.999^528). This changes both moment clocks
as one predeclared sensitivity arm; individual first/second-moment effects are
not isolated. Do not add the previously ineffective decay-matching change.

Coefficient half-life in exampleunits nowmatches sequentialreference for always
active parameters. Averaging gradients then squaring is not the same as averaging
squares, and biascorrection/stepcount differ. No claim of optimizer equivalence.
This aggressive beta1 is nearlyzero; it is a controlled test, not a recommended
production default. Finite loss/selection checks remain mandatory.

Epoch selection includes0and uses only unchangedsource macroMAE. Freeze before
same reusedcohort evaluation. Primary reference originalaccumulatedsqrtLR;
also report sequentialexpanded,old,population. No targetwiseoracle choice or
gate relaxation. No matchednewprocess/placebo yet; candidateonly,no promotion.
All39statuses retained33trained/6pending. No interval/future/causal claim.
