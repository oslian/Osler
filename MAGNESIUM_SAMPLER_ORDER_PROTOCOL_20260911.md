# Fixed-partition sampling diagnosis

Use partition7 for all fits, including fit/tune/evaluation/hospital and event
statistics. Own/whole inputs x seven sampling seeds. Identical initialization
has been verified.32epochs, original losses and source-tune selection unchanged.
Seed7 should reproduce original matched own/whole logs and predictions.
Other seeds vary only minibatch draws/order and resulting selected epoch.
This is not seven patient split validation. Do not select a best seed on hospital.
No runtime/calibration/promotion or untouched external cohort.
