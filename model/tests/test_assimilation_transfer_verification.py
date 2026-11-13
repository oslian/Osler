from copy import deepcopy
import unittest

import pandas as pd

from scripts.probe_assimilation_transfer import ARMS, summarize
from scripts.verify_assimilation_transfer import verify_summary


class TransferVerificationTest(unittest.TestCase):
    def fixture(self):
        data = pd.DataFrame([dict(patient=p, component='c', variable='v', context=0,
            fold=f, truth=0., epoch10_exposed=True,
            **{a: (value if a == 'immediate' else 1.) for a in ARMS})
            for p, f, value in [('a', 0, 0.), ('a', 1, 0.), ('b', 0, 3.)]])
        old = dict(status='scored', fit_patient_mae=dict(immediate=0., unbridged=1.),
                   normalized_mae=dict(patient=dict(immediate=2., unbridged=1.)))
        parent = dict(results=dict(v=old, w=deepcopy(old)))
        result = dict(results=summarize(data, ['v', 'w']), summary=dict(
            occurrences=3, patients=2, epoch10_exposed_occurrences=3, scored_targets=1,
            prior_reversals=dict(v='same_source_also_no_gain', w='unsupported'),
            counts=dict(unsupported=1, same_source_also_no_gain=1, same_source_mean_gain=0)))
        return data, result, parent

    def test_patient_weighted_reversal_and_missing_support(self):
        verify_summary(*self.fixture())

    def test_tampered_count_classification_and_support_rejected(self):
        data, result, parent = self.fixture()
        for section, key, value in [('summary', 'patients', 3),
                ('summary', 'prior_reversals', {'v': 'same_source_mean_gain', 'w': 'unsupported'})]:
            bad = deepcopy(result); bad[section][key] = value
            with self.assertRaises(ValueError): verify_summary(data, bad, parent)
        bad = deepcopy(result); bad['results']['v']['components'] = 2
        with self.assertRaises(ValueError): verify_summary(data, bad, parent)

    def test_duplicate_and_nonfinite_rejected(self):
        data, result, parent = self.fixture()
        with self.assertRaises(ValueError):
            verify_summary(pd.concat([data, data.iloc[:1]]), result, parent)
        data.loc[0, 'immediate'] = float('nan')
        with self.assertRaises(ValueError): verify_summary(data, result, parent)


if __name__ == '__main__':
    unittest.main()
