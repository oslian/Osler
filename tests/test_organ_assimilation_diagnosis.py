from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.diagnose_organ_assimilation import (
    correction_summary, failure_positions, require_verification, support,
)
from scripts.train_organ_assimilation import CONTROLS
from whole_body_jepa.multisource_intervals import sha256


def target(fit, selected, component, patients=30):
    return dict(status='scored', patients=patients, components=3,
        fit_patient_mae=dict(immediate=fit, **{c: 1. for c in CONTROLS if c != 'raw'}),
        normalized_mae={level: dict(immediate=value, **{c: 1. for c in CONTROLS})
                        for level, value in [('patient', selected), ('component', component)]},
        comparisons={c: dict(screen_pass=False) for c in CONTROLS})


class OrganAssimilationDiagnosisTest(unittest.TestCase):
    def test_separates_fit_selection_support_and_missing(self):
        data = dict(a=target(1., .9, .9), b=target(.8, 1.1, .9),
            c=target(.8, .9, 1.1, patients=29), d=dict(status='unsupported'))
        result = failure_positions(data)
        self.assertEqual(result['minimum_support'], 2)
        self.assertEqual(result['unsupported'], ['d'])
        counts = result['contrasts']['unbridged']['counts']
        self.assertEqual(counts['no_fit_mean_gain'], 1)
        self.assertEqual(counts['fit_gain_reversed_on_selection'], 1)
        self.assertEqual(counts['fit_and_selection_mean_gain'], 1)
        self.assertEqual(result['contrasts']['unbridged']['direction_disagreement'], 2)
        self.assertEqual(result['contrasts']['raw']['counts']['fit_comparison_unavailable'], 3)
        data['a']['normalized_mae']['patient']['immediate'] = float('nan')
        with self.assertRaises(ValueError): failure_positions(data)

    def test_support_keeps_repeated_contexts_distinct_from_patients(self):
        frame = pd.DataFrame(dict(patient=['p', 'p'], component=['c', 'c'], variable=['a', 'a'],
            fold=[0, 1], context=[0, 0], zero_gap=[True, False], event_count=[1, 0]))
        result = support(frame)
        self.assertEqual(result['patients'], 1)
        self.assertEqual(result['occurrences'], 2)
        self.assertEqual(result['single_observed_event_zero_gap_occurrences'], 1)
        bad = frame.copy(); bad.loc[0, 'patient'] = None
        with self.assertRaises(ValueError): support(bad)
        bad = frame.copy(); bad['zero_gap'] = [1, 0]
        with self.assertRaises(ValueError): support(bad)
        bad = frame.copy(); bad.loc[0, 'event_count'] = -1
        with self.assertRaises(ValueError): support(bad)

    def test_weighted_correction_summary(self):
        def report(n, s, total, maximum):
            return dict(diagnostics={c: {r: dict(coordinates=n, saturated=s,
                absolute_sum=total, maximum=maximum) for r in ('fit', 'selection')}
                for c in ('immediate', 'delayed', 'body')})
        reports = [report(10, 0, 1., .2), report(30, 2, 9., .4)]
        result = correction_summary(reports)['immediate']['fit']
        self.assertEqual(result['mean_absolute_correction'], .25)
        self.assertEqual(result['saturated_fraction'], .05)
        self.assertEqual(result['maximum_correction'], .4)
        bad = deepcopy(reports); bad[0]['diagnostics']['body']['fit']['saturated'] = 11
        with self.assertRaises(ValueError): correction_summary(bad)

    def test_verification_is_bound_and_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'assessment.json'
            path.write_text('{}')
            evidence = dict(complete=True, bindings={str(path): sha256(path)})
            require_verification(path, evidence)
            with self.assertRaises(ValueError): require_verification(path, dict(evidence, complete=False))
            path.write_text('{"changed": true}')
            with self.assertRaises(ValueError): require_verification(path, evidence)


if __name__ == '__main__':
    unittest.main()
