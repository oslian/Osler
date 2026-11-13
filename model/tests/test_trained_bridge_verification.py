from copy import deepcopy
import unittest

from scripts.audit_trained_bridge import summarize
from scripts.verify_trained_bridge import verify_results
from tests import test_trained_bridge


class BridgeSummaryTest(unittest.TestCase):
    def test_independent_means_and_tampering(self):
        f = test_trained_bridge.TrainedBridgeTest().frame()
        saved = summarize(f, ['map', 'bun'])
        self.assertLess(verify_results(f, saved), 1e-12)
        for field in ('patients', 'occurrences', 'components'):
            altered = deepcopy(saved); altered['map'][field] += 1
            with self.assertRaises(ValueError): verify_results(f, altered)
        altered = deepcopy(saved); altered['map']['contrasts']['body']['patient']['bridge'] += .1
        with self.assertRaises(ValueError): verify_results(f, altered)
        altered = deepcopy(saved); altered['bun'] = dict(status='scored')
        with self.assertRaises(ValueError): verify_results(f, altered)
        altered = deepcopy(saved); altered['map']['normalized_mae']['patient']['frozen'] = float('nan')
        with self.assertRaises(ValueError): verify_results(f, altered)


if __name__ == '__main__': unittest.main()
