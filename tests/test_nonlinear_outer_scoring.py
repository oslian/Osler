import unittest

from tests.test_nonlinear_probe_scoring import example
from scripts.score_nonlinear_outer_probe import verify_outer_probe_frame


class OuterProbeScoringTests(unittest.TestCase):
    def frames(self):
        frame = example()
        base = frame.copy()
        base['fit_scale'] = 2.
        base['truth'] = frame.truth * 2 + 10
        base['population'] = frame.population * 2 + 10
        native = frame[['patient', 'target', 'fold']].copy()
        native['real10'] = frame.native * 2 + 10
        return frame, base, native

    def test_match_ignores_row_order_not_identity(self):
        frame, base, native = self.frames()
        verify_outer_probe_frame(frame.iloc[::-1], base, native, 0)
        with self.assertRaises(ValueError):
            verify_outer_probe_frame(frame.iloc[:2], base, native, 0)

    def test_changed_predictions_or_component_rejected(self):
        frame, base, native = self.frames()
        frame.loc[0, 'native'] += .1
        with self.assertRaises(ValueError):
            verify_outer_probe_frame(frame, base, native, 0)
        frame, base, native = self.frames()
        frame.loc[0, 'component'] = 'changed'
        with self.assertRaises(ValueError):
            verify_outer_probe_frame(frame, base, native, 0)


if __name__ == '__main__':
    unittest.main()
