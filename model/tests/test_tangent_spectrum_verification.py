import unittest

import numpy as np
import pandas as pd

from scripts import audit_tangent_spectrum as audit
from scripts.verify_tangent_spectrum import verify_saved_rows, scalar_variants, macro_summary
from scripts.verify_frozen_bridge import compare


class TangentSpectrumVerificationTest(unittest.TestCase):
    def frames(self):
        original = pd.DataFrame([dict(patient=f'p{i//2}', component=f'c{i//4}',
            fold=i%2, context=0, variable='map', truth=float(i), local_linear=.5,
            tangent_base=0., tangent_slope=1.,
            **{name:.1 for name in (audit.parent.PRIMARY, *audit.parent.CONTROLS)})
            for i in range(6)])
        frame = original.assign(strong_slope=.75, weak_slope=.25, null_slope=0., roundoff_slope=0.,
            strong_actual=.2, weak_actual=.3, strong_linear=.375, weak_linear=.125)
        return original, frame

    def test_saved_decomposition_and_controls(self):
        original, frame = self.frames()
        self.assertEqual(verify_saved_rows(frame, original, .5), 0.)
        for key in ('truth', 'bound_start', 'strong_slope', 'weak_linear', 'roundoff_slope'):
            changed = frame.copy()
            changed.loc[0, key] += .1
            with self.assertRaises((AssertionError, ValueError)):
                verify_saved_rows(changed, original, .5)

    def test_nonfinite_rejected(self):
        original, frame = self.frames()
        frame.loc[0, 'strong_actual'] = np.nan
        with self.assertRaises(ValueError):
            verify_saved_rows(frame, original, .5)

    def test_scalar_aggregation_and_inventory(self):
        _, frame = self.frames()
        expected, _ = audit.descriptions(frame, ['map', 'bun'])
        result = scalar_variants(frame, ['map', 'bun'])
        self.assertLess(compare(result, expected), 1e-12)
        self.assertEqual(result['weak_actual']['bun']['status'], 'unsupported')
        summary = macro_summary({'selection':result})['selection']['strong_actual']
        self.assertEqual(summary['scored_targets'], 1)
        self.assertEqual(summary['mae']['patient'], result['strong_actual']['map']['mae']['patient']['tangent_step'])


if __name__ == '__main__':
    unittest.main()
