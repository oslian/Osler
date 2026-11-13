import unittest
from unittest.mock import Mock

import numpy as np

from scripts.infer_nonlinear_outer_probe import frozen_predictions
from scripts.train_nonlinear_probe import PARAMETERS


class FrozenProbeTests(unittest.TestCase):
    def bundle(self):
        models = {}
        for arm in ('body_probe', 'raw_probe', 'process_summary'):
            model = Mock()
            model.n_iter_ = 100
            model.get_params.return_value = PARAMETERS
            model.predict.return_value = np.zeros(2)
            models[arm] = model
        return dict(target='map', parameters=PARAMETERS, models=models)

    def test_predict_only_and_numeric_process_mask(self):
        bundle = self.bundle()
        raw = np.ones((2, 265))
        result = frozen_predictions(bundle, np.ones((2, 64)), raw, 'map')
        self.assertEqual(set(result), set(bundle['models']))
        for model in bundle['models'].values():
            model.fit.assert_not_called()
        features = bundle['models']['process_summary'].predict.call_args.args[0]
        np.testing.assert_array_equal(features[:, :106], 0.)
        np.testing.assert_array_equal(features[:, 106:], 1.)
        np.testing.assert_array_equal(raw, 1.)

    def test_wrong_identity_or_settings_rejected(self):
        bundle = self.bundle()
        with self.assertRaises(ValueError):
            frozen_predictions(bundle, np.ones((2, 64)), np.ones((2, 265)), 'bun')
        bundle['models']['raw_probe'].n_iter_ = 99
        with self.assertRaises(ValueError):
            frozen_predictions(bundle, np.ones((2, 64)), np.ones((2, 265)), 'map')


if __name__ == '__main__':
    unittest.main()
