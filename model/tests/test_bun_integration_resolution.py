import unittest
from types import SimpleNamespace
from scripts.audit_bun_integration_resolution import resolution


class ResolutionTests(unittest.TestCase):
    def test_restores_after_error(self):
        layer=SimpleNamespace(max_step_hours=1.)
        with self.assertRaises(RuntimeError):
            with resolution(layer,.25):
                self.assertEqual(layer.max_step_hours,.25)
                raise RuntimeError('expected')
        self.assertEqual(layer.max_step_hours,1.)

    def test_invalid(self):
        for step in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                with resolution(SimpleNamespace(max_step_hours=1.),step):pass
