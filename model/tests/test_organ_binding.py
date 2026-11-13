from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_organ_assignment import cases, pools
from scripts.train_content_retention import replay
from scripts.train_organ_binding import select_arm, fit_arm, predict, assess, validate_roles
from scripts.verify_organ_binding import verify_rows, verify_statistics
from tests.test_organ_assignment import OrganAssignmentTest
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction, mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations


class OrganBindingTest(unittest.TestCase):
    def setUp(self):
        setup = OrganAssignmentTest(); setup.setUp()
        self.model = NativeBodyReconstruction(setup.trajectory).double().eval()
        self.variables = setup.variables
        self.tokens, self.swapped, _ = cases(setup.model, self.variables)

    def example(self):
        tokens = self.tokens[:2]
        times = tokens.new_tensor([0., 1.])
        clean, times = mask_native_target(tokens, times, self.variables, 'glucose')
        with torch.no_grad():
            body, organs, presence = encode_native_organ_observations(self.model.trajectory.jepa, self.variables, clean)
            _, bound = pools(self.model.trajectory.jepa, self.variables, clean)
            frozen = float(self.model(tokens, times, 3., 'glucose')[0])
        return dict(body=body, organs=organs, presence=presence, times=times,
                    anchor=3., target=2, truth=1., frozen=frozen, patient='a', component='x'), bound

    def test_prepool_separates_and_postpool_does_not(self):
        model = self.model.trajectory.jepa
        for x, y in zip(self.tokens, self.swapped):
            native, bound = pools(model, self.variables, x[None])
            changed, bound_changed = pools(model, self.variables, y[None])
            torch.testing.assert_close(torch.nn.functional.silu(native), torch.nn.functional.silu(changed), atol=1e-10, rtol=0)
            self.assertGreater(float((bound-bound_changed).detach().abs().max()), 1e-10)

    def test_arms_preserve_body_times_and_native_parity(self):
        e, bound = self.example()
        native_point = float(self.model.readout(replay(self.model.trajectory.layer, e))[0, 2].detach())
        self.assertEqual(native_point, e['frozen'])
        for arm in ('native', 'binding', 'postpool'):
            selected = select_arm(e, bound, arm)
            self.assertIs(selected['body'], e['body'])
            self.assertIs(selected['times'], e['times'])
            self.assertEqual(selected['truth'], e['truth'])
        with self.assertRaises(ValueError): select_arm(e, bound, 'unknown')
        with self.assertRaises(ValueError): select_arm(e, bound[:1], 'binding')
        with self.assertRaises(ValueError): select_arm(e, bound * float('nan'), 'native')

    def test_train_equal_capacity_frozen_encoder_and_saved_replay(self):
        e, bound = self.example()
        before = deepcopy(self.model.state_dict())
        count = sum(p.numel() for p in self.model.parameters())
        for arm in ('native', 'binding', 'postpool'):
            trained, log = fit_arm(self.model, [e], [bound], arm)
            self.assertEqual(len(log), 2)
            self.assertEqual(sum(p.numel() for p in trained.parameters()), count)
            self.assertTrue(any(not torch.equal(v, trained.state_dict()[k]) for k, v in before.items()))
            for k, v in before.items():
                torch.testing.assert_close(self.model.state_dict()[k], v, atol=0, rtol=0)
                if k.startswith('trajectory.jepa.'):
                    torch.testing.assert_close(trained.state_dict()[k], v, atol=0, rtol=0)
            fresh = deepcopy(self.model); fresh.load_state_dict(trained.state_dict(), strict=True)
            np.testing.assert_array_equal(predict(trained, [e], [bound], arm), predict(fresh, [e], [bound], arm))
            frame = pd.DataFrame(dict(patient=['a'], component=['x'], variable=['glucose'], truth=[1.],
                                      **{arm: predict(trained, [e], [bound], arm)}))
            self.assertEqual(verify_rows(frame, [e], fresh, [bound], arm, self.variables), 0.)
            frame.loc[0, arm] += 1.
            with self.assertRaises(AssertionError): verify_rows(frame, [e], fresh, [bound], arm, self.variables)
        with self.assertRaises(ValueError): fit_arm(self.model, [e], [], 'native')

    def test_roles_and_scoring_fail_closed(self):
        e, _ = self.example()
        examples = dict(fit=[e], selection=[dict(e, patient='b')])
        validate_roles(examples, dict(a='fit', b='selection'))
        with self.assertRaises(ValueError): validate_roles(examples, dict(a='selection', b='fit'))
        with self.assertRaises(ValueError): validate_roles(dict(fit=[e], selection=[e]), dict(a='fit'))
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], component=['x', 'x', 'y'],
                                  variable=['glucose']*3, truth=[0.]*3))
        for arm in ('native', 'binding', 'postpool', 'frozen', 'raw'):
            frame[arm] = [1., 3., 6.]
        result = assess(frame, 1, 20)
        self.assertEqual(result['normalized_mae']['patient']['binding'], 4.)
        self.assertFalse(any(v['screen_pass'] for v in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 1, 20)
        frame.loc[0, 'binding'] = float('nan')
        with self.assertRaises(ValueError): assess(frame, 1, 20)

    def test_verification_allows_only_mean_roundoff(self):
        result = dict(status='scored', patients=30, normalized_mae=dict(patient=dict(binding=.5)),
                      comparisons=dict(native=dict(screen_pass=False, paired=dict(p_less=.01, ci95=[-.1, .1]))))
        actual = deepcopy(result)
        actual['normalized_mae']['patient']['binding'] = np.nextafter(.5, 1.).item()
        self.assertGreater(verify_statistics(actual, result), 0.)
        actual['normalized_mae']['patient']['binding'] += 1e-8
        with self.assertRaises(ValueError): verify_statistics(actual, result)
        for key, value in (('screen_pass', True), ('paired', dict(p_less=.010000000000000002, ci95=[-.1, .1]))):
            actual = deepcopy(result); actual['comparisons']['native'][key] = value
            with self.assertRaises(ValueError): verify_statistics(actual, result)
        actual = deepcopy(result); actual['patients'] += 1
        with self.assertRaises(ValueError): verify_statistics(actual, result)
        actual = deepcopy(result); actual.pop('normalized_mae')
        with self.assertRaises(ValueError): verify_statistics(actual, result)
        actual = deepcopy(result); actual['normalized_mae']['patient']['binding'] = float('nan')
        with self.assertRaises(ValueError): verify_statistics(actual, result)
        self.assertEqual(verify_statistics(dict(status='unsupported'), dict(status='unsupported')), 0.)


if __name__ == '__main__':
    unittest.main()
