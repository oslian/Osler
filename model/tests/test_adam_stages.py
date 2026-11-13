from copy import deepcopy
import unittest

import numpy as np
import torch

from scripts.diagnose_adam_stages import adam_stages, gradient_metrics, trace_fit, summarize
from scripts.train_pair_gradient import fit_pair
from tests.test_parameter_path import ParameterPathTest


class AdamStagesTest(unittest.TestCase):
    def stages(self, before, proposal, moment, variance, decay=0.):
        state = dict(exp_avg=torch.tensor(moment, dtype=torch.double),
                     exp_avg_sq=torch.tensor(variance, dtype=torch.double), step=torch.tensor(1.))
        group = dict(lr=1., betas=(0.,0.), eps=1e-8, weight_decay=decay)
        before = torch.tensor(before, dtype=torch.double)
        return adam_stages(before, before, torch.tensor(proposal), state, group)

    def test_momentum_changes_protected_direction(self):
        s = self.stages([0.,0.], [1.,1.], [-1.,-1.], [1.,1.])
        r = gradient_metrics(torch.ones(2), s)
        self.assertLess(r['proposal_dot'], 0)
        self.assertGreater(r['momentum_dot'], 0)

    def test_positive_diagonal_can_harm_individual_gradient(self):
        s = self.stages([0.,0.], [2.,1.], [2.,1.], [100.,.01])
        r = gradient_metrics(torch.tensor([1.,-1.]), s)
        self.assertLess(r['momentum_dot'], 0)
        self.assertGreater(r['preconditioned_dot'], 0)

    def test_decay_can_reverse_direction_and_zero_is_safe(self):
        s = self.stages([-100.,0.], [1.,0.], [1.,0.], [1.,1.], .1)
        r = gradient_metrics(torch.tensor([1.,0.]), s)
        self.assertLess(r['preconditioned_dot'], 0)
        self.assertGreater(r['with_decay_dot'], 0)
        zero = gradient_metrics(torch.zeros(2), s)
        self.assertTrue(all(zero[k] == 0 for k in zero if k.endswith('_dot')))

    def test_real_adam_step_formula_and_no_mutation(self):
        p = torch.nn.Parameter(torch.tensor([.2,-.3], dtype=torch.float64))
        opt = torch.optim.AdamW([p], lr=.001, weight_decay=.03)
        for values in ([1.,-2.], [-1.,1.], [0.,0.]):
            p.grad = torch.tensor(values, dtype=p.dtype)
            b = p.detach().clone(); opt.step()
            saved = deepcopy(opt.state[p])
            stages = adam_stages(b, p, p.grad, opt.state[p], opt.param_groups[0])
            torch.testing.assert_close(stages['with_decay'], stages['actual'], atol=1e-16, rtol=0)
            for k, v in saved.items():
                torch.testing.assert_close(v, opt.state[p][k], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            adam_stages(b, p, p.grad, opt.state[p], dict(opt.param_groups[0], amsgrad=True))
        with self.assertRaises(ValueError):
            adam_stages(b, p, p.grad, dict(opt.state[p], exp_avg_sq=-torch.ones_like(p)), opt.param_groups[0])
        with self.assertRaises(ValueError):
            gradient_metrics(torch.tensor([np.nan,np.nan]), stages)

    def test_exact_parent_replay_both_arms_and_finite_losses(self):
        c, e = ParameterPathTest().candidate(); original = deepcopy(c.model).float()
        e = {k:v.float() if k in ('body','organs','presence') else v for k,v in e.items()}
        examples = [dict(e,truth=t,target=i%2) for i,t in enumerate([2.,-1.,1.,-2.])]
        bound = [e['organs']]*len(examples)
        for arm in ('mean','pcgrad'):
            a, expected = fit_pair(original, examples, bound, arm)
            b, actual, rows, rounding = trace_fit(original, examples, bound, arm)
            self.assertEqual(expected, actual)
            for k, v in a.state_dict().items():
                torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
            summary = summarize(rows)
            self.assertEqual(summary['examples'], 8)
            self.assertEqual(summary['unscaled_proposal_harm'], sum(x['proposed_harmful_dots'] for x in actual))
            self.assertEqual(summary['harmful']['actual'], sum(x['actual_harmful_dots'] for x in actual))
            self.assertLess(rounding, 1e-7)
            self.assertEqual(summarize(rows.iloc[:0])['status'], 'unsupported')
            with self.assertRaises(ValueError):
                summarize(rows.assign(actual_dot=np.nan))


if __name__ == '__main__':
    unittest.main()
