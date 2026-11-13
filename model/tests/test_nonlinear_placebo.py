import unittest

import numpy as np
import pandas as pd

from scripts.train_nonlinear_placebo import donor_indices, fit_placebo, verify_query
from scripts.train_nonlinear_probe import PARAMETERS


class NonlinearPlaceboTests(unittest.TestCase):
    def test_within_component_derangement_and_order_invariance(self):
        ids = np.array(['c','a','b','d','e','f'])
        groups = np.array(['x','x','x','y','y','z'])
        ix = donor_indices(ids, groups, 3)
        np.testing.assert_array_equal(groups, groups[ix])
        self.assertEqual(set(ix), set(range(6)))
        self.assertEqual(np.flatnonzero(ix == np.arange(6)).tolist(), [5])
        order = np.array([5,2,3,0,1,4])
        j = donor_indices(ids[order], groups[order], 3)
        self.assertEqual(dict(zip(ids,ids[ix])), dict(zip(ids[order],ids[order][j])))

    def test_capacity_and_singleton_count(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(50,64)); y = x[:,0]
        model, unchanged = fit_placebo(x,y,[str(i) for i in range(50)],['a']*49+['b'],19)
        self.assertEqual(unchanged,1)
        self.assertEqual(model.n_iter_,100)
        for k,v in PARAMETERS.items(): self.assertEqual(model.get_params()[k],v)

    def test_identity_and_component_leakage_rejected(self):
        ids=['a','b']; values=dict(truth=np.array([1.,2.]),prediction=np.array([2.,3.]))
        prior=pd.DataFrame(dict(patient=ids,component=['x','x'],truth=[1.,2.],native=[2.,3.]))
        roles=dict(a='selection',b='selection')
        verify_query(ids,values,prior,roles,'selection',['c'],['z'])
        with self.assertRaises(ValueError): verify_query(ids,values,prior,roles,'test',['c'],['z'])
        with self.assertRaises(ValueError): verify_query(ids,values,prior,roles,'selection',['a'],['z'])
        with self.assertRaises(ValueError): verify_query(ids,values,prior,roles,'selection',['c'],['x'])
        values['truth'][0]=3
        with self.assertRaises(ValueError): verify_query(ids,values,prior,roles,'selection',['c'],['z'])

    def test_duplicate_identity_rejected(self):
        with self.assertRaises(ValueError): donor_indices(['a','a'],['x','x'],19)


if __name__ == '__main__':
    unittest.main()
