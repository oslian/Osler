from copy import deepcopy
import unittest
import numpy as np
from scripts.preflight_full_source_exposure import draw_queue


class QueueTest(unittest.TestCase):
    def test_no_repeat_until_exhaustion(self):
        state={};rng=np.random.default_rng(19);keys=list(range(10))
        first=draw_queue(keys,state,rng,6);second=draw_queue(keys,state,rng,4)
        self.assertEqual(len(set(first+second)),10)

    def test_resume_is_exact(self):
        state={};rng=np.random.default_rng(7);keys=list(range(5))
        draw_queue(keys,state,rng,3);saved=deepcopy(state);rs=deepcopy(rng.bit_generator.state)
        expected=draw_queue(keys,state,rng,20)
        rng2=np.random.default_rng();rng2.bit_generator.state=rs
        self.assertEqual(expected,draw_queue(keys,saved,rng2,20))

    def test_small_population_cycles(self):
        result=draw_queue([1,2],{},np.random.default_rng(3),6)
        self.assertEqual(result.count(1),3);self.assertEqual(result.count(2),3)
