import unittest
import pandas as pd
from scripts.build_additional_domain_anchors import choose_anchors


class AdditionalAnchorTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(uniquepid=['a','a','a','b'],variable=['bun']*4,
            stay_id=['s1','s1','s2','s3'],t=pd.to_datetime(['2100-01-01','2100-01-02','2100-01-03','2100-01-04']),
            value=[1,2,3,4]))

    def test_order_and_duplicate_invariance(self):
        frame=self.frame()
        pd.testing.assert_frame_equal(choose_anchors(frame),
            choose_anchors(pd.concat([frame,frame]).sample(frac=1,random_state=7)))

    def test_numeric_values_not_selection_inputs(self):
        frame=self.frame(); changed=frame.copy(); changed['value']=[100,-50,0,900]
        pd.testing.assert_frame_equal(choose_anchors(frame),choose_anchors(changed))

    def test_local_minima_equal_global_across_stays(self):
        frame=self.frame()
        parts=pd.concat([choose_anchors(g) for _,g in frame.groupby('stay_id')])
        pd.testing.assert_frame_equal(choose_anchors(frame),choose_anchors(parts))
        self.assertEqual(len(choose_anchors(frame)),2)

    def test_empty_site(self):
        result=choose_anchors(self.frame().iloc[:0])
        self.assertTrue(result.empty)
        self.assertIn('priority',result)
