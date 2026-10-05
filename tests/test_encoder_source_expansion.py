import unittest
import pandas as pd
from scripts.prepare_encoder_source_expansion import select_patients


class SourceExpansionTest(unittest.TestCase):
    def test_patient_not_stay_exclusion_and_nested_sampling(self):
        metadata=pd.DataFrame([
            ('a','old',1),('b','test',2),('c','cross',1),('d','cross',2),
            ('e','new1',1),('f','new1',1),('g','new2',1),('h','old',1),
        ],columns=['stay_id','uniquepid','hospitalid'])
        dev=pd.DataFrame({'stay_id':['a']});test=pd.DataFrame({'stay_id':['b']})
        full,report=select_patients(metadata,dev,test)
        self.assertEqual(set(full.uniquepid),{'new1','new2'})
        self.assertEqual(report['selected_patients'],2)
        small,_=select_patients(metadata,dev,test,1)
        reordered,_=select_patients(metadata.iloc[::-1],dev,test,1)
        self.assertEqual(set(small.stay_id),set(reordered.stay_id))
        self.assertTrue(set(small.uniquepid)<=set(full.uniquepid))

    def test_hospital_overlap_rejected(self):
        metadata=pd.DataFrame([('a','p1',1),('b','p2',1)],columns=['stay_id','uniquepid','hospitalid'])
        with self.assertRaisesRegex(ValueError,'overlapping hospitals'):
            select_patients(metadata,pd.DataFrame({'stay_id':['a']}),pd.DataFrame({'stay_id':['b']}))


if __name__=='__main__':unittest.main()
