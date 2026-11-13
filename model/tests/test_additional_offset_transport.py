import unittest
import pandas as pd
from scripts.diagnose_additional_offset_transport import leave_component_out,transfer_delta


class TransportTests(unittest.TestCase):
    def test_held_component_never_fits_itself(self):
        f=pd.DataFrame(dict(patient=[str(i) for i in range(120)],component=['a']*30+['b']*30+['c']*30+['d']*30,
            truth=[9.]*30+[2.]*90,real=0.,process=0.,population=0.))
        r=leave_component_out(f)
        self.assertTrue(r['complete'])
        self.assertEqual(r['groups'][0]['offset'],2.)
        f.loc[f.component=='a','truth']=99.
        self.assertEqual(leave_component_out(f)['groups'][0]['offset'],2.)

    def test_insufficient_not_success(self):
        f=pd.DataFrame(dict(patient=[str(i) for i in range(90)],component=['a']*30+['b']*30+['c']*30,
            truth=2.,real=0.,process=0.,population=0.))
        r=leave_component_out(f)
        self.assertFalse(r['complete']);self.assertIsNone(r['mean_delta'])
        self.assertEqual(transfer_delta(f,2.),dict(patient=-2.,component=-2.))


if __name__=='__main__':unittest.main()
