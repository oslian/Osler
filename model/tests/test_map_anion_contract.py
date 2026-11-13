import unittest
import pandas as pd
from scripts.audit_map_anion_contract import panels


class PanelTests(unittest.TestCase):
    def test_exact_panel_and_conflict_exclusion(self):
        f = pd.DataFrame({'stay_id':['s']*5,'measured_at':[pd.Timestamp('2100-01-01')]*5,
            'variable':['anion_gap','sodium','chloride','bicarbonate','potassium'],
            'value':[10.,140.,106.,24.,4.]})
        p = panels(f)
        self.assertEqual(p.without_k_error.iloc[0],0.)
        self.assertEqual(p.with_k_error.iloc[0],-4.)
        other = f.iloc[[0]].copy()
        other['value'] = 12.
        self.assertEqual(len(panels(pd.concat([f,other]))),0)
        f.loc[1,'measured_at'] += pd.Timedelta(minutes=1)
        self.assertEqual(len(panels(f)),0)


if __name__ == '__main__':
    unittest.main()
