import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from probe_structured_rank import minmax, metrics, ndcg_pool

class StructuredRankProbeTests(unittest.TestCase):
    def test_minmax_constant_is_neutral(self):
        np.testing.assert_array_equal(minmax([2,2]),[.5,.5])
    def test_unknown_is_not_zero(self):
        m=metrics(['unknown','zero','positive'],{'zero':0,'positive':1})
        self.assertEqual(m['known_zero_count10'],1)
        self.assertEqual(m['known_recall10'],1)
        self.assertAlmostEqual(m['judged_fraction10'],2/3)
    def test_grade_one_is_relevant(self):
        self.assertEqual(metrics(['related'],{'related':1})['known_recall10'],1)
    def test_pool_ndcg_ignores_unknown(self):
        self.assertEqual(ndcg_pool(['same','unknown','related'],{'same':2,'related':1}),1)

if __name__=='__main__': unittest.main()
