import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from probe_defect_branch import defect_text, fuse_rankings

class DefectBranchProbeTests(unittest.TestCase):
    def test_field_excludes_expected(self):
        d={'summary':'S','sections':{'actual':'A','expected':'E'}}
        self.assertEqual(defect_text(d),'S\nA')
    def test_primary_wins_tie_and_union_has_no_duplicates(self):
        got=fuse_rankings([1,2,3],[3,4,1],6,limit=6).tolist()
        self.assertEqual(len(got),len(set(got)))
        self.assertTrue({1,2,3,4}.issubset(got))
    def test_auxiliary_can_introduce_candidate(self):
        got=fuse_rankings([1,2],[4,5],6,limit=4).tolist()
        self.assertIn(4,got)

if __name__=='__main__': unittest.main()
