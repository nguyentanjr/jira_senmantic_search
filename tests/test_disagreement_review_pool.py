import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from build_disagreement_review_pool import family, defect_text, rid

class DisagreementPoolTests(unittest.TestCase):
    def test_defect_text_does_not_leak_expected(self):
        self.assertEqual(defect_text({'summary':'S','sections':{'actual':'A','expected':'E'}}),'S\nA')
    def test_family_drops_channel_actor_tags(self):
        self.assertEqual(family({'key':'X','summary':'[CMS Agent][Register Channel][Save draft] lỗi'}),'register_channel+save_draft')
    def test_review_id_is_directional_and_stable(self):
        self.assertEqual(rid('A','B'),rid('A','B')); self.assertNotEqual(rid('A','B'),rid('B','A'))

if __name__=='__main__': unittest.main()
