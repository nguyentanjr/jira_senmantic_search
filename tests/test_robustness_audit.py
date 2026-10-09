import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from audit_search_robustness import pilot_diagnostics, replay_stage, variants


class RobustnessAuditTest(unittest.TestCase):
    def test_variants_preserve_identifiers_and_meaning_bearing_words(self):
        text = "Lỗi đăng nhập ER80011 /api/v1/login trans_id không thành công"
        values = variants(text)
        for name in ("mixed_a", "mixed_b"):
            self.assertNotEqual(values[name], text)
            self.assertNotEqual(values[name], values["khong_dau"])
            for term in ("ER80011", "/api/v1/login", "trans_id"):
                self.assertIn(term, values[name])
        self.assertEqual(variants("ER80011 API")["mixed_a"], "ER80011 API")

    def test_stage_partition_distinguishes_missing_candidates_and_head(self):
        keys = [str(i) for i in range(100)]
        for relevant, expected in (("missing", "missing_top100"), ("90", "outside_top20"),
                                   ("18", "inside_head_ranking"), ("0", "success")):
            result = replay_stage({"hybrid": keys, "ce": [0.5] * 30, "relevant": [relevant]})
            self.assertEqual(result["failure_stage"], expected)

    def test_cache_with_missing_ce_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_stage({"hybrid": ["a", "b"], "ce": [0.5], "relevant": ["b"]})

    def test_feedback_clears_and_unjudged_are_respected(self):
        searches = [{"search_id": "s", "query": "q"}, {"search_id": "unjudged"}]
        events = [{"search_id": "s", "result_key": "A", "verdict": "not_related", "reason": "wrong_module"},
                  {"search_id": "s", "result_key": "A", "verdict": "cleared"},
                  {"search_id": "s", "verdict": "no_answer"},
                  {"search_id": "s", "result_key": "B", "verdict": "related"},
                  {"search_id": "orphan", "result_key": "C", "verdict": "not_related"}]
        report, failures = pilot_diagnostics(searches, events)
        self.assertEqual(report["n_final_labels"], 1)
        self.assertEqual(report["no_answer_in_shown_results"], 0)
        self.assertEqual(report["negative_reasons"], {})
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
