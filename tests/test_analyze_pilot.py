from __future__ import annotations

import unittest

from pilot.analyze_feedback import analyze


class AnalyzePilotTest(unittest.TestCase):
    def test_precision_only_uses_complete_top5(self):
        searches = [
            {"search_id": "s1", "flow": "text", "review_mode": "full_top5",
             "results": [{"rank": i, "key": f"D{i}"} for i in range(1, 11)],
             "latency": {"total_ms": 100}},
            {"search_id": "s2", "flow": "text", "review_mode": "standard",
             "results": [{"rank": 1, "key": "X"}], "latency": {"total_ms": 200}},
        ]
        feedback = [
            *[{"search_id": "s1", "result_key": f"D{i}",
               "verdict": "related" if i in (1, 4) else "not_related",
               "reason": "wrong_module"} for i in range(1, 6)],
            {"search_id": "s2", "result_key": "X", "verdict": "related"},
        ]
        report = analyze(searches, feedback)
        self.assertEqual(report["precision_from_complete_top5_only"]["n"], 1)
        self.assertEqual(report["precision_from_complete_top5_only"]["p@3"], 0.3333)
        self.assertEqual(report["precision_from_complete_top5_only"]["p@5"], 0.4)
        self.assertEqual(report["latency_ms"]["text"]["total_ms"]["p50"], 150.0)

    def test_no_answer_is_explicit_and_completes_review(self):
        searches = [{"search_id": "s", "flow": "issue", "review_mode": "full_top5",
                     "results": [{"rank": i, "key": f"D{i}"} for i in range(1, 6)],
                     "latency": {}}]
        report = analyze(searches, [{"search_id": "s", "verdict": "no_answer"}])
        self.assertEqual(report["status_counts"]["no_answer"], 1)
        self.assertEqual(report["precision_from_complete_top5_only"]["p@5"], 0.0)


    def test_changed_mind_and_cleared_labels(self):
        searches = [{"search_id": "s", "flow": "text", "review_mode": "standard",
                     "results": [{"rank": 1, "key": "A"}, {"rank": 2, "key": "B"}],
                     "latency": {}}]
        feedback = [
            {"search_id": "s", "result_key": "A", "verdict": "related"},
            {"search_id": "s", "result_key": "A", "verdict": "not_related",
             "reason": "wrong_module"},
            {"search_id": "s", "result_key": "B", "verdict": "related"},
            {"search_id": "s", "result_key": "B", "verdict": "cleared"},
        ]
        report = analyze(searches, feedback)
        row = report["per_search"][0]
        self.assertEqual(row["n_labeled"], 1)
        self.assertFalse(row["has_related_click"])
        self.assertEqual(report["negative_reasons"], {"wrong_module": 1})
        self.assertEqual(report["label_changes"]["labels_cleared"], 1)

    def test_related_after_no_answer_wins(self):
        searches = [{"search_id": "s", "flow": "text", "review_mode": "standard",
                     "results": [{"rank": 1, "key": "A"}], "latency": {}}]
        report = analyze(searches, [
            {"search_id": "s", "verdict": "no_answer"},
            {"search_id": "s", "result_key": "A", "verdict": "related"},
        ])
        self.assertEqual(report["status_counts"].get("no_answer", 0), 0)
        self.assertEqual(report["status_counts"]["has_related_click"], 1)

    def test_disabled_searches_are_not_counted_as_unknown(self):
        searches = [{"search_id": "s", "flow": "text", "review_mode": "disabled",
                     "results": [{"rank": 1, "key": "A"}], "latency": {}}]
        report = analyze(searches, [])
        self.assertEqual(report["status_counts"], {"feedback_disabled": 1})


if __name__ == "__main__":
    unittest.main()
