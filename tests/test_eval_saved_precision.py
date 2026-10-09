from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_saved_precision import evaluate


class SavedPrecisionTest(unittest.TestCase):
    def test_fixed_denominator_relevance_and_unjudged(self):
        judgments = {
            "Q1": {"A": 2, "B": 0, "C": 1},
            "Q2": {"D": 0},  # excluded: no relevant document in pool
        }
        runs = {"keyword": {"Q1": ["A", "B", "X"]}}
        result = evaluate(judgments, runs, [1, 3], n_boot=50, seed=1)
        self.assertEqual(result["n_queries_scored"], 1)
        self.assertEqual(result["excluded_no_relevant"], ["Q2"])
        metrics = result["configs"]["keyword"]
        self.assertEqual(metrics["precision@1"]["mean"], 1.0)
        self.assertEqual(metrics["precision@3"]["mean"], 0.3333)
        self.assertEqual(metrics["unjudged_rate@3"]["mean"], 0.3333)

    def test_paired_difference_uses_same_queries(self):
        judgments = {"Q": {"A": 1, "B": 0}}
        runs = {"keyword": {"Q": ["B"]}, "hybrid": {"Q": ["A"]}}
        result = evaluate(judgments, runs, [1], n_boot=20, seed=2)
        delta = result["paired_vs_baseline"]["hybrid"]["precision@1"]
        self.assertEqual(delta["delta"], 1.0)
        self.assertTrue(delta["significant"])


if __name__ == "__main__":
    unittest.main()
