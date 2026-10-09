# -*- coding: utf-8 -*-
"""Test phan loai ket qua rerank tung truy van -- khong nap model.

    python -m unittest discover -s tests -p "test_analyze_rerank_damage.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from analyze_rerank_damage import _auc, d_rr, outcome, summarize
from eval_bm25_fold import answer_rank


class OutcomeTest(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(outcome(3, 3), "same")
        self.assertEqual(outcome(2, 14), "pushed_out")
        self.assertEqual(outcome(25, 4), "pulled_in")
        self.assertEqual(outcome(5, 2), "up")
        self.assertEqual(outcome(12, 20), "down")
        self.assertEqual(outcome(None, None), "same")
        self.assertEqual(outcome(1, None), "pushed_out")

    def test_d_rr(self):
        self.assertAlmostEqual(d_rr(1, 2), -0.5)
        self.assertAlmostEqual(d_rr(None, 4), 0.25)
        self.assertEqual(d_rr(None, None), 0.0)

    def test_summary_counts_only_in_scope(self):
        rows = [{"rank_hybrid": 1, "rank_rerank": 20, "outcome": "pushed_out", "d_rr": d_rr(1, 20)},
                {"rank_hybrid": 8, "rank_rerank": 2, "outcome": "up", "d_rr": d_rr(8, 2)},
                {"rank_hybrid": None, "rank_rerank": None, "outcome": "same", "d_rr": 0.0}]
        s = summarize(rows, k=30)
        self.assertEqual((s["n_total"], s["n_in_scope"]), (3, 2))
        self.assertEqual((s["helped_pct"], s["hurt_pct"], s["severe_pct"]), (50.0, 50.0, 50.0))

    def test_answer_rank_and_auc(self):
        self.assertEqual(answer_rank([7, 3, 9], {9}), 3)
        self.assertIsNone(answer_rank([7, 3], {9}))
        self.assertEqual(_auc([0.1, 0.2], [0.5, 0.6]), 1.0)
        self.assertIsNone(_auc([], [1.0]))


if __name__ == "__main__":
    unittest.main()
