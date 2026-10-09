# -*- coding: utf-8 -*-
"""Test tach dev/test va loc du lieu train cross-encoder -- khong nap model.

    python -m unittest discover -s tests -p "test_eval_rerankers.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from eval_rerankers import filter_train_rows, make_split, metrics_from_ranked


class SplitTest(unittest.TestCase):
    def test_dev_test_disjoint_and_cover_clean(self):
        s = make_split(["a", "b", "c", "d"], ["b", "d"])
        self.assertEqual(s, {"dev": ["b", "d"], "test": ["a", "c"]})

    def test_dev_outside_clean_rejected(self):
        with self.assertRaises(ValueError):
            make_split(["a", "b"], ["z"])

    def test_filter_drops_rows_touching_eval_keys(self):
        rows = [{"a_key": "x", "b_key": "y"}, {"a_key": "a", "b_key": "y"},
                {"a_key": "x", "b_key": "c"}]
        self.assertEqual(filter_train_rows(rows, ["a", "c"]), [rows[0]])

    def test_metrics_single_relevant(self):
        m = metrics_from_ranked([5, 7, 9], {7})
        self.assertEqual(m["recall@10"], 1.0)
        self.assertEqual(m["mrr@10"], 0.5)
        self.assertAlmostEqual(m["ndcg@10"], 1 / 1.5849625, places=5)
        self.assertEqual(metrics_from_ranked([1, 2], {3})["recall@10"], 0.0)


if __name__ == "__main__":
    unittest.main()
