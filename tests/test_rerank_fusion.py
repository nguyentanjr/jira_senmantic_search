# -*- coding: utf-8 -*-
"""Test fuse_rerank va quy tac chon cua select_rerank_fusion -- khong nap model.

    python -m unittest discover -s tests -p "test_rerank_fusion.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from retrieval import SearchConfig, fuse_rerank
from select_rerank_fusion import choose, ranking


class FuseTest(unittest.TestCase):
    head = np.array([10, 11, 12, 13])          # thu tu hybrid
    ce = np.array([0.10, 0.20, 0.30, 0.95])    # cross-encoder thich doc cuoi

    def test_replace_is_old_behaviour(self):
        self.assertEqual(fuse_rerank(self.head, self.ce).tolist(), [13, 12, 11, 10])

    def test_weight_extremes(self):
        for m in ("rrf", "blend"):
            self.assertEqual(fuse_rerank(self.head, self.ce, m, 0.0).tolist(), [10, 11, 12, 13])
            self.assertEqual(fuse_rerank(self.head, self.ce, m, 1.0).tolist(), [13, 12, 11, 10])

    def test_blend_protects_hybrid_top_when_ce_is_flat(self):
        ce = np.array([0.50, 0.51, 0.52, 0.53])  # CE gan nhu khong phan biet
        self.assertEqual(fuse_rerank(self.head, ce, "replace")[0], 13)
        self.assertEqual(fuse_rerank(self.head, ce, "blend", 0.5)[0], 10)

    def test_invalid_and_empty(self):
        with self.assertRaises(ValueError):
            fuse_rerank(self.head, self.ce, "vote")
        self.assertEqual(len(fuse_rerank(np.array([], dtype=int), np.array([]), "rrf")), 0)

    def test_config_default_keeps_replace(self):
        self.assertEqual(SearchConfig().rerank_fusion, "replace")

    def test_ranking_keeps_tail(self):
        row = {"hybrid": ["A", "B", "C", "D"], "ce": [0.1, 0.9]}
        self.assertEqual(ranking(row, "replace", None), ["B", "A", "C", "D"])
        self.assertEqual(ranking(row, "hybrid", None), ["A", "B", "C", "D"])


class ChooseTest(unittest.TestCase):
    @staticmethod
    def _d(delta, lo, hi):
        return {"delta": delta, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}

    def _blk(self, nd_mrr, nd_r10, go_r10, go_mrr):
        return {"khong_dau": {"x": {"mrr@10": self._d(*nd_mrr), "recall@10": self._d(*nd_r10)}},
                "goc": {"x": {"mrr@10": self._d(*go_mrr), "recall@10": self._d(*go_r10)}}}

    def test_passes(self):
        b = self._blk((0.03, 0.01, 0.05), (-0.005, -0.02, 0.01), (0.0, -0.01, 0.01), (0.0, -0.01, 0.01))
        self.assertEqual(choose(b, ["x"])["chosen"], "x")

    def test_fails_if_diacritic_recall_drops(self):
        b = self._blk((0.03, 0.01, 0.05), (0.0, -0.01, 0.01), (-0.03, -0.05, -0.01), (0.0, -0.01, 0.01))
        self.assertEqual(choose(b, ["x"])["chosen"], "replace")

    def test_fails_without_significant_mrr_gain(self):
        b = self._blk((0.01, -0.01, 0.03), (0.0, -0.01, 0.01), (0.0, -0.01, 0.01), (0.0, -0.01, 0.01))
        self.assertEqual(choose(b, ["x"])["chosen"], "replace")


if __name__ == "__main__":
    unittest.main()
