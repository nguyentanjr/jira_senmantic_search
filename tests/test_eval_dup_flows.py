# -*- coding: utf-8 -*-
"""Test quy tac chon va bootstrap theo cum cua eval_dup_flows -- khong nap model.

    python -m unittest discover -s tests -p "test_eval_dup_flows.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

from eval_dup_flows import choose, cluster_means, paired_diff


def _d(delta, sig):
    return {"delta": delta, "lo": delta - 0.01, "hi": delta + 0.01, "significant": sig}


class ChooseTest(unittest.TestCase):
    def test_issue_needs_significant_ndcg_gain(self):
        self.assertFalse(choose("issue", {"rerank - norerank": {"ndcg@10": _d(0.02, False)}})["rerank"])
        self.assertFalse(choose("issue", {"rerank - norerank": {"ndcg@10": _d(-0.05, True)}})["rerank"])
        self.assertTrue(choose("issue", {"rerank - norerank": {"ndcg@10": _d(0.05, True)}})["rerank"])

    def test_text_defaults_to_asym_and_checks_rerank_of_chosen_prefix(self):
        blk = {"sym - asym": {"recall@10": _d(0.01, False)},
               "asym+rerank - asym": {"recall@10": _d(0.04, True)},
               "sym+rerank - sym": {"recall@10": _d(-0.02, False)}}
        self.assertEqual(choose("text", blk), {"prefix": "asym", "rerank": True})

    def test_text_switches_to_sym_only_when_significant(self):
        blk = {"sym - asym": {"recall@10": _d(0.03, True)},
               "asym+rerank - asym": {"recall@10": _d(0.04, True)},
               "sym+rerank - sym": {"recall@10": _d(0.01, False)}}
        self.assertEqual(choose("text", blk), {"prefix": "sym", "rerank": False})


class ClusterBootstrapTest(unittest.TestCase):
    def test_single_member_clusters_match_plain_mean(self):
        v = np.array([0.0, 1.0, 1.0, 0.0])
        bs = cluster_means(v, np.arange(4), 2000)
        self.assertAlmostEqual(bs.mean(), 0.5, delta=0.03)

    def test_identical_arrays_give_zero_difference(self):
        v = np.array([0.2, 0.4, 0.9])
        d = paired_diff(v, v, np.array([0, 0, 1]), 500)
        self.assertEqual((d["delta"], d["significant"]), (0.0, False))


if __name__ == "__main__":
    unittest.main()
