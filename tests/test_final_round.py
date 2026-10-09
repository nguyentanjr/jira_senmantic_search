# -*- coding: utf-8 -*-
"""Test dung cap train va quy tac chon cua lan thu cuoi -- khong nap model.

    python -m unittest discover -s tests -p "test_final_round.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from build_task_pairs import expand, mine_negatives, near_dup_map, select
from final_eval import CURRENT, choose_final
from train_task_models import bi_columns, ce_columns


def _doc(key, actual, summary="s"):
    return {"key": key, "summary": summary, "sections": {"actual": actual}}


def _ho(text):
    return {"text_dense": text, "text_bm25": text}


class BuildPairsTest(unittest.TestCase):
    def test_expand_adds_near_duplicates(self):
        nb = near_dup_map([{"a": "RO-1", "b": "RO-2"}, {"a": "RO-3", "b": "RO-4"}])
        self.assertEqual(expand({"RO-1"}, nb), {"RO-1", "RO-2"})

    def test_select_skips_excluded_and_short(self):
        docs = [_doc("RO-1", "x" * 50), _doc("RO-2", "y" * 10), _doc("RO-3", "z" * 50)]
        hos = [_ho("doc " * 20), _ho("doc " * 20), _ho("doc " * 20)]
        got = [d["key"] for d, _, _ in select(docs, hos, {"RO-3"}, 40, 40, 1200)]
        self.assertEqual(got, ["RO-1"])

    def test_negatives_exclude_self_near_dup_and_same_summary(self):
        items = [(_doc("RO-1", "", "loi otp"), _ho("nhap sai otp khoa tai khoan"), "sai otp khoa"),
                 (_doc("RO-2", "", "khac"), _ho("nhap sai otp khoa tai khoan"), "q2"),
                 (_doc("RO-3", "", "loi otp"), _ho("sai otp khoa tai khoan"), "q3"),
                 (_doc("RO-4", "", "b"), _ho("sai otp"), "q4"),
                 (_doc("RO-5", "", "c"), _ho("so du vi"), "q5")]
        nb = near_dup_map([{"a": "RO-1", "b": "RO-2"}])
        neg = mine_negatives(items, nb, n_hard=2, n_random=1, pool=4, seed=1)[0]
        keys = {items[j][0]["key"] for j in neg}
        self.assertNotIn("RO-1", keys)        # chinh no
        self.assertNotIn("RO-2", keys)        # gan trung
        self.assertNotIn("RO-3", keys)        # cung summary
        self.assertIn("RO-4", keys)


class TrainColumnsTest(unittest.TestCase):
    rows = [{"query": "q", "positive": "p", "negatives": ["n1", "n2"]}]

    def test_bi_uses_first_negative(self):
        self.assertEqual(bi_columns(self.rows), {"anchor": ["q"], "positive": ["p"], "negative": ["n1"]})

    def test_ce_one_positive_per_query(self):
        c = ce_columns(self.rows)
        self.assertEqual(c["label"], [1.0, 0.0, 0.0])
        self.assertEqual(c["doc"], ["p", "n1", "n2"])


class ChooseFinalTest(unittest.TestCase):
    @staticmethod
    def _blocks(spec):
        ci = {qv: {} for qv in ("goc", "khong_dau")}
        pa = {qv: {} for qv in ("goc", "khong_dau")}
        for name, (mrr_g, mrr_n, hi_g, hi_n) in spec.items():
            ci["goc"][name] = {"mrr@10": {"mean": mrr_g}}
            ci["khong_dau"][name] = {"mrr@10": {"mean": mrr_n}}
            if name != CURRENT:
                pa["goc"][name] = {"recall@10": {"hi": hi_g}}
                pa["khong_dau"][name] = {"recall@10": {"hi": hi_n}}
        return ci, pa

    def test_picks_best_eligible(self):
        ci, pa = self._blocks({CURRENT: (0.45, 0.37, 0, 0),
                               "e5-task|ce-task|0.5": (0.50, 0.45, 0.01, 0.02),
                               "bge-m3|bge|0.3": (0.60, 0.50, -0.01, 0.02)})   # R@10 co dau giam
        self.assertEqual(choose_final(ci, pa, list(ci["goc"]))["chosen"], "e5-task|ce-task|0.5")

    def test_keeps_current_when_nothing_better(self):
        ci, pa = self._blocks({CURRENT: (0.45, 0.37, 0, 0), "x|y|0.3": (0.44, 0.37, 0.01, 0.01)})
        self.assertEqual(choose_final(ci, pa, list(ci["goc"]))["chosen"], CURRENT)


if __name__ == "__main__":
    unittest.main()
