# -*- coding: utf-8 -*-
"""Test bo do tieu de, cap train v2 va quy tac chon vong 2 -- khong nap model.

    python -m unittest discover -s tests -p "test_round2.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from build_task_pairs import mine_negatives, near_dup_map
from build_title_eval import build_title_holdout, title_query
from final_eval2 import CURRENT, choose2


class TitleTest(unittest.TestCase):
    def test_title_query_strips_tags(self):
        self.assertEqual(title_query("[EU APP][Buy Bundle] Lỗi đơ màn hình  sau khi xem"),
                         "Lỗi đơ màn hình sau khi xem")

    def test_holdout_removes_summary_and_hidden_signals(self):
        d = {"key": "RO-1", "summary": "[APP] Lỗi ER10004 khi tạo request",
             "text_business": "Bấm submit báo lỗi", "text_technical": "trace abc",
             "sections": {"expected": "tạo thành công"}, "labels": [],
             "signals": {"error_code": ["ER10004"]}}
        h = build_title_holdout(d)
        self.assertNotIn("Lỗi ER10004 khi tạo request", h["text_dense"] + h["text_bm25"])
        self.assertNotIn("[APP]", h["text_bm25"])
        self.assertEqual(h["signals"], {})
        self.assertIn("tạo thành công", h["text_dense"])

    def test_holdout_removes_title_repeated_in_description(self):
        d = {"key": "RO-2", "summary": "[APP] Lỗi đơ màn hình sau khi xem saved list",
             "text_business": "Mô tả: lỗi đơ màn hình sau khi xem Saved list\nBước 1 mở app",
             "text_technical": "", "sections": {}, "labels": [], "signals": {}}
        h = build_title_holdout(d)
        self.assertNotIn("sau khi xem saved list", h["text_dense"].lower())
        self.assertIn("Bước 1 mở app", h["text_dense"])


class NegativesTest(unittest.TestCase):
    def test_never_other_view_of_same_issue(self):
        same = {"key": "RO-1", "summary": ""}
        items = [(same, {"text_bm25": "otp khoa tai khoan"}, "otp khoa"),
                 (same, {"text_bm25": "otp khoa tai khoan sai"}, "otp khoa"),
                 ({"key": "RO-2", "summary": ""}, {"text_bm25": "otp sai"}, "q"),
                 ({"key": "RO-3", "summary": ""}, {"text_bm25": "so du"}, "q")]
        neg = mine_negatives(items, near_dup_map([]), n_hard=3, n_random=0, pool=3, seed=1)[0]
        self.assertNotIn("RO-1", {items[j][0]["key"] for j in neg})


class Choose2Test(unittest.TestCase):
    slices = ["paraphrase:goc", "paraphrase:khong_dau", "title:goc", "title:khong_dau"]

    def _blocks(self, spec):
        ci = {s: {} for s in self.slices}
        pa = {s: {} for s in self.slices}
        for name, (mrrs, his) in spec.items():
            for s, m, h in zip(self.slices, mrrs, his):
                ci[s][name] = {"mrr@10": {"mean": m}}
                if name != CURRENT:
                    pa[s][name] = {"recall@10": {"hi": h}}
        return ci, pa

    def test_requires_all_four_slices(self):
        ci, pa = self._blocks({
            CURRENT: ([0.5] * 4, [0] * 4),
            "a|b|0.3": ([0.7, 0.7, 0.7, 0.7], [0.01, 0.01, 0.01, -0.01]),   # title khong dau giam
            "c|d|0.5": ([0.6, 0.6, 0.6, 0.6], [0.02] * 4)})
        self.assertEqual(choose2(ci, pa, list(ci[self.slices[0]]), self.slices)["chosen"], "c|d|0.5")

    def test_keeps_current(self):
        ci, pa = self._blocks({CURRENT: ([0.5] * 4, [0] * 4), "x|y|0.3": ([0.49] * 4, [0.01] * 4)})
        self.assertEqual(choose2(ci, pa, list(ci[self.slices[0]]), self.slices)["chosen"], CURRENT)


if __name__ == "__main__":
    unittest.main()
