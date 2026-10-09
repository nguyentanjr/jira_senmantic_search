# -*- coding: utf-8 -*-
"""Test cho pipeline/build_finetune_pairs.py -- khong can corpus.

    python -m unittest discover -s tests -p "test_build_finetune_pairs.py" -v

Kiem tra dung cho da hong o tap train cu: issue cua bo do lot vao cap train.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.build_finetune_pairs import GOLD_LABEL, WEAK_CE_LABEL, build

TEXT = {k: f"text {k}" for k in ("A", "B", "C", "D", "E", "F")}


class BuildTest(unittest.TestCase):
    def test_excluded_issue_never_in_output(self):
        gold = [{"a": "A", "b": "B", "grade": 2}, {"a": "A", "b": "E", "grade": 1}]
        weak = [{"a": "C", "b": "E"}, {"a": "C", "b": "D"}]
        hard = [{"anchor": "E", "negative": "F"}]
        easy = [{"anchor": "B", "negative": "D"}]
        bi, ce, stat = build(gold, weak, hard, easy, TEXT, exclude={"E"})
        keys = {k for r in bi + ce for k in (r["a_key"], r["b_key"])}
        self.assertNotIn("E", keys)
        self.assertEqual(stat["gold_bo_do_loai"], 1)
        self.assertEqual(stat["weak_bo_do_loai"], 1)
        self.assertEqual(stat["hard_neg_bo_do_loai"], 1)

    def test_labels_by_grade_and_source(self):
        bi, ce, _ = build([{"a": "A", "b": "B", "grade": 2},
                           {"a": "C", "b": "D", "grade": 1}],
                          [{"a": "E", "b": "F"}], [], [], TEXT, exclude=set())
        lab = {(r["a_key"], r["b_key"]): r["label"] for r in ce}
        self.assertEqual(lab[("A", "B")], GOLD_LABEL[2])
        self.assertEqual(lab[("C", "D")], GOLD_LABEL[1])
        self.assertEqual(lab[("E", "F")], WEAK_CE_LABEL)
        self.assertEqual(len(bi), 3)

    def test_duplicate_positive_kept_once_gold_first(self):
        bi, _, _ = build([{"a": "A", "b": "B", "grade": 2}],
                         [{"a": "B", "b": "A"}], [], [], TEXT, exclude=set())
        self.assertEqual([r["source"] for r in bi], ["gold2"])

    def test_negative_that_is_a_positive_is_dropped(self):
        _, ce, _ = build([{"a": "A", "b": "B", "grade": 2}], [], [],
                         [{"anchor": "B", "negative": "A"}], TEXT, exclude=set())
        self.assertEqual([r["source"] for r in ce], ["gold2"])

    def test_missing_text_and_unknown_grade_skipped(self):
        bi, _, stat = build([{"a": "A", "b": "ZZ", "grade": 2},
                             {"a": "A", "b": "B", "grade": 5}], [], [], [],
                            TEXT, exclude=set())
        self.assertEqual(bi, [])
        self.assertEqual(stat["gold_bo_khong_co_van_ban"], 1)
        self.assertEqual(stat["gold_bo_grade_la"], 1)


if __name__ == "__main__":
    unittest.main()
