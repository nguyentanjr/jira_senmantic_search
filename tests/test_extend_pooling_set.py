# -*- coding: utf-8 -*-
"""Test cho pipeline/extend_pooling_set.py -- khong chay retrieval.

    python -m unittest discover -s tests -p "test_extend_pooling_set.py" -v

Kiem tra dung nhung cho lam hong so do pooling neu sai: dong da cham bi dong
vao, cap da cham bi them lai, thu tu dong moi lo thu hang cua he moi.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.extend_pooling_set import merge_rows, plan_new_pairs


class PlanTest(unittest.TestCase):
    def test_skips_judged_self_and_duplicates(self):
        plan = plan_new_pairs({"Q1": ["A", "B", "Q1", "C", "A"], "Q2": ["D"]},
                              {"Q1": {"B"}, "Q2": {"D"}}, seed=15)
        self.assertEqual(sorted(plan["Q1"]), ["A", "C"])
        self.assertEqual(plan["Q2"], [])

    def test_order_does_not_leak_rank(self):
        judged = {"Q1": set()}
        a = plan_new_pairs({"Q1": ["A", "B", "C", "D", "E"]}, judged, seed=15)
        b = plan_new_pairs({"Q1": ["E", "D", "C", "B", "A"]}, judged, seed=15)
        self.assertEqual(a, b)            # dao thu hang -> cung thu tu dau ra


class MergeTest(unittest.TestCase):
    def test_base_rows_untouched_new_rows_after_their_query(self):
        base = [{"query_key": "Q1", "doc_key": "A", "grade": "2", "label_source": "llm"},
                {"query_key": "Q1", "doc_key": "B", "grade": "?", "label_source": "llm"},
                {"query_key": "Q2", "doc_key": "C", "grade": "0", "label_source": "user"}]
        snapshot = [dict(r) for r in base]
        new = [{"query_key": "Q2", "doc_key": "N2", "grade": "", "label_source": ""},
               {"query_key": "Q1", "doc_key": "N1", "grade": "", "label_source": ""}]
        out = merge_rows(base, new)
        self.assertEqual([(r["query_key"], r["doc_key"]) for r in out],
                         [("Q1", "A"), ("Q1", "B"), ("Q1", "N1"), ("Q2", "C"), ("Q2", "N2")])
        self.assertEqual(base, snapshot)
        self.assertEqual([r for r in out if r["grade"]], snapshot)

    def test_unknown_query_rejected(self):
        with self.assertRaises(ValueError):
            merge_rows([{"query_key": "Q1", "doc_key": "A"}],
                       [{"query_key": "QX", "doc_key": "N"}])


if __name__ == "__main__":
    unittest.main()
