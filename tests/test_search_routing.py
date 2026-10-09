# -*- coding: utf-8 -*-
"""Test dinh tuyen hai luong cua pipeline/search.py -- khong nap model.

    python -m unittest discover -s tests -p "test_search_routing.py" -v

Quy tac duoc chot tu so do (HANDOFF.md §5.4c): mo ta tu do -> hybrid+rerank,
tu mot issue -> hybrid. Test nay giu cho quy tac khong bi doi am tham.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.search import plan


class RoutingTest(unittest.TestCase):
    def test_free_text_uses_rerank_and_asymmetric_prefix(self):
        p = plan(query="khong dang nhap duoc sau khi doi mat khau")
        self.assertEqual((p["flow"], p["config"]), ("text", "hybrid+rerank"))
        self.assertEqual((p["query_prefix"], p["passage_prefix"]), ("query: ", "passage: "))
        self.assertEqual(p["bm25_fold"], "auto")
        self.assertEqual((p["rerank_fusion"], p["rerank_weight"]), ("blend", 0.3))
        self.assertEqual(p["rerank_k"], 20)
        self.assertEqual((p["model"], p["reranker"]),
                         ("models/e5-base-task2", "models/bge-reranker-base-task"))
        self.assertEqual(plan(key="RO-11708")["model"], "models/e5-base-task2")
        self.assertTrue(p["measured"])
        self.assertEqual(plan(key="RO-11708")["bm25_fold"], "none")

    def test_issue_key_uses_hybrid_and_symmetric_prefix(self):
        p = plan(key="RO-11708")
        self.assertEqual((p["flow"], p["config"]), ("issue", "hybrid"))
        self.assertEqual(p["rerank_k"], 20)  # chi dung neu ghi de --rerank on
        self.assertEqual((p["query_prefix"], p["passage_prefix"]), ("query: ", "query: "))

    def test_query_that_is_an_issue_key_routes_to_issue(self):
        p = plan(query=" RO-11708 ")
        self.assertEqual((p["flow"], p["key"], p["config"]), ("issue", "RO-11708", "hybrid"))

    def test_override_marks_unmeasured(self):
        p = plan(key="RO-11708", rerank="on")
        self.assertEqual(p["config"], "hybrid+rerank")
        self.assertFalse(p["measured"])
        self.assertEqual(plan(query="abc", rerank="off")["config"], "hybrid")

    def test_invalid_inputs(self):
        for kw in ({}, {"query": "abc", "key": "RO-1"}, {"query": "abc", "rerank": "x"}):
            with self.assertRaises(ValueError):
                plan(**kw)


if __name__ == "__main__":
    unittest.main()
