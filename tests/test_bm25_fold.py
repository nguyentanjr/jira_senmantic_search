# -*- coding: utf-8 -*-
"""Test tokenize(fold=...) va quy tac chon cua eval_bm25_fold -- khong nap model.

    python -m unittest discover -s tests -p "test_bm25_fold.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from retrieval import BM25, SearchEngine, tokenize


class TokenizeFoldTest(unittest.TestCase):
    def test_none_is_old_behaviour(self):
        self.assertEqual(tokenize("đăng nhập"), tokenize("đăng nhập", fold="none"))
        self.assertNotIn("dang", tokenize("đăng nhập"))

    def test_add_keeps_original_and_adds_folded_once(self):
        toks = tokenize("đăng nhập lỗi ER10005", fold="add")
        for t in ("đăng", "dang", "nhập", "nhap", "đăng_nhập", "dang_nhap", "er10005"):
            self.assertIn(t, toks)
        self.assertEqual(toks.count("er10005"), 1)

    def test_only_strips_everything(self):
        self.assertEqual(tokenize("đăng nhập", fold="only"), tokenize("dang nhap"))

    def test_undiacritized_query_matches_diacritized_doc_only_when_folded(self):
        docs = ["không đăng nhập được", "số dư ví"]
        q = "khong dang nhap duoc"
        for fold, want in (("none", 0.0), ("add", None), ("only", None)):
            bm = BM25([tokenize(d, fold=fold) for d in docs])
            s = bm.score(tokenize(q, fold=fold))
            if want is None:
                self.assertGreater(s[0], s[1])
            else:
                self.assertEqual(float(s[0]), want)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            tokenize("x", fold="maybe")
        with self.assertRaises(ValueError):
            SearchEngine([], bm25_fold="maybe")


class EngineAutoFoldTest(unittest.TestCase):
    docs = [{"key": "A", "text_bm25": "không đăng nhập được sau khi đổi mật khẩu"},
            {"key": "B", "text_bm25": "mất kết nối ER80011 system busy"},
            {"key": "C", "text_bm25": "sms mật khẩu không gửi"}]

    def _scores(self, fold, q):
        e = SearchEngine(self.docs, bm25_fold=fold)
        e.build_bm25()
        return e.bm25_scores(q).tolist()

    def test_diacritic_query_uses_plain_index(self):
        q = "không đăng nhập mật khẩu"
        self.assertEqual(self._scores("auto", q), self._scores("none", q))

    def test_undiacritized_query_uses_add_index(self):
        q = "khong dang nhap mat khau"
        self.assertEqual(self._scores("auto", q), self._scores("add", q))
        self.assertGreater(self._scores("auto", q)[0], 0.0)


class ChooseTest(unittest.TestCase):
    @staticmethod
    def _blk(nd, go):
        def d(x):
            delta, lo, hi, sig = x
            return {"recall@10": {"delta": delta, "lo": lo, "hi": hi, "significant": sig}}
        return {"khong_dau": {"hybrid": {f"{v} - none": d(nd[v]) for v in nd}},
                "goc": {"hybrid": {f"{v} - none": d(go[v]) for v in go}}}

    def test_rule(self):
        from eval_bm25_fold import choose
        blk = self._blk(
            nd={"add": (0.08, 0.05, 0.11, True), "only": (0.10, 0.07, 0.13, True)},
            go={"add": (-0.004, -0.02, 0.01, False), "only": (-0.03, -0.05, -0.01, True)})
        c = choose(blk, ["none", "add", "only"], "hybrid")
        self.assertEqual(c["chosen"], "add")                 # only lam hai truy van co dau
        self.assertFalse(c["checks"]["only"]["b_no_diacritic_loss"])

    def test_nothing_qualifies(self):
        from eval_bm25_fold import choose
        blk = self._blk(nd={"add": (0.01, -0.01, 0.03, False)},
                        go={"add": (0.0, -0.01, 0.01, False)})
        self.assertEqual(choose(blk, ["none", "add"], "hybrid")["chosen"], "none")


if __name__ == "__main__":
    unittest.main()
