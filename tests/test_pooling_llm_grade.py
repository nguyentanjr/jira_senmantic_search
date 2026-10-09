# -*- coding: utf-8 -*-
"""Test cho pipeline/pooling_llm_grade.py -- khong goi LLM.

    python -m unittest discover -s tests -p "test_pooling_llm_grade.py" -v

Kiem tra dung nhung cho da tung hong khi gan nhan bang tay: nhan lech dong,
dong da gan bi ghi de, vi du sai lot vao prompt, quy tac 1 bi LLM bo qua.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pooling_llm_grade import (REVIEW, RULE1_TAG, build_prompt,
                                        calibrate_rows, enforce_rule1,
                                        fill_rows, parse_judgments,
                                        weighted_kappa)


def _row(q, d, grade="", note="", res=""):
    return {"query_key": q, "doc_key": d, "grade": grade, "note": note,
            "query_summary": f"sum {q}", "doc_summary": f"sum {d}",
            "query_desc": "desc q", "doc_desc": "desc d",
            "query_error_codes": "", "doc_error_codes": "", "doc_status": "",
            "doc_resolution": res, "doc_has_resolution": str(int(bool(res))),
            "doc_url": ""}


def _reply(pairs, same_flow=True):
    return json.dumps({"judgments": [{"doc_key": k, "grade": g, "note": "n",
                                      "same_flow": same_flow}
                                     for k, g in pairs]})


class ParseTest(unittest.TestCase):
    def test_ok_int_grade_and_string_bool(self):
        got, err = parse_judgments("rac " + _reply([("B1", 2), ("B2", "?")], "false")
                                   + " rac", {"B1", "B2"})
        self.assertEqual(err, "")
        self.assertEqual(got["B1"], ("2", "n", False))
        self.assertEqual(got["B2"][0], "?")

    def test_missing_extra_dup_bad(self):
        for pairs in ([("B1", "1")],                          # thieu B2
                      [("B1", "1"), ("B2", "1"), ("B3", "1")],  # thua
                      [("B1", "1"), ("B1", "1")],             # lap
                      [("B1", "1"), ("B2", "3")]):            # grade sai
            got, err = parse_judgments(_reply(pairs), {"B1", "B2"})
            self.assertEqual(got, {})
            self.assertTrue(err)

    def test_same_flow_required(self):
        got, err = parse_judgments(_reply([("B1", "1")], same_flow=None), {"B1"})
        self.assertEqual(got, {})
        self.assertIn("same_flow", err)


class Rule1Test(unittest.TestCase):
    def test_cap_only_when_other_flow_and_no_resolution(self):
        g, n = enforce_rule1(_row("Q", "B"), "2", "x", same_flow=False)
        self.assertEqual(g, "1")
        self.assertIn(RULE1_TAG, n)
        self.assertEqual(enforce_rule1(_row("Q", "B", res="Root cause: ..."), "2", "x", False)[0], "2")
        self.assertEqual(enforce_rule1(_row("Q", "B"), "2", "x", True)[0], "2")
        self.assertEqual(enforce_rule1(_row("Q", "B"), "1", "x", False), ("1", "x"))


class FillTest(unittest.TestCase):
    def test_never_overwrites_labeled_rows(self):
        rows = [_row("Q1", "B1", "0", "nguoi"), _row("Q1", "B2"),
                _row("Q2", "B3", "2", "nguoi"), _row("Q2", "B4")]

        def judge(qrow, targets, examples):
            return {t["doc_key"]: ("2", "llm", False) for t in targets}, ""

        fill_rows(rows, judge, "m")
        self.assertEqual((rows[0]["grade"], rows[0]["note"]), ("0", "nguoi"))
        self.assertEqual(rows[0]["label_source"], "user")
        # LLM cham 2, khac flow, khong co cach sua -> ha xuong 1, giu diem goc
        self.assertEqual((rows[1]["grade"], rows[1]["llm_grade_raw"]), ("1", "2"))
        self.assertEqual(rows[1]["label_source"], "llm")

    def test_batches_split_and_partial_failure(self):
        rows = [_row("Q1", f"B{i}") for i in range(5)]
        sizes = []

        def judge(qrow, targets, examples):
            sizes.append(len(targets))
            if len(sizes) == 2:                      # lo thu hai loi
                return {}, "loi"
            return {t["doc_key"]: ("1", "", True) for t in targets}, ""

        stat = fill_rows(rows, judge, "m", batch_size=2)
        self.assertEqual(sizes, [2, 2, 1])
        self.assertEqual([r["grade"] for r in rows], ["1", "1", "", "", "1"])
        self.assertEqual(stat["loi_truy_van"], 1)

    def test_failed_query_left_blank(self):
        rows = [_row("Q1", "B1")]
        fill_rows(rows, lambda *a: ({}, "loi"), "m")
        self.assertEqual(rows[0]["grade"], "")

    def test_review_rows_not_examples_but_kept(self):
        q, d = next(iter(REVIEW))
        rows = [_row(q, d, "0", "sai truy van"), _row("QX", "BX", "2", "tot"),
                _row(q, "NEW")]
        seen = {}

        def judge(qrow, targets, examples):
            seen["ex"] = {(e["query_key"], e["doc_key"]) for e in examples}
            return {t["doc_key"]: ("1", "", True) for t in targets}, ""

        fill_rows(rows, judge, "m")
        self.assertNotIn((q, d), seen["ex"])
        self.assertIn(("QX", "BX"), seen["ex"])
        self.assertEqual((rows[0]["grade"], rows[0]["label_source"]), ("0", "review"))

    def test_refill_keeps_old_llm_rows_and_learns_only_from_humans(self):
        """Pool mo rong (v4): dau vao da qua fill -- nhan LLM cu giu nguyen tung o,
        khong cham lai, khong lam vi du; chi dong moi duoc cham."""
        old_llm = dict(_row("Q1", "L1", "0", "llm cu"), label_source="llm",
                       llm_model="gm", llm_grade_raw="0", llm_same_flow="1")
        old_review = dict(_row("Q2", "R1", "1", "llm review"), label_source="llm_review",
                          llm_model="gm", llm_grade_raw="1", llm_same_flow="0")
        user = dict(_row("Q1", "U1", "2", "nguoi"), label_source="user", llm_model="",
                    llm_grade_raw="", llm_same_flow="")
        rows = [user, old_llm, old_review, _row("Q1", "N1")]
        snapshot = (dict(old_llm), dict(old_review))
        seen = {}

        def judge(qrow, targets, examples):
            seen["targets"] = [t["doc_key"] for t in targets]
            seen["examples"] = [e["doc_key"] for e in examples]
            return {t["doc_key"]: ("1", "moi", True) for t in targets}, ""

        stat = fill_rows(rows, judge, "m")
        self.assertEqual(seen["targets"], ["N1"])
        self.assertEqual(seen["examples"], ["U1"])          # khong hoc tu nhan LLM
        self.assertEqual((rows[1], rows[2]), snapshot)
        self.assertEqual((rows[3]["grade"], rows[3]["label_source"]), ("1", "llm"))
        self.assertEqual((rows[0]["label_source"], stat["llm_cu"], stat["llm"]), ("user", 2, 1))


class ReviewTest(unittest.TestCase):
    def test_regrade_only_review_rows_with_user_examples(self):
        rows = [dict(_row("Q1", "U1", "2", "nguoi"), label_source="user"),
                dict(_row("Q1", "L1", "0", "llm"), label_source="llm"),
                dict(_row("Q1", "R1", "0", "sai truy van"), label_source="review")]
        seen = {}

        def judge(qrow, targets, examples):
            seen["targets"] = [t["doc_key"] for t in targets]
            seen["examples"] = [e["doc_key"] for e in examples]
            return {t["doc_key"]: ("1", "moi", True) for t in targets}, ""

        from pipeline.pooling_llm_grade import regrade_review_rows
        stat = regrade_review_rows(rows, judge, "m")
        self.assertEqual(seen["targets"], ["R1"])
        self.assertEqual(seen["examples"], ["U1"])          # khong lay dong llm
        r = rows[2]
        self.assertEqual((r["grade"], r["label_source"]), ("1", "llm_review"))
        self.assertEqual((r["review_old_grade"], r["review_old_note"]), ("0", "sai truy van"))
        self.assertEqual((rows[1]["grade"], rows[1]["label_source"]), ("0", "llm"))
        self.assertEqual(stat["llm_review"], 1)


class PromptTest(unittest.TestCase):
    def test_same_query_examples_separate_and_targets_listed(self):
        ex = [_row("Q1", "B1", "2", "cung query"), _row("Q2", "B9", "0", "khac")]
        p = build_prompt(_row("Q1", "x"), [_row("Q1", "B2")], ex)
        self.assertIn("ỨNG VIÊN CỦA CHÍNH A", p)
        self.assertLess(p.index("B = B9"), p.index("ISSUE ĐANG ĐIỀU TRA"))
        self.assertGreater(p.index("B = B2"), p.index("ỨNG VIÊN CẦN CHẤM"))
        # quy tac 1 can dong nay ke ca khi khong co resolution
        self.assertIn("Xử lý đã ghi nhận: (không có)", p)
        self.assertIn("QUY TẮC NGƯỜI GÁN ĐÃ CHỐT", p)
        self.assertIn("same_flow", p)


class CalibrateTest(unittest.TestCase):
    def test_holds_out_own_query(self):
        rows = [_row("Q1", "B1", "2"), _row("Q2", "B2", "0")]
        calls = []

        def judge(qrow, targets, examples):
            calls.append((qrow["query_key"], {e["query_key"] for e in examples}))
            return {t["doc_key"]: ("1", "", True) for t in targets}, ""

        res = calibrate_rows(rows, judge)
        self.assertEqual(len(res), 2)
        for q, ex_qs in calls:
            self.assertNotIn(q, ex_qs)

    def test_weighted_kappa(self):
        self.assertAlmostEqual(weighted_kappa([("0", "0"), ("1", "1"), ("2", "2")]), 1.0)
        near = weighted_kappa([("0", "0"), ("1", "2"), ("2", "2"), ("1", "1")])
        far = weighted_kappa([("0", "2"), ("1", "1"), ("2", "0"), ("1", "1")])
        self.assertGreater(near, far)


if __name__ == "__main__":
    unittest.main()
