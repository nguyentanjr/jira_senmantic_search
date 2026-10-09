from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))

import csv
import json
import tempfile

from pipeline.build_feedback_pairs import build, existing_eval_keys


class FeedbackPairsTest(unittest.TestCase):
    def test_existing_eval_keys_covers_pooling_docs_and_text_flow_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold = os.path.join(tmp, "gold.jsonl")
            pooling = os.path.join(tmp, "pooling.csv")
            contrast = os.path.join(tmp, "contrast.csv")
            clean = os.path.join(tmp, "clean.json")
            with open(gold, "w", encoding="utf-8") as f:
                f.write(json.dumps({"a": "g-1", "b": "G-2"}) + "\n")
            with open(pooling, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["query_key", "doc_key", "grade"])
                w.writeheader()
                w.writerow({"query_key": "PQ-1", "doc_key": "pd-1", "grade": "0"})
            with open(contrast, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["issue_a", "issue_b", "source_a", "source_b"])
                w.writeheader()
                w.writerow({"issue_a": "CA-1", "issue_b": "CB-1", "source_a": "", "source_b": "CS-1"})
            with open(clean, "w", encoding="utf-8") as f:
                json.dump({"paraphrase_clean_keys": ["tx-1", "TX-2"]}, f)

            keys = existing_eval_keys(gold, pooling, contrast, clean)
            self.assertEqual(keys, {"G-1", "G-2", "PQ-1", "PD-1", "CA-1", "CB-1",
                                    "CS-1", "TX-1", "TX-2"})
            # Khong truyen clean split thi van giu hanh vi cu cho ba bo con lai.
            self.assertNotIn("TX-1", existing_eval_keys(gold, pooling, contrast))

    def test_click_on_text_flow_eval_issue_never_reaches_train(self):
        docs = {"P": {"summary": "p"}, "N": {"summary": "n"}, "TX": {"summary": "t"}}
        events = [
            {"flow": "text", "query": "q", "result_key": "TX", "verdict": "related"},
            {"flow": "text", "query": "q", "result_key": "N", "verdict": "not_related"},
            {"flow": "issue", "source_key": "TX", "result_key": "P", "verdict": "related"},
            {"flow": "issue", "source_key": "TX", "result_key": "N", "verdict": "not_related"},
        ]
        train, eval_rows, audit, _ = build(events, docs, {"TX"}, eval_fraction=1e-12)
        self.assertEqual(train, [])
        self.assertEqual(eval_rows, [])
        self.assertEqual(sum(r["status"] == "touches_existing_eval" for r in audit), 3)

    def test_latest_click_wins_and_only_explicit_negative_is_used(self):
        docs = {
            "P": {"summary": "positive"}, "N": {"summary": "negative"},
            "X": {"summary": "excluded"},
        }
        events = [
            {"flow": "text", "query": " Login lỗi ", "result_key": "P", "verdict": "not_related"},
            {"flow": "text", "query": "login   lỗi", "result_key": "P", "verdict": "related"},
            {"flow": "text", "query": "login lỗi", "result_key": "N", "verdict": "not_related"},
            {"flow": "text", "query": "login lỗi", "result_key": "X", "verdict": "related"},
        ]
        # Force all usable clicks to train without relying on a particular hash.
        train, eval_rows, audit, stats = build(events, docs, {"X"}, eval_fraction=1e-12)
        self.assertEqual(len(train), 1)
        self.assertEqual(train[0]["positive_key"], "P")
        self.assertEqual(train[0]["neg_keys"], ["N"])
        self.assertEqual(eval_rows, [])
        self.assertEqual(stats["conflicting_click_pairs"], 1)
        self.assertTrue(any(r["result_key"] == "X" and
                            r["status"] == "touches_existing_eval" for r in audit))

    def test_search_log_groups_clicks_and_tracks_full_top5(self):
        docs = {k: {"summary": k} for k in "PNABCD"}
        search = {"search_id": "s1", "flow": "text", "query": "q",
                  "source_key": "", "filters": {"project": "RO"},
                  "review_mode": "full_top5",
                  "results": [{"rank": i, "key": k}
                              for i, k in enumerate("PABCD", 1)]}
        events = [
            {"search_id": "s1", "result_key": "P", "rank": 99,
             "verdict": "related"},
            *[{"search_id": "s1", "result_key": k, "verdict": "not_related",
               "reason": "wrong_module"} for k in "ABCD"],
        ]
        train, _, _, stats = build(events, docs, set(), eval_fraction=1e-12,
                                   searches=[search])
        self.assertEqual(stats["full_top5_completed"], 1)
        self.assertEqual(train[0]["filters"], {"project": "RO"})
        self.assertEqual(set(train[0]["neg_keys"]), set("ABCD"))

    def test_cleared_and_no_answer_follow_event_order(self):
        docs = {k: {"summary": k} for k in "PNQ"}
        search = {"search_id": "s2", "flow": "text", "query": "q2", "source_key": "",
                  "filters": {}, "review_mode": "standard",
                  "results": [{"rank": i, "key": k} for i, k in enumerate("PNQ", 1)]}
        events = [
            {"search_id": "s2", "result_key": "P", "verdict": "related"},
            {"search_id": "s2", "result_key": "N", "verdict": "not_related",
             "reason": "wrong_defect_type"},
            {"search_id": "s2", "result_key": "Q", "verdict": "related"},
            {"search_id": "s2", "result_key": "Q", "verdict": "cleared"},
        ]
        train, _, audit, _ = build(events, docs, set(), eval_fraction=1e-12,
                                   searches=[search])
        self.assertEqual([(t["positive_key"], t["neg_keys"]) for t in train], [("P", ["N"])])
        self.assertNotIn("Q", {r["result_key"] for r in audit})

        # "Khong co dap an" bo positive nhung giu negative ro rang.
        train, eval_rows, audit, stats = build(
            events + [{"search_id": "s2", "verdict": "no_answer"}], docs, set(),
            eval_fraction=1e-12, searches=[search])
        self.assertEqual(train, [])
        self.assertEqual({(r["result_key"], r["status"]) for r in audit}, {("N", "train")})
        self.assertEqual(stats["no_answer"], 1)
        self.assertEqual([r["verdict"] for r in eval_rows], ["no_answer"])


if __name__ == "__main__":
    unittest.main()
