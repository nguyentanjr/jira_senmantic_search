# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.retrieval import SearchConfig, SearchEngine
from pilot.service import PilotError, PilotSearchService, filter_doc_ids, matched_snippet


DOCS = [
    {"key": "RO-1", "summary": "Login error", "text_bm25": "login password error",
     "text_dense": "login error", "text_business": "Actual: không đăng nhập được",
     "status": "Open", "priority": "High", "components": ["Auth"],
     "facets": {}, "created": "2026-01-10", "text_comments": ""},
    {"key": "RO-2", "summary": "Payment", "text_bm25": "payment timeout",
     "text_dense": "payment timeout", "text_business": "Actual: thanh toán timeout",
     "status": "Closed", "priority": "Medium", "components": ["Payment"],
     "facets": {}, "created": "2026-02-10", "text_comments": ""},
    {"key": "ABC-3", "summary": "Unrelated", "text_bm25": "nothing here",
     "text_dense": "nothing", "text_business": "Không có dữ liệu",
     "status": "Open", "priority": "Low", "components": [],
     "facets": {"team": "Core"}, "created": "2025-12-10", "text_comments": ""},
]


class RetrievalPrefilterTest(unittest.TestCase):
    def test_candidate_is_drawn_from_allowed_set(self):
        engine = SearchEngine(DOCS)
        engine.build_bm25()
        cfg = SearchConfig(use_bm25=True, use_dense=False, candidate_k=1, top_k=1)
        ranked, scores = engine.search("nothing", cfg, allowed={2}, return_scores=True)
        self.assertEqual(ranked.tolist(), [2])
        self.assertEqual(len(scores), 1)

    def test_empty_allowed_returns_empty(self):
        engine = SearchEngine(DOCS)
        engine.build_bm25()
        cfg = SearchConfig(use_bm25=True, use_dense=False, top_k=10)
        self.assertEqual(engine.search("login", cfg, allowed=set()).tolist(), [])


class PilotHelpersTest(unittest.TestCase):
    def test_combined_metadata_filters(self):
        ids = filter_doc_ids(DOCS, {"project": "ro", "component": "payment",
                                    "status": "closed", "created_from": "2026-02-01"})
        self.assertEqual(ids, {1})

    def test_team_is_a_service_component(self):
        self.assertEqual(filter_doc_ids(DOCS, {"component": "core"}), {2})

    def test_team_group_does_not_duplicate_full_team(self):
        doc = {**DOCS[0], "facets": {"team": "VDS | Nạp rút", "team_group": "Nạp rút"}}
        self.assertEqual(filter_doc_ids([doc], {"component": "VDS | Nạp rút"}), {0})
        self.assertEqual(filter_doc_ids([doc], {"component": "Nạp rút"}), set())

    def test_date_validation(self):
        with self.assertRaises(PilotError):
            filter_doc_ids(DOCS, {"created_from": "01/02/2026"})
        with self.assertRaises(PilotError):
            filter_doc_ids(DOCS, {"created_from": "2026-03-01",
                                  "created_to": "2026-02-01"})

    def test_matched_snippet_prefers_matching_line(self):
        doc = dict(DOCS[0])
        doc["text_business"] = "Bước: mở ứng dụng\nThực tế: không đăng nhập được sau đổi mật khẩu"
        self.assertIn("đăng nhập", matched_snippet(doc, "lỗi đăng nhập"))


class PilotServiceLightTest(unittest.TestCase):
    def test_compare_returns_three_modes_without_writing_logs(self):
        class Engine:
            dense = type("Dense", (), {"_model": object()})()

            def search(self, query, cfg, timing=None, return_scores=False):
                if timing is not None:
                    timing.embed_ms.append(1.0 if cfg.use_dense else 0.0)
                    timing.search_ms.append(2.0)
                ranked = np.array([0, 1, 2])
                return (ranked[:cfg.top_k], np.array([3.0, 2.0, 1.0])) if return_scores else ranked

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(x) + "\n" for x in DOCS), encoding="utf-8")
            searches, feedback = root / "searches.jsonl", root / "feedback.jsonl"
            service = PilotSearchService(corpus, feedback, search_log_path=searches)
            service._engine = lambda flow: Engine()  # type: ignore[method-assign]
            response = service.compare({"query": "login error", "top_k": 2})
            self.assertEqual(set(response["results"]), {"keyword", "semantic", "hybrid"})
            self.assertTrue(all(len(items) <= 2 for items in response["results"].values()))
            self.assertFalse(searches.exists())
            self.assertFalse(feedback.exists())

    def test_metadata_suggestions_and_feedback_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in DOCS),
                              encoding="utf-8")
            feedback = root / "feedback.jsonl"
            searches = root / "searches.jsonl"
            service = PilotSearchService(corpus, feedback, search_log_path=searches)
            self.assertEqual(service.metadata()["corpus_size"], 3)
            self.assertEqual(service.issue_suggestions("RO-2")[0]["key"], "RO-2")
            response = service._log_search({
                "eligible_docs": 3, "config": "hybrid+rerank", "model": "test",
                "rerank_k": 30,
                "latency": {"embed_ms": 1, "search_ms": 2, "rerank_ms": 3,
                            "total_ms": 6},
                "results": [{"key": "RO-2", "rank": 1, "ranking_score": 0.5}],
            }, "text", "timeout password=secret", "", {"project": "RO"}, 10)
            saved = service.record_feedback({"search_id": response["search_id"],
                                             "result_key": "RO-2", "verdict": "related"})
            self.assertTrue(saved["saved"])
            event = json.loads(feedback.read_text(encoding="utf-8"))
            self.assertEqual((event["result_key"], event["jira_writeback"]),
                             ("RO-2", "pending"))
            search = json.loads(searches.read_text(encoding="utf-8"))
            self.assertEqual(search["results"][0]["key"], "RO-2")
            self.assertEqual(search["filters"], {"project": "RO"})
            self.assertNotIn("secret", search["query"])

    def test_feedback_requires_reason_and_supports_no_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(x) + "\n" for x in DOCS), encoding="utf-8")
            service = PilotSearchService(corpus, root / "feedback.jsonl",
                                         search_log_path=root / "searches.jsonl")
            response = service._log_search({
                "eligible_docs": 3, "config": "hybrid", "model": "test", "rerank_k": 0,
                "latency": {"embed_ms": 1, "search_ms": 2, "rerank_ms": 0,
                            "total_ms": 3},
                "results": [{"key": "RO-2", "rank": 1, "ranking_score": 0.5}],
            }, "text", "timeout", "", {}, 10)
            with self.assertRaises(PilotError):
                service.record_feedback({"search_id": response["search_id"],
                                         "result_key": "RO-2", "verdict": "not_related"})
            service.record_feedback({"search_id": response["search_id"],
                                     "result_key": "RO-2", "verdict": "not_related",
                                     "reason": "wrong_module"})
            service.record_feedback({"search_id": response["search_id"],
                                     "result_key": "", "verdict": "no_answer"})
            rows = [json.loads(x) for x in (root / "feedback.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["verdict"], "no_answer")
            # Bo chon ket qua va bo "khong co dap an" deu duoc ghi thanh su kien.
            service.record_feedback({"search_id": response["search_id"],
                                     "result_key": "RO-2", "verdict": "cleared"})
            service.record_feedback({"search_id": response["search_id"],
                                     "result_key": "", "verdict": "cleared"})
            with self.assertRaises(PilotError):
                service.record_feedback({"search_id": response["search_id"],
                                         "result_key": "RO-3", "verdict": "cleared"})
            rows = [json.loads(x) for x in (root / "feedback.jsonl").read_text().splitlines()]
            self.assertEqual([(r["verdict"], r["result_key"], r["rank"]) for r in rows[-2:]],
                             [("cleared", "RO-2", 1), ("cleared", "", 0)])

    def test_disabled_feedback_is_logged_without_review_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(x) + "\n" for x in DOCS), encoding="utf-8")
            service = PilotSearchService(corpus, root / "feedback.jsonl",
                                         search_log_path=root / "searches.jsonl")
            base = {"eligible_docs": 3, "config": "hybrid", "model": "test", "rerank_k": 0,
                    "latency": {"embed_ms": 1, "search_ms": 2, "rerank_ms": 0, "total_ms": 3},
                    "results": [{"key": "RO-2", "rank": 1, "ranking_score": 0.5}]}
            # Luot dau tien binh thuong se la full_top5; khi tat danh gia thi khong.
            off = service._log_search(dict(base), "text", "q", "", {}, 10, False)
            on = service._log_search(dict(base), "text", "q", "", {}, 10)
            self.assertEqual(off["review_mode"], "disabled")
            self.assertEqual(off["review_required_ranks"], [])
            self.assertEqual(on["review_mode"], "standard")


if __name__ == "__main__":
    unittest.main()
