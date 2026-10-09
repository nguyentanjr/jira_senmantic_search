# -*- coding: utf-8 -*-
"""Test diem CE offline luong issue -- khong nap model.

    python -m unittest discover -s tests -p "test_ce_offline.py" -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from ce_offline import OfflineCEStore, import_result, text_sha  # noqa: E402
from pipeline.search import plan  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "score_issue_ce_offline", ROOT / "colab/issue_ce_offline_bundle/score_issue_ce_offline.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

DOCS = [{"key": f"RO-{i}", "text_dense": f"van ban {i}" * (i + 1)} for i in range(5)]


def fake_score(pairs):
    # diem = do dai ung vien -> thu tu biet truoc, doc lap voi thu tu cham
    return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)


def make_bundle(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    with open(data / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in DOCS:
            t = d["text_dense"]
            f.write(json.dumps({"key": d["key"], "text": t, "text_sha": text_sha(t)}) + "\n")
    pools = np.array([[1, 2, 3], [0, 2, -1], [4, 3, 1], [0, 1, 2], [3, 2, 0]], dtype=np.int32)
    np.save(data / "pools.npy", pools)
    (data / "manifest.json").write_text(json.dumps({"n_queries": 5, "n_docs": 5}))
    return data


class ScoreAndImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = make_bundle(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, calls):
        def counted(pairs):
            calls.append(len(pairs))
            return fake_score(pairs)
        return scorer.run(self.data, self.root / "out", counted, {"model": "fake"},
                          shard_size=2, log=lambda s: None)

    def test_scores_match_pairs_and_resume_skips_done_shards(self):
        calls = []
        m = self._run(calls)
        self.assertEqual(m["scored_pairs"], 14)
        self.assertEqual(sum(calls), 14)
        s = np.load(self.root / "out/ce_scores.npy")
        self.assertTrue(np.isnan(s[1, 2]))
        self.assertEqual(s[0, 0], len(DOCS[1]["text_dense"]))
        self.assertEqual(s[2, 0], len(DOCS[4]["text_dense"]))
        again = []
        self._run(again)
        self.assertEqual(again, [])

    def test_shards_from_other_bundle_rejected(self):
        self._run([])
        np.save(self.data / "pools.npy", np.zeros((5, 3), dtype=np.int32))
        with self.assertRaises(RuntimeError):
            self._run([])

    def test_import_then_rerank(self):
        self._run([])
        import_result(self.root / "out", self.data, self.root / "store")
        store = OfflineCEStore.load(self.root / "store")
        self.assertEqual(store.bind(DOCS), {"valid": 5, "changed": 0, "new": 0})
        ranked, scores = store.rerank(0, np.array([1, 2, 3]))
        self.assertEqual(ranked.tolist(), [3, 2, 1])          # van ban dai hon -> diem cao hon
        self.assertTrue((np.diff(scores) <= 0).all())
        # ung vien ngoai pool da cham -> khong tron, giu hybrid
        self.assertIsNone(store.rerank(0, np.array([1, 4])))

    def test_stale_or_new_docs_fall_back(self):
        self._run([])
        import_result(self.root / "out", self.data, self.root / "store")
        store = OfflineCEStore.load(self.root / "store")
        docs = [dict(d) for d in DOCS]
        docs[2]["text_dense"] = "da sua"
        docs.append({"key": "RO-99", "text_dense": "moi"})
        self.assertEqual(store.bind(docs), {"valid": 4, "changed": 1, "new": 1})
        self.assertIsNone(store.rerank(0, np.array([1, 2, 3])))   # ung vien da sua
        self.assertIsNone(store.rerank(2, np.array([4, 3, 1])))   # truy van da sua
        self.assertIsNone(store.rerank(5, np.array([0])))         # truy van moi
        self.assertEqual(store.rerank(4, np.array([3, 0]))[0].tolist(), [3, 0])

    def test_import_rejects_mismatched_pools(self):
        self._run([])
        np.save(self.data / "pools.npy", np.ones((5, 3), dtype=np.int32))
        with self.assertRaises(SystemExit):
            import_result(self.root / "out", self.data, self.root / "store")


class RoutingTest(unittest.TestCase):
    def test_offline_only_for_issue_flow(self):
        p = plan(key="RO-11708", rerank="offline")
        self.assertEqual((p["flow"], p["config"], p["ce_offline"], p["measured"]),
                         ("issue", "hybrid", True, False))
        self.assertFalse(plan(key="RO-11708")["ce_offline"])
        with self.assertRaises(ValueError):
            plan(query="khong gui sms", rerank="offline")


if __name__ == "__main__":
    unittest.main()
