"""Candidate membership contracts; no models or relevance labels required."""
import unittest

import numpy as np

from pipeline.retrieval import SearchConfig, SearchEngine, top_k_allowed


class FixedDense:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def encode(self, *args, **kwargs):
        return np.zeros((1, 1), dtype=np.float32)

    def score(self, query):
        return self.scores


class CandidateEvidenceTest(unittest.TestCase):
    def engine(self, texts, scores=None):
        engine = SearchEngine(
            [{"key": str(i), "text_bm25": t} for i, t in enumerate(texts)],
            dense=FixedDense(scores) if scores is not None else None,
            bm25_fold="auto")
        engine.build_bm25()
        return engine

    def test_oov_keyword_is_empty_across_sizes_and_forms(self):
        for n in (1, 7, 101, 237):
            eng = self.engine(["đăng nhập timeout ER80011"] * n)
            cfg = SearchConfig(use_dense=False)
            for query in ("ZXQ987654321", "", "...", "zzqv_not_in_corpus"):
                with self.subTest(n=n, query=query):
                    ids, scores = eng.search(query, cfg, return_scores=True)
                    self.assertEqual(ids.tolist(), [])
                    self.assertEqual(scores.tolist(), [])

    def test_sparse_matches_are_not_padded(self):
        for query in ("ER80011", "đăng nhập", "dang nhap", "timeout"):
            eng = self.engine(["unrelated"] * 40 + ["đăng nhập ER80011 timeout"])
            self.assertEqual(eng.search(query, SearchConfig(use_dense=False)).tolist(), [40])

    def test_empty_lexical_branch_cannot_override_dense(self):
        rng = np.random.default_rng(17)
        for n in (7, 101, 237):
            scores = rng.permutation(n).astype(float) - n  # negative cosine remains valid
            eng = self.engine(["login"] * n, scores)
            for allowed in (None, {0, 2, 5}, set()):
                hybrid = eng.search("ZXQ987654321", SearchConfig(), allowed=allowed)
                dense = eng.search("ZXQ987654321", SearchConfig(use_bm25=False), allowed=allowed)
                self.assertEqual(hybrid.tolist(), dense.tolist())

    def test_filter_with_no_lexical_match_returns_empty(self):
        eng = self.engine(["login", "payment"])
        cfg = SearchConfig(use_dense=False, candidate_k=1)
        self.assertEqual(eng.search("login", cfg, allowed={1}).tolist(), [])
        self.assertEqual(eng.search("login", cfg, exclude={0}).tolist(), [])

    def test_fusion_union_is_not_padded(self):
        eng = self.engine(["login"] * 7)
        cfg = SearchConfig(use_bm25=False, use_dense=False)
        self.assertEqual(eng.search("anything", cfg).tolist(), [])

    def test_positive_keyword_order_is_unchanged(self):
        eng = self.engine(["login " * (i + 1) for i in range(120)])
        expected = top_k_allowed(eng.bm25_scores("login"), 100)[:10]
        self.assertEqual(eng.search("login", SearchConfig(use_dense=False)).tolist(),
                         expected.tolist())

    def test_exclude_before_dense_candidate_boundary(self):
        eng = self.engine(["unrelated"] * 4, [4, 3, 2, 1])
        cfg = SearchConfig(use_bm25=False, candidate_k=2, top_k=2)
        self.assertEqual(eng.search("x", cfg, exclude={0}).tolist(), [1, 2])
        self.assertEqual(eng.search("x", cfg, exclude={0}, allowed={0, 2, 3}).tolist(), [2, 3])

    def test_exclude_before_keyword_boundary_and_empty_eligibility(self):
        eng = self.engine(["login " * (i + 1) for i in range(4)])
        cfg = SearchConfig(use_dense=False, candidate_k=2, top_k=2)
        self.assertEqual(eng.search("login", cfg, exclude={3}).tolist(), [2, 1])
        self.assertEqual(eng.search("login", cfg, exclude={0, 1, 2, 3}).tolist(), [])

    def test_trace_is_observational_and_excludes_ineligible_documents(self):
        eng = self.engine(["login", "payment", "login login", "unrelated"], [4, 3, 2, 1])
        cfg = SearchConfig(candidate_k=2, top_k=1)
        trace = {"stale": 1}
        expected = eng.search("login", cfg, exclude={0}, allowed={0, 1, 2})
        result = eng.search("login", cfg, exclude={0}, allowed={0, 1, 2}, trace=trace)
        self.assertEqual(result.tolist(), expected.tolist())
        self.assertNotIn("stale", trace)
        self.assertEqual(trace["bm25"], [2])
        self.assertEqual(trace["dense"], [1])
        self.assertEqual(trace["union"], [1, 2])
        self.assertEqual(trace["final"], result.tolist())
        self.assertEqual(len(trace["fused"]), 2)

    def test_hybrid_backfill_preserves_existing_prefix(self):
        eng = self.engine(["login login", "login", "login login login", "unrelated"], [4, 3, 2, 1])
        cfg = SearchConfig(candidate_k=3, top_k=3)
        before = eng.search("login", cfg).tolist()
        expected_prefix = [i for i in before if i != 0]
        after, scores = eng.search("login", cfg, exclude={0}, return_scores=True)
        self.assertEqual(after[:len(expected_prefix)].tolist(), expected_prefix)
        self.assertEqual(set(after), {1, 2, 3})
        self.assertEqual(len(scores), len(after))


if __name__ == "__main__":
    unittest.main()
