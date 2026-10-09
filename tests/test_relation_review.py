import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from build_relation_review import components, expand_pack, make_pack


class RelationReviewTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{"key": "A", "positive_key": "B", "neg_keys": ["C"],
                      "axis": "button", "side": "enable"},
                     {"key": "D", "positive_key": "C", "neg_keys": ["E"],
                      "axis": "button", "side": "disable"}]
        self.docs = {k: {"summary": k, "text_business": "body", "sections": {"expected": "expected"}}
                     for k in "ABCDE"}
        self.scores = [{"key": "A", "pos_key": "B", "cos_pos": .4, "cos_neg": {"C": .5}},
                       {"key": "D", "pos_key": "C", "cos_pos": .8, "cos_neg": {"E": .3}}]

    def test_entire_component_excluded_even_for_shared_negative(self):
        self.assertEqual(len(set(components(self.rows).values())), 1)
        reviews, _, stats = make_pack(self.rows, self.docs, self.scores, {"E"})
        self.assertEqual(reviews, [])
        self.assertEqual(stats["excluded_components"], 1)

    def test_blind_unknown_labels_and_no_training_output(self):
        reviews, provenance, stats = make_pack(self.rows, self.docs, self.scores, set())
        self.assertEqual(len(reviews), 4)
        self.assertEqual(stats["training_rows_emitted"], 0)
        self.assertEqual({r["review_id"] for r in reviews}, {r["review_id"] for r in provenance})
        for row in reviews:
            self.assertEqual(row["grade"], "")
            self.assertNotIn("proposed_role", row)
            self.assertNotIn("historical_cosine", row)
            self.assertIn("expected", row["query_context"])

    def test_score_mismatch_fails_and_order_reproducible(self):
        a = make_pack(self.rows, self.docs, self.scores, set())
        self.assertEqual(a, make_pack(self.rows, self.docs, self.scores, set()))
        self.scores[0]["cos_neg"] = {"Z": .5}
        with self.assertRaisesRegex(ValueError, "mismatched"):
            make_pack(self.rows, self.docs, self.scores, set())

    def test_expansion_excludes_existing_pairs_and_protected_documents(self):
        from retrieval import SearchEngine
        for k in "FG":
            self.docs[k] = {"summary": "new", "text_dense": "login", "text_bm25": "login"}
        self.docs["A"]["text_dense"] = "login"
        self.docs["D"]["text_dense"] = "login"
        import numpy as np
        class FakeDense:
            def encode(self, texts, is_query):
                return np.zeros((len(texts), 1))
            def score(self, vector):
                return np.arange(7, dtype=np.float32)
        engine = SearchEngine([dict(doc, key=k) for k, doc in self.docs.items()], dense=FakeDense())
        engine.build_bm25()
        reviews, sidecar, _ = make_pack(self.rows, self.docs, self.scores, set())
        stats = expand_pack(reviews, sidecar, self.rows, self.docs, engine, {"G"}, 1)
        self.assertEqual(stats["new_unjudged_candidates"], 2)
        self.assertEqual([r["candidate_key"] for r in reviews if r["candidate_key"] in {"F", "G"}], ["F", "F"])
        self.assertEqual({r["grade"] for r in reviews}, {""})
