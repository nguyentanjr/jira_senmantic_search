import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from probe_relation_adapter import connected_groups, fit_adapter, FrozenDense, metrics, judged_ndcg


class RelationAdapterProbeTests(unittest.TestCase):
    def test_connected_split_includes_candidate_bridges_and_old_triples(self):
        triples = [dict(key='a', positive_key='b', neg_keys=['c']),
                   dict(key='d', positive_key='e', neg_keys=['f'])]
        labels = [dict(query_key='c', candidate_key='e'),
                  dict(query_key='x', candidate_key='y')]
        groups = connected_groups(triples, labels)
        self.assertEqual(len({groups[k] for k in 'abcdef'}), 1)
        self.assertNotEqual(groups['a'], groups['x'])

    def test_fitting_an_explicit_preference_increases_its_margin(self):
        emb = np.array([[1., 0.], [0., 1.], [.8, .6]], dtype=np.float32)
        pref = [dict(query_key='q', positive_key='p', negative_key='n')]
        index = {'q': 0, 'p': 1, 'n': 2}
        w = fit_adapter(emb, index, pref)
        d = FrozenDense(emb, ['q', 'p', 'n'])
        before = d.score(emb[0])
        d.set_adapter(w)
        after = d.score(emb[0])
        self.assertGreater(after[1] - after[2], before[1] - before[2])
        # Duplicating identical comparisons must not increase an anchor's weight.
        np.testing.assert_allclose(w, fit_adapter(emb, index, pref * 3), atol=1e-12)

    def test_empty_train_fold_fails_instead_of_using_heldout_labels(self):
        with self.assertRaises(ValueError):
            fit_adapter(np.eye(2), {}, [])

    def test_unjudged_is_not_negative_and_no_positive_is_not_zero_recall(self):
        result = metrics(['unjudged', 'negative', 'positive'], {'negative': 0, 'positive': 1})
        self.assertEqual(result['known_zero_count10'], 1)
        self.assertAlmostEqual(result['judged_fraction10'], 2/3)
        self.assertEqual(result['known_recall10'], 1)
        self.assertIsNone(metrics(['negative'], {'negative': 0})['known_recall10'])

    def test_graded_ndcg_preserves_related_grade_one(self):
        labels = {'same': 2, 'related': 1, 'wrong': 0}
        self.assertEqual(judged_ndcg(['same', 'related', 'wrong'], labels), 1.)
        self.assertLess(judged_ndcg(['same', 'wrong', 'related'], labels), 1.)
        self.assertIsNone(judged_ndcg(['wrong'], {'wrong': 0}))


if __name__ == '__main__':
    unittest.main()
