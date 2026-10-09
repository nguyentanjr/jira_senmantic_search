from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feedback_state import replay


def ev(verdict, key="", **kw):
    return {"search_id": "s", "verdict": verdict, "result_key": key, **kw}


def run(events):
    return replay(events, lambda e: e["search_id"])


class ReplayTest(unittest.TestCase):
    def test_switch_and_clear(self):
        labels, _, seen, stats = run([
            ev("related", "a"), ev("not_related", "a", reason="wrong_module"),
            ev("related", "b"), ev("cleared", "b"),
        ])
        self.assertEqual(labels[("s", "A")]["verdict"], "not_related")
        self.assertNotIn(("s", "B"), labels)
        self.assertEqual(seen[("s", "A")], {"related", "not_related"})
        self.assertEqual(stats["labels_cleared"], 1)

    def test_no_answer_drops_only_related(self):
        labels, no_answer, _, _ = run([
            ev("related", "a"), ev("not_related", "b", reason="other"), ev("no_answer"),
        ])
        self.assertEqual(set(labels), {("s", "B")})
        self.assertIn("s", no_answer)

    def test_related_cancels_no_answer_and_cleared_removes_it(self):
        _, no_answer, _, stats = run([ev("no_answer"), ev("related", "a")])
        self.assertEqual(no_answer, {})
        self.assertEqual(stats["no_answer_cancelled_by_related"], 1)
        _, no_answer, _, _ = run([ev("no_answer"), ev("cleared")])
        self.assertEqual(no_answer, {})

    def test_clearing_unknown_label_is_harmless(self):
        labels, _, _, stats = run([ev("cleared", "x")])
        self.assertEqual(labels, {})
        self.assertEqual(stats["labels_cleared"], 0)


if __name__ == "__main__":
    unittest.main()
