from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))

from pipeline.eval_contrastive import score_rankings
from pipeline.validate_contrastive import training_keys, validate


class ContrastiveTest(unittest.TestCase):
    def test_pairwise_requires_preferred_candidate_to_be_visible(self):
        base = {"family_id": "f", "group_id": "g", "flow": "text",
                "source_key": "", "preferred": "A", "contrast": "B"}
        summary, details = score_rankings([
            {**base, "direction": "A", "ranked_keys": ["A", "B"]},
            {**base, "direction": "B", "preferred": "B", "contrast": "A",
             "ranked_keys": ["X"]},
        ])
        self.assertEqual(summary["pairwise_accuracy"], 0.5)
        self.assertEqual(summary["two_way_group_accuracy"], 0.0)
        self.assertFalse(details[1]["pairwise_correct"])

    def test_family_cannot_leak_across_splits(self):
        rows = [
            {"family_id": "f", "group_id": "g1", "split": "dev",
             "issue_a": "A", "issue_b": "B", "source_a": "", "source_b": ""},
            {"family_id": "f", "group_id": "g2", "split": "test",
             "issue_a": "A", "issue_b": "B", "source_a": "", "source_b": ""},
        ]
        self.assertTrue(any("nhieu split" in e for e in validate(rows, {"A", "B"})))

    def _row(self, **kw):
        base = {"family_id": "f", "group_id": "g", "split": "dev", "issue_a": "A",
                "issue_b": "B", "source_a": "SA", "source_b": "SB"}
        return {**base, **kw}

    def test_rejects_issue_seen_in_training_pairs(self):
        corpus = {"A", "B", "SA", "SB"}
        errors = validate([self._row()], corpus, train_keys={"SB"})
        self.assertTrue(any("du lieu train" in e and "SB" in e for e in errors))
        self.assertEqual(validate([self._row()], corpus, train_keys=set()), [])

    def test_rejects_duplicate_issues_inside_group(self):
        corpus = {"A", "B", "SA", "SB"}
        docs = {"A": {"summary": "a"}, "SA": {"summary": "sa"},
                "B": {"summary": "Menu  hien thi"}, "SB": {"summary": "menu hien thi"}}
        self.assertTrue(any("trung lap" in e for e in validate([self._row()], corpus, docs=docs)))
        docs["SB"] = {"summary": "khac", "dup_of": "B"}
        self.assertTrue(any("trung lap" in e for e in validate([self._row()], corpus, docs=docs)))
        self.assertTrue(any("lap lai" in e for e in validate([self._row(source_b="B")], corpus)))

    def test_same_issue_cannot_be_in_dev_and_test(self):
        rows = [self._row(), self._row(family_id="f2", group_id="g2", split="test",
                                       issue_a="X", source_a="", source_b="")]
        errors = validate(rows, {"A", "B", "SA", "SB", "X"})
        self.assertTrue(any("B xuat hien" in e for e in errors))

    def test_training_keys_reads_keys_and_negatives(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"key": "ro-1", "neg_keys": ["RO-2"]}) + "\n")
                f.write(json.dumps({"key": "RO-3", "neg_keys": "['ro-4']"}) + "\n")
            self.assertEqual(training_keys([path]), {"RO-1", "RO-2", "RO-3", "RO-4"})


if __name__ == "__main__":
    unittest.main()
