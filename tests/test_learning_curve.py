# -*- coding: utf-8 -*-
"""Test cho duong cong du lieu: chon tap issue (train_task_models.py) va nguong
(learning_curve_eval.py) -- khong train, khong nap model.

    python -m unittest discover -s tests -p "test_learning_curve.py" -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.learning_curve_eval import model_info, threshold
from pipeline.train_task_models import subset_by_issue


def _rows(n_issues=40):
    rows = []
    for i in range(n_issues):
        for kind in ("title", "actual"):
            rows.append({"key": f"RO-{i}", "kind": kind, "query": f"q{i}{kind}"})
    return rows


class SubsetTest(unittest.TestCase):
    def test_whole_issues_nested_and_deterministic(self):
        rows = _rows()
        s10, s25, s50 = (subset_by_issue(rows, f, 13) for f in (0.1, 0.25, 0.5))
        keys = [{r["key"] for r in s} for s in (s10, s25, s50)]
        self.assertEqual([len(k) for k in keys], [math.ceil(40 * f) for f in (0.1, 0.25, 0.5)])
        self.assertTrue(keys[0] <= keys[1] <= keys[2])              # tap long nhau
        for s, k in zip((s10, s25, s50), keys):
            self.assertEqual(len(s), 2 * len(k))                   # giu du dong moi issue
        self.assertEqual(subset_by_issue(rows, 0.25, 13), s25)
        self.assertNotEqual({r["key"] for r in subset_by_issue(rows, 0.25, 99)}, keys[1])

    def test_full_fraction_is_unchanged_and_bad_input_rejected(self):
        rows = _rows(5)
        self.assertIs(subset_by_issue(rows, 1.0, 13), rows)
        with self.assertRaises(ValueError):
            subset_by_issue(rows, 0.0, 13)
        with self.assertRaises(ValueError):
            subset_by_issue([{"query": "x"}], 0.5, 13)


class ThresholdTest(unittest.TestCase):
    def test_smallest_fraction_with_all_larger_passing(self):
        passes = {0.0: [False], 0.1: [True, False], 0.25: [True, True],
                  0.5: [True, True], 1.0: [True, True]}
        self.assertEqual(threshold(passes), 0.25)

    def test_gap_in_the_middle_counts_from_the_top(self):
        passes = {0.1: [True, True], 0.25: [True, False], 0.5: [True, True], 1.0: [True, True]}
        self.assertEqual(threshold(passes), 0.5)

    def test_undefined_when_full_data_seed_fails(self):
        self.assertIsNone(threshold({0.5: [True, True], 1.0: [True, False]}))
        self.assertEqual(threshold({0.0: [True], 1.0: [True, True]}), 0.0)


class ModelInfoTest(unittest.TestCase):
    def test_reads_meta_or_treats_as_base(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "task_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"fraction": 0.25, "seed": 14, "frac_seed": 13, "n_train_rows": 10}, f)
            m = model_info("f025-s14", d)
            self.assertEqual((m["fraction"], m["seed"]), (0.25, 14))
            old = os.path.join(d, "old")
            os.makedirs(old)
            with open(os.path.join(old, "task_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"seed": 13}, f)                          # model v2 cu: khong ghi fraction
            self.assertEqual(model_info("f100-s13", old)["fraction"], 1.0)
        self.assertEqual(model_info("f000", "intfloat/multilingual-e5-base")["fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
