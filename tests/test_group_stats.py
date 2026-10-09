# -*- coding: utf-8 -*-
"""Test cho pipeline/group_stats.py -- phan nhom truy van va thong ke theo nhom.

    python -m unittest discover -s tests -p "test_group_stats.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.group_stats import GROUPS, diff_of_means, group_report, has_signal


class SignalTest(unittest.TestCase):
    def test_codes_exceptions_api_count(self):
        self.assertTrue(has_signal("Thanh toán báo lỗi ER80011 thay vì ER10005"))
        self.assertTrue(has_signal("NullPointerException khi gọi dịch vụ"))
        self.assertTrue(has_signal("gọi /api/v1/payment/confirm bị lỗi"))

    def test_plain_text_and_jira_keys_do_not(self):
        self.assertFalse(has_signal("khong dang nhap duoc sau khi doi mat khau"))
        self.assertFalse(has_signal("lỗi giống RO-12345, hiển thị sai thông báo"))
        self.assertFalse(has_signal(""))


class ReportTest(unittest.TestCase):
    def test_deltas_and_gap_per_group(self):
        co, khong = GROUPS
        scores = {"keyword": {"m": np.array([1.0, 1.0, 0.0, 0.0])},
                  "x": {"m": np.array([1.0, 1.0, 1.0, 1.0])}}
        rep = group_report(scores, [co, co, khong, khong], "keyword", ["m"], n_boot=200)
        self.assertEqual(rep["n"], {co: 2, khong: 2})
        self.assertEqual(rep["vs_baseline"][co]["x"]["m"]["delta"], 0.0)
        self.assertEqual(rep["vs_baseline"][khong]["x"]["m"]["delta"], 1.0)
        # loi the so voi keyword lon hon o truy van tu nhien -> gap am
        self.assertEqual(rep["gap"]["x"]["m"]["delta"], -1.0)
        self.assertTrue(rep["gap"]["x"]["m"]["significant"])
        self.assertNotIn("keyword", rep["vs_baseline"][co])

    def test_missing_group_has_no_gap(self):
        co, _ = GROUPS
        scores = {"keyword": {"m": np.array([0.0, 1.0])}, "x": {"m": np.array([1.0, 1.0])}}
        rep = group_report(scores, [co, co], "keyword", ["m"], n_boot=100)
        self.assertEqual(rep["gap"], {})
        self.assertEqual(rep["n"][GROUPS[1]], 0)

    def test_diff_of_means_independent_groups(self):
        d = diff_of_means(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5]), n_boot=100)
        self.assertEqual((d["delta"], d["significant"]), (0.0, False))


if __name__ == "__main__":
    unittest.main()
