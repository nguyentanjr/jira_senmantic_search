# -*- coding: utf-8 -*-
"""Test ham bo dau -- khong nap model.

    python -m unittest discover -s tests -p "test_eval_undiacritized.py" -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from eval_undiacritized import strip_diacritics


class StripDiacriticsTest(unittest.TestCase):
    def test_vietnamese_sentence(self):
        self.assertEqual(strip_diacritics("không đăng nhập được sau khi đổi mật khẩu"),
                         "khong dang nhap duoc sau khi doi mat khau")

    def test_uppercase_d_and_mixed_codes_kept(self):
        self.assertEqual(strip_diacritics("Đổi PIN lỗi ER10005 /api/v1"),
                         "Doi PIN loi ER10005 /api/v1")

    def test_ascii_unchanged_and_none_safe(self):
        self.assertEqual(strip_diacritics("Session Expired"), "Session Expired")
        self.assertEqual(strip_diacritics(None), "")


if __name__ == "__main__":
    unittest.main()
