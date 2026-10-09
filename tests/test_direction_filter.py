# -*- coding: utf-8 -*-
"""Kiem tra bo ha diem nguoc chieu o buoc xep hang."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pipeline"))
from direction_filter import demote, doc_labels, is_opposite, query_labels

LABELS = {
    "RO-1": {("send_count", "missing")},
    "RO-2": {("send_count", "extra")},
    "RO-3": set(),
    "RO-4": {("button_state", "enable_wrong")},
}


class LabelTest(unittest.TestCase):
    def test_gan_nhan_theo_tieu_de(self):
        docs = [{"key": "RO-9", "summary": "[USSD] Không gửi sms cho checker",
                 "sections": {"actual": ""}}]
        self.assertEqual(doc_labels(docs), {"RO-9": {("send_count", "missing")}})

    def test_truy_van_tu_do(self):
        self.assertEqual(query_labels("giao dịch xong mà không gửi sms cho khách"),
                         {("send_count", "missing")})

    def test_nguoc_chieu_chi_tinh_cung_truc(self):
        q = {("send_count", "missing")}
        self.assertTrue(is_opposite(q, {("send_count", "extra")}))
        self.assertFalse(is_opposite(q, {("send_count", "missing")}))
        self.assertFalse(is_opposite(q, {("button_state", "enable_wrong")}))


class DemoteTest(unittest.TestCase):
    def test_day_ung_vien_nguoc_chieu_xuong_cuoi_window(self):
        out = demote("không gửi sms", ["RO-2", "RO-1", "RO-3"], LABELS, window=3)
        self.assertEqual(out, ["RO-1", "RO-3", "RO-2"])

    def test_khong_dung_den_phan_ngoai_window(self):
        ranked = ["RO-2", "RO-1", "RO-2", "RO-3"]
        out = demote("không gửi sms", ranked, LABELS, window=2)
        self.assertEqual(out, ["RO-1", "RO-2", "RO-2", "RO-3"])

    def test_truy_van_khong_ro_chieu_thi_giu_nguyen(self):
        ranked = ["RO-2", "RO-1"]
        self.assertEqual(demote("lỗi màn hình", ranked, LABELS), ranked)

    def test_ca_window_deu_nguoc_chieu_thi_giu_nguyen(self):
        # Day tat ca xuong nhu nhau la vo nghia, va de lo bo phan loai sai.
        ranked = ["RO-2", "RO-2"]
        self.assertEqual(demote("không gửi sms", ranked, LABELS, window=2), ranked)

    def test_luong_issue_dung_nhan_cua_issue_nguon(self):
        out = demote("", ["RO-2", "RO-1"], LABELS, window=2,
                     source_labels={("send_count", "missing")})
        self.assertEqual(out, ["RO-1", "RO-2"])


if __name__ == "__main__":
    unittest.main()


class NegationWindowTest(unittest.TestCase):
    def test_khong_bi_yeu_cau_otp_la_thieu_otp_khong_phai_thua(self):
        from build_direction_pairs import contains
        q = "nhập sai pin nhiều lần khi đăng nhập nhưng không bị yêu cầu xác thực otp"
        self.assertFalse(contains(q, "yêu cầu nhập otp"))
        self.assertFalse(contains(q, "yêu cầu xác thực"))
