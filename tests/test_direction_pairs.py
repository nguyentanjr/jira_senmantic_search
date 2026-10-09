# -*- coding: utf-8 -*-
"""Kiem tra luat gan chieu va rao chan cua build_direction_pairs."""
import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pipeline"))
from build_direction_pairs import build, classify, judging_text, text_of, topic_of


def doc(key, summary, actual="", steps="", dense=None):
    return {"key": key, "summary": summary,
            "sections": {"actual": actual, "steps": steps},
            "text_dense": dense if dense is not None else f"{summary}\n{actual}"}


def args(**over):
    base = dict(min_jaccard=0.15, dup_jaccard=0.80, max_negatives=4,
                review_sample=10, topic_mode="none", min_sim=0.0,
                min_shared_rare=0, rare_idf=4.0)
    base.update(over)
    return argparse.Namespace(**base)


class ClassifyTest(unittest.TestCase):
    def test_tieu_de_quyet_dinh_khi_than_bai_im_lang(self):
        d = doc("RO-1", "[CMS] Vẫn đang cho phép chỉnh sửa trường Currency",
                actual="Kiểm tra trường Currency và Country màn edit request")
        self.assertEqual(classify(judging_text(d), d["summary"]),
                         [("button_state", "enable_wrong")])

    def test_tieu_de_co_chieu_nhung_than_bai_noi_nguoc_thi_bo(self):
        # Chon do CHINH XAC hon do phu: "can disable" trong actual thuong la
        # ky vong chu khong phai hien trang, nhung phan biet duoc hai thu do
        # bang luat thi khong dang tin, nen bo han cho an toan.
        d = doc("RO-1b", "[CMS] Vẫn đang cho phép chỉnh sửa trường Currency",
                actual="Cần đang disable trường này")
        self.assertEqual(classify(judging_text(d), d["summary"]), [])

    def test_hai_nguon_mau_thuan_thi_bo(self):
        d = doc("RO-2", "[USSD] Không gửi sms cho checker",
                actual="Thực tế gửi double sms cho checker")
        self.assertEqual(classify(judging_text(d), d["summary"]), [])

    def test_mot_nguon_cham_ca_hai_ben_thi_bo(self):
        d = doc("RO-3", "[USSD] Lỗi sms", actual="Không gửi sms, sau đó gửi double")
        self.assertEqual(classify(judging_text(d), d["summary"]), [])

    def test_steps_khong_duoc_quyet_dinh_chieu(self):
        d = doc("RO-4", "[CMS] Lỗi nút", steps="Chọn nút đang disable rồi kiểm tra")
        self.assertEqual(classify(judging_text(d), d["summary"]), [])

    def test_than_bai_dung_khi_tieu_de_im_lang(self):
        d = doc("RO-5", "[USSD] Lỗi thông báo", actual="Hệ thống gửi double sms")
        self.assertEqual(classify(judging_text(d), d["summary"]),
                         [("send_count", "extra")])


class TextTest(unittest.TestCase):
    def test_train_dung_text_dense(self):
        self.assertEqual(text_of(doc("RO-6", "S", "A", dense="VAN BAN INDEX")),
                         "VAN BAN INDEX")

    def test_topic_lay_the_dau_tien(self):
        self.assertEqual(topic_of({"summary": "[CMS Merchant][request delete] x"}),
                         "cms merchant")


class BuildTest(unittest.TestCase):
    def corpus(self):
        # Du gan de qua nguong jaccard toi thieu, du khac de khong bi coi la
        # ban trung (moi issue them mot cum tu rieng).
        common = "giao dịch thanh toán merchant thành công trên kênh ussd"
        return [
            doc("RO-10", f"[USSD] Không gửi sms {common} mã dịch vụ cashout hạn mức ngày"),
            doc("RO-11", f"[USSD] Chưa gửi sms {common} cho sender tài khoản đại lý cấp hai"),
            doc("RO-12", f"[USSD] Gửi double sms {common} bản ghi lịch sử trùng nhau"),
            doc("RO-13", f"[USSD] Gửi thừa sms {common} cho checker luồng phê duyệt bổ sung"),
        ]

    def test_sinh_duoc_cap_va_negative_nguoc_chieu(self):
        pairs, review, _ = build(self.corpus(), set(), args())
        self.assertTrue(pairs)
        missing = {p["key"] for p in pairs if p["side"] == "missing"}
        self.assertEqual(missing, {"RO-10", "RO-11"})
        for p in pairs:
            if p["side"] == "missing":
                self.assertTrue(set(p["neg_keys"]) <= {"RO-12", "RO-13"})
                self.assertIn(p["positive_key"], {"RO-10", "RO-11"})
        self.assertEqual(len(review), min(len(pairs), 10))

    def test_khong_lay_issue_cua_bo_do(self):
        pairs, _, stats = build(self.corpus(), {"RO-10", "RO-12"}, args())
        self.assertNotIn("RO-10", {p["key"] for p in pairs})
        for p in pairs:
            self.assertNotIn("RO-12", p["neg_keys"])
        self.assertEqual(stats["bo_vi_nam_trong_bo_do"], 2)

    def test_cap_gan_trung_bi_loai(self):
        same = "[USSD] Không gửi sms giao dịch thanh toán merchant thành công"
        docs = [doc("RO-20", same), doc("RO-21", same),
                doc("RO-22", "[USSD] Gửi double sms giao dịch thanh toán merchant")]
        pairs, _, _ = build(docs, set(), args(dup_jaccard=0.80))
        self.assertEqual(pairs, [])   # positive duy nhat la ban trung -> bo


class NegationTest(unittest.TestCase):
    """Cum tu chi chieu co the nam TRONG mot cum phu dinh nghia nguoc lai."""

    def test_khong_co_quyen_khong_bi_hieu_thanh_co_quyen(self):
        # "co quyen" nam trong "khong co quyen": truoc khi sua, issue nay bi gan
        # chieu "co quyen ma bi chan" - nguoc han. Gio khong con nhan sai; cum
        # "khong co quyen" DUNG MOT MINH van mo ho (RO-21141 "bao loi khong co
        # quyen approve" lai la chieu bi chan oan) nen de trong la dung.
        d = doc("RO-13385", "[CMS] Create Merchant thành công với role không có quyền",
                actual="Create merchant thành công")
        self.assertEqual(classify(judging_text(d), d["summary"]), [])

    def test_khong_co_quyen_van_lam_duoc_thi_nhan_dung(self):
        d = doc("RO-12603", "[FE] Tài khoản không có quyền vẫn truy cập được menu",
                actual="Vẫn hiển thị menu chức năng")
        self.assertEqual(classify(judging_text(d), d["summary"]),
                         [("permission", "not_enforced")])

    def test_co_quyen_that_su_van_nhan_dung(self):
        d = doc("RO-14296", "[CMS] Các role có quyền thực hiện không vào được menu",
                actual="")
        self.assertEqual(classify(judging_text(d), d["summary"]),
                         [("permission", "blocked")])


if __name__ == "__main__":
    unittest.main()
