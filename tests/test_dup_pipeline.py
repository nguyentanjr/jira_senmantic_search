# -*- coding: utf-8 -*-
"""Test cho pipeline duplicate retrieval.

    python -m unittest discover -s tests -v

Khong test nao o day can tai model hay corpus -- chung kiem tra dung phan
LOGIC de vo nhat: chuan hoa mat toan tu, mau thuan bi bo qua, nguong bi
hard-code, config khong bat/tat duoc.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dup.config import DupConfig, ablation_presets, load_config
from pipeline.dup.decide import DecisionEngine
from pipeline.dup.extract import StructuredExtractor
from pipeline.dup.features import PairFeatureExtractor
from pipeline.dup.preprocess import Preprocessor
from pipeline.dup.scoring import HybridScorer, ScoredCandidate


def _pair(a_txt, b_txt, cfg=None, semantic=0.95, lexical=0.9):
    cfg = cfg or DupConfig()
    pre = Preprocessor(cfg.preprocess)
    ex = StructuredExtractor(cfg.extract, pre)
    fx = PairFeatureExtractor(cfg.scoring)
    sc = HybridScorer(cfg.scoring)
    f = fx.extract(ex.extract_query(a_txt, "A"), ex.extract_query(b_txt, "B"),
                   semantic=semantic, lexical=lexical)
    return f, sc.score(f)


# ---------------------------------------------------------------- phase 1
class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.pre = Preprocessor()

    def test_operator_survives_tokenization(self):
        """Loi cu: '=' va '<' bi tokenizer vut di -> hai test case thanh mot."""
        a = set(self.pre.tokenize("From Date = To Date"))
        b = set(self.pre.tokenize("From Date < To Date"))
        self.assertIn("op_eq", a)
        self.assertIn("op_lt", b)
        self.assertNotEqual(a, b)

    def test_numbers_are_distinct(self):
        a = set(self.pre.tokenize("nhap so tien = 0"))
        b = set(self.pre.tokenize("nhap so tien = 1"))
        self.assertIn("num_0", a)
        self.assertIn("num_1", b)
        self.assertNotEqual(a, b)

    def test_negation_kept(self):
        self.assertIn("neg_khong", self.pre.tokenize("khong hien thi"))
        self.assertNotIn("neg_khong", self.pre.tokenize("hien thi"))

    def test_enum_identity_kept(self):
        """TERMINATED va LOCKED khong duoc gop thanh mot token."""
        a = set(self.pre.tokenize("Agent = TERMINATED"))
        b = set(self.pre.tokenize("Agent = LOCKED"))
        self.assertIn("terminated", a)
        self.assertIn("locked", b)
        self.assertNotIn("locked", a)

    def test_identifier_not_shattered(self):
        toks = self.pre.tokenize("loi ER80011 tren /api/v1/mini-app")
        self.assertIn("er80011", toks)
        # dau "/" dan dau bi cat, nhung duong dan van la MOT token
        self.assertIn("api/v1/mini-app", toks)

    def test_arrows_are_not_operators(self):
        """"->" trong mo ta QA khong phai toan tu so sanh."""
        self.assertEqual(self.pre.operators("thuc hien X -> he thong bao loi"), [])
        self.assertEqual(self.pre.operators("chon A --> hien thi B"), [])
        self.assertEqual(self.pre.operators("a => b"), [])

    def test_xml_placeholder_is_not_operator(self):
        self.assertEqual(
            self.pre.operators("transaction ID <transactionId> is created"), [])

    def test_real_operators_still_read(self):
        self.assertEqual(self.pre.operators("so tien >= 1000"), ["op_ge"])
        self.assertEqual(self.pre.operators("From Date < To Date"), ["op_lt"])

    def test_negation_scope_is_bigrams(self):
        s = self.pre.negation_scopes("man hinh khong hien thi thong bao")
        self.assertIn("hien_thi", s)
        self.assertNotIn("man_hinh", s)      # nam TRUOC tu phu dinh

    def test_negation_scope_ignores_lone_syllable(self):
        """Mot am tiet roi khong phai vi ngu -- "du" trong "du lieu"."""
        s = self.pre.negation_scopes("man search khong hien thi du lieu")
        self.assertNotIn("du", s)
        self.assertIn("hien_thi", s)

    def test_numeric_context(self):
        self.assertEqual(self.pre.numeric_contexts("Nhap so tien = 0"),
                         {"tien": {"0"}})
        self.assertEqual(self.pre.numeric_contexts("sau 5 ngay")["sau"], {"5"})

    def test_hex_identifier_is_not_a_number(self):
        """traceId hex khong duoc bam thanh cac manh so."""
        ctx = self.pre.numeric_contexts('"traceId": "5bc6fe8424984f3d"')
        self.assertEqual(ctx, {})

    def test_identifier_context_is_ignored(self):
        """So sau traceId/transId la duy nhat moi issue -> khong so sanh."""
        self.assertEqual(
            self.pre.numeric_contexts("transaction ID 1356438234920685568"), {})

    def test_issue_ref_masked(self):
        self.assertNotIn("RO-10049", self.pre.mask_issue_refs("trung bug RO-10049"))


# ---------------------------------------------------------------- phase 2
class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.ex = StructuredExtractor()

    def test_tag_actor_and_channel(self):
        r = self.ex.extract_query("[USSD supper agent][check balance] loi")
        self.assertIn("ussd", r.channel)
        self.assertIn("agent", r.actor)
        self.assertIn("super", r.actor)

    def test_status_detected(self):
        r = self.ex.extract_query("Agent = TERMINATED khong giao dich duoc")
        self.assertIn("terminated", r.statuses)

    def test_provider_detected(self):
        r = self.ex.extract_query("Chuyen tien sang ngan hang CRDB loi")
        self.assertIn("crdb", r.provider)
        self.assertIn("crdb", r.bank)

    def test_unknown_stays_empty_not_guessed(self):
        """Khong co thong tin -> rong, KHONG duoc bia."""
        r = self.ex.extract_query("loi la")
        self.assertEqual(r.statuses, set())
        self.assertEqual(r.provider, set())
        self.assertIsNone(r.expected_behavior)


# ---------------------------------------------------------------- phase 4
class TestContradictions(unittest.TestCase):
    """Cac cap trong yeu cau: van ban rat giong nhung KHAC hanh vi."""

    CASES = [
        ("From Date = To Date", "From Date < To Date", "operator_conflict"),
        ("Agent = TERMINATED loi", "Agent = LOCKED loi", "status_conflict"),
        ("Bam Save man hinh kenh", "Bam Submit man hinh kenh", "action_conflict"),
        ("so tien = 0 bao loi", "so tien = 1 bao loi", "numeric_conflict"),
        ("man hinh hien thi thong bao", "man hinh khong hien thi thong bao",
         "negation_conflict"),
        ("thieu truong requestTime bao loi", "thieu truong accessChannelID bao loi",
         "entity_conflict"),
        ("[USSD EU][Show balance] loi", "[USSD agent][check balance] loi",
         "actor_conflict"),
    ]

    def test_each_contradiction_is_detected(self):
        for a, b, want in self.CASES:
            with self.subTest(signal=want):
                f, _ = _pair(a, b)
                self.assertGreater(f.get(want), 0, f"khong bat duoc {want}")

    def test_contradiction_blocks_duplicate_label(self):
        """Diem semantic cao + mau thuan manh KHONG duoc ra 'duplicate'."""
        eng = DecisionEngine()
        for a, b, want in self.CASES:
            with self.subTest(signal=want):
                f, bd = _pair(a, b, semantic=0.99, lexical=0.99)
                sc = ScoredCandidate(idx=0, key="B", features=f, breakdown=bd,
                                     final_score=bd.final)
                self.assertNotEqual(eng.label(sc), "duplicate",
                                    f"{want}: bi gan duplicate ({bd.final})")

    def test_paraphrase_is_not_penalised(self):
        """Cap dien dat khac nhau -- muc tieu chinh cua de tai -- khong bi phat."""
        f, bd = _pair("nguoi dung khong dang nhap duoc sau khi doi mat khau",
                      "tai khoan bi tu choi xac thuc sau khi cap nhat mat khau",
                      semantic=0.9, lexical=0.5)
        self.assertEqual(f.get("negation_conflict"), 0.0)
        self.assertEqual(bd.negatives, [])

    def test_shared_syllable_is_not_a_negation_conflict(self):
        """Loi that tim thay khi soat tap eval: "du" trong "du lieu".

        A phu dinh "hien thi du lieu"; B chi nhac "du lieu" o ngu canh khac.
        Doi chieu tung am tiet thi bao mau thuan -- hoan toan bia.
        """
        f, _ = _pair("ra man search khong hien thi du lieu",
                     "Loi phan trang du lieu cua field")
        self.assertEqual(f.get("negation_conflict"), 0.0)

    def test_numbers_in_unrelated_places_are_not_conflict(self):
        """"buoc 1" vs "sau 5 ngay" khong phai mau thuan gia tri so."""
        f, _ = _pair("buoc 1 tao request", "sau 5 ngay request het han")
        self.assertEqual(f.get("numeric_conflict"), 0.0)

    def test_numbers_in_same_field_are_conflict(self):
        f, _ = _pair("Nhap so tien = 0 bao loi", "Nhap so tien = 1 bao loi")
        self.assertEqual(f.get("numeric_conflict"), 1.0)

    def test_one_sided_missing_info_is_not_conflict(self):
        """Mot ben thieu du lieu = THIEU BANG CHUNG, khong phai bang chung am."""
        f, _ = _pair("Agent = TERMINATED loi", "Agent loi")
        self.assertEqual(f.get("status_conflict"), 0.0)

    def test_gate_caps_score_when_bonus_would_push_it_up(self):
        """Bang chung duong nhieu KHONG duoc keo mot hard negative len duplicate."""
        cfg = DupConfig()
        f, bd = _pair("[USSD EU][Buy bundle] Agent = TERMINATED loi ER80011",
                      "[USSD EU][Buy bundle] Agent = LOCKED loi ER80011",
                      cfg, semantic=1.0, lexical=1.0)
        self.assertGreater(bd.bonus, 0.0)             # co bang chung duong
        self.assertIn("status_conflict", bd.strong_conflicts)
        self.assertTrue(bd.gate_applied)
        self.assertLessEqual(bd.final, cfg.scoring.hard_negative_cap)

    def test_reranker_cannot_break_the_gate(self):
        """Cross-encoder cham 1.0 van khong pha duoc tran hard-negative."""
        from pipeline.dup.rerank import _reapply
        cfg = DupConfig()
        f, bd = _pair("[USSD EU][Buy bundle] Agent = TERMINATED loi ER80011",
                      "[USSD EU][Buy bundle] Agent = LOCKED loi ER80011",
                      cfg, semantic=1.0, lexical=1.0)
        sc = ScoredCandidate(idx=0, key="B", features=f, breakdown=bd,
                             final_score=bd.final)
        self.assertLessEqual(_reapply(sc, 1.0, cfg.scoring),
                             cfg.scoring.hard_negative_cap)

    def test_symptom_conflict_separates_same_module_bugs(self):
        """Cung module, cung ky vong, KHAC hien tuong -> khong phai trung lap.

        Ba bug that trong corpus: RO-10517 "{token} trung voi
        {referenceNumber}", RO-10520 "sms gui {token} sai", RO-11398 "Get
        token tra ve null". Ca ba cung [EU APP][Buy Luku-History], `actual`
        RONG, va `expected` gan nhu trung nhau -- nen expected/observed
        similarity khong tach duoc. Trieu chung doc tu summary thi tach duoc.
        """
        f, _ = _pair("[EU APP][Buy Luku-History] Get token hien thi token "
                     "trung voi referenceNumber",
                     "[EU APP][Buy Luku-History] Get token tra ve null")
        self.assertEqual(f.get("symptom_conflict"), 1.0)

    def test_symptom_conflict_needs_both_sides(self):
        """Mot ben khong ta hien tuong -> thieu bang chung, khong mau thuan."""
        f, _ = _pair("[EU APP][Buy Luku] Get token tra ve null",
                     "[EU APP][Buy Luku] Get token")
        self.assertEqual(f.get("symptom_conflict"), 0.0)

    def test_same_symptom_is_not_a_conflict(self):
        f, _ = _pair("[EU APP][Luku] hien thi sai token",
                     "[EU APP][Luku] sms gui token sai")
        self.assertEqual(f.get("symptom_conflict"), 0.0)

    def test_bonus_alone_cannot_create_a_duplicate(self):
        """Bang chung duong chi CHUNG THUC, khong tu tao ket luan trung lap.

        Loi that: hai bug cung module [EU APP][Buy Luku-History] nhung khac
        hanh vi, base chi 0.75, cong don bonus thi len 1.000 -> 'duplicate'.
        """
        cfg = DupConfig()
        f, bd = _pair("[EU APP][Buy Luku-History] Get token hien thi token "
                      "trung referenceNumber",
                      "[EU APP][Buy Luku-History] lay token thanh cong, sms "
                      "gui token sai", cfg, semantic=0.75, lexical=0.75)
        self.assertLessEqual(bd.bonus, cfg.scoring.max_bonus)
        self.assertLess(bd.final, 1.0)

    def test_provider_difference_is_soft(self):
        """Khac ngan hang chi la penalty nhe, khong phai loai bo."""
        f, bd = _pair("chuyen tien sang CRDB loi timeout",
                      "chuyen tien sang NMB loi timeout")
        self.assertEqual(f.get("provider_conflict"), 1.0)
        self.assertNotIn("provider_conflict", bd.strong_conflicts)
        self.assertGreater(bd.final, 0.5)


# ---------------------------------------------------------------- phase 6
class TestDecision(unittest.TestCase):
    def _sc(self, score, negatives=(), strong=(), gate=False):
        from pipeline.dup.scoring import ScoreBreakdown
        bd = ScoreBreakdown(final=score, negatives=list(negatives),
                            strong_conflicts=list(strong), gate_applied=gate)
        return ScoredCandidate(idx=0, key="B", breakdown=bd, final_score=score)

    def test_three_bands(self):
        eng = DecisionEngine()
        self.assertEqual(eng.label(self._sc(0.95)), "duplicate")
        self.assertEqual(eng.label(self._sc(0.60)), "related")
        self.assertEqual(eng.label(self._sc(0.10)), "different")

    def test_uncertain_band_is_review_not_duplicate(self):
        eng = DecisionEngine()
        self.assertEqual(eng.label(self._sc(0.75)), "review")

    def test_gate_never_yields_duplicate(self):
        eng = DecisionEngine()
        self.assertEqual(eng.label(self._sc(0.99, gate=True)), "related")

    def test_thresholds_are_configurable(self):
        cfg = DupConfig()
        cfg.set_path("decision.t_duplicate", "0.95")
        eng = DecisionEngine(cfg.decision)
        self.assertEqual(eng.label(self._sc(0.96)), "duplicate")
        self.assertEqual(eng.label(self._sc(0.90)), "review")    # sat nguong moi
        self.assertEqual(eng.label(self._sc(0.70)), "related")


# ------------------------------------------------------------- config/E2E
class TestConfig(unittest.TestCase):
    def test_set_path_coerces_types(self):
        cfg = DupConfig()
        cfg.set_path("scoring.use_structured", "false")
        cfg.set_path("candidates.pool_k", "42")
        cfg.set_path("scoring.weights.semantic", "0.9")
        self.assertIs(cfg.scoring.use_structured, False)
        self.assertEqual(cfg.candidates.pool_k, 42)
        self.assertAlmostEqual(cfg.scoring.weights.semantic, 0.9)

    def test_unknown_field_raises(self):
        with self.assertRaises(KeyError):
            DupConfig().set_path("scoring.khong_ton_tai", "1")

    def test_roundtrip_json(self):
        import json
        import tempfile
        cfg = DupConfig()
        cfg.set_path("decision.t_related", "0.42")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            cfg.save(p)
            back = load_config(p)
        self.assertAlmostEqual(back.decision.t_related, 0.42)
        json.loads(cfg.dumps())

    def test_ablation_presets_toggle_phases(self):
        p = ablation_presets()
        self.assertFalse(p["lexical_only"].candidates.use_semantic)
        self.assertFalse(p["semantic_only"].candidates.use_lexical)
        self.assertFalse(p["hybrid"].scoring.use_structured)
        self.assertTrue(p["hybrid_structured"].scoring.use_structured)
        self.assertTrue(p["hybrid_structured_rerank"].rerank.enabled)

    def test_disabling_structured_removes_bonus_and_penalty(self):
        cfg = DupConfig()
        cfg.set_path("scoring.use_structured", "false")
        cfg.set_path("scoring.use_contradiction", "false")
        f, bd = _pair("Agent = TERMINATED loi", "Agent = LOCKED loi", cfg)
        self.assertEqual(bd.bonus, 0.0)
        self.assertEqual(bd.penalty, 0.0)
        self.assertGreater(f.get("status_conflict"), 0)   # van tinh, chi khong dung


class TestFlowRouting(unittest.TestCase):
    """Quy tac chot tu eval_dup_flows (2026-09-12): text -> bat doi xung +
    cross-encoder; issue -> doi xung, khong rerank."""

    def test_defaults_per_flow(self):
        from pipeline.dup.pipeline import flow_plan
        cfg = DupConfig()
        self.assertEqual(flow_plan(cfg, "text"),
                         {"flow": "text", "symmetric": False,
                          "rerank": "cross_encoder", "labels_measured": False})
        self.assertEqual(flow_plan(cfg, "issue"),
                         {"flow": "issue", "symmetric": True,
                          "rerank": "none", "labels_measured": True})

    def test_frozen_config_keeps_routing(self):
        from pipeline.dup.pipeline import flow_plan
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "dup_v2_frozen.json")
        cfg = load_config(path)
        self.assertEqual(flow_plan(cfg, "text")["rerank"], "cross_encoder")
        self.assertEqual(flow_plan(cfg, "issue")["rerank"], "none")

    def test_enabled_forces_kind_on_both_flows(self):
        from pipeline.dup.pipeline import flow_plan
        cfg = load_config(None, ["rerank.enabled=true", "rerank.kind=sklearn"])
        self.assertEqual(flow_plan(cfg, "issue")["rerank"], "sklearn")
        self.assertEqual(flow_plan(cfg, "text")["rerank"], "sklearn")

    def test_flow_flags_can_be_turned_off(self):
        from pipeline.dup.pipeline import flow_plan
        cfg = load_config(None, ["rerank.text=none"])
        self.assertEqual(flow_plan(cfg, "text")["rerank"], "none")
        with self.assertRaises(ValueError):
            flow_plan(cfg, "other")


class TestFeatureVector(unittest.TestCase):
    def test_vector_matches_names(self):
        from pipeline.dup.features import FEATURE_NAMES
        f, _ = _pair("a b c", "a b d")
        self.assertEqual(len(f.vector()), len(FEATURE_NAMES))

    def test_no_label_derived_feature(self):
        from pipeline.dup.features import FEATURE_NAMES
        bad = [n for n in FEATURE_NAMES
               if any(w in n for w in ("label", "gold", "cue", "mention"))]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
