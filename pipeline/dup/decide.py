# -*- coding: utf-8 -*-
"""Phase 6 -- quyet dinh nhan cuoi cung + giai trinh.

Bon nhan, KHONG phai hai:

    duplicate   diem cao VA khong con tin hieu mau thuan nao
    review      diem cao NHUNG co mau thuan, hoac diem nam sat nguong
                -> chua du bang chung, de NGUOI quyet
    related     lien quan, khong ket luan trung lap
    different   khac nhau

'review' ton tai vi mot ly do cu the: khong duoc am tham bien "khong chac"
thanh "trung lap". Trong tap huan luyen, mot false positive day hai bug khac
nhau lai gan nhau va loi do se duoc hoc lai; mot false negative chi la bo sot.

Nguong deu cau hinh duoc (decision.t_duplicate / t_related / review_margin),
khong hard-code o bat ky cho nao khac.

DOI TRUC BAI TOAN -- DOC TRUOC KHI SUA GI O DAY
De tai da chuyen tu "phat hien trung lap" sang "truy hoi issue tuong dong de
tai su dung tri thuc". Ly do la so lieu, khong phai so thich: tren Jira QA noi
bo nay, ty le trung lap that chi 0,69% (75 cap co cue 'trung' tren 10.925 bug),
44% cac cap do cach nhau <= 1 ngay va 45% cung nguoi bao -- tuc QA chu dong tao
ticket song song khi quet mot man hinh qua nhieu portal, khong phai ai do bao
lai loi da fix. Bai toan DBRD kinh dien (Bugzilla/Mozilla, nguoi dung cuoi, ty
le 10-30%) khong ton tai o day.

Ten bon nhan GIU NGUYEN de khong pha config/test da co, nhung y nghia doc theo
truc moi la MUC DO DANG DOC, khong phai phan quyet trung lap:

    duplicate  -> "gan nhu chac chan giup duoc": cung van de hoac cung co che
    review     -> "co the giup, nhung co diem mau thuan -- nguoi doc tu can"
    related    -> "dang doc, co the co manh moi"
    different  -> "khong giup gi"

He qua: nguong t_duplicate khong con la mot quyet dinh nghiep vu nang (dong
ticket hay khong), ma chi la ranh gioi hien thi. Bo may `--mode tune-threshold`
theo san precision vi the bot quan trong -- xep hang de doc thi khong can mot
quyet dinh cung. Cong hard-negative thi VAN GIU: no ngan viec day len dau mot
issue trong giong nhung mau thuan ve thao tac/trang thai, thu se lam ky su di
sai huong.
"""
from __future__ import annotations

from .scoring import ScoredCandidate

# Ten dac trung -> cach doc cho nguoi
_VI = {
    "semantic_similarity": "ngu nghia", "lexical_similarity": "tu vung",
    "token_jaccard": "trung token",
    "function_match": "cung chuc nang", "module_match": "cung module",
    "domain_match": "cung domain", "bank_match": "cung ngan hang",
    "provider_match": "cung nha cung cap", "api_match": "cung API",
    "channel_match": "cung kenh", "entity_overlap": "trung thuc the",
    "field_overlap": "trung field", "action_overlap": "cung thao tac",
    "status_match": "cung trang thai", "error_code_match": "cung ma loi",
    "expected_behavior_similarity": "ky vong giong",
    "observed_behavior_similarity": "hien tuong giong",
    "operator_conflict": "KHAC TOAN TU", "numeric_conflict": "KHAC GIA TRI SO",
    "negation_conflict": "LECH PHU DINH", "status_conflict": "KHAC TRANG THAI",
    "action_conflict": "KHAC THAO TAC", "identifier_conflict": "KHAC DINH DANH",
    "entity_conflict": "KHAC FIELD", "actor_conflict": "KHAC TAC NHAN",
    "channel_conflict": "KHAC KENH", "provider_conflict": "khac nha cung cap",
    "error_code_conflict": "khac ma loi",
}


class DecisionEngine:
    def __init__(self, cfg=None):
        from .config import DecisionConfig
        self.cfg = cfg or DecisionConfig()

    def label(self, sc: ScoredCandidate) -> str:
        cfg = self.cfg
        s = sc.final_score
        bd = sc.breakdown
        if bd.gate_applied:
            # Hard negative: van ban rat giong nhung mau thuan manh.
            # Van tra ve cho ky su xem, nhung KHONG BAO GIO la 'duplicate'.
            return "related"
        if s >= cfg.t_duplicate:
            # Con mau thuan manh o vung diem cao -> de nguoi quyet.
            return "review" if bd.strong_conflicts else "duplicate"
        if s >= cfg.t_duplicate - cfg.review_margin:
            return "review"
        if s >= cfg.t_related:
            return "related"
        return "different"

    def explain(self, sc: ScoredCandidate) -> str:
        cfg = self.cfg
        bd = sc.breakdown
        n = cfg.max_explanation_signals
        parts = [f"ngu nghia={bd.semantic:.2f} tu vung={bd.lexical:.2f} "
                 f"base={bd.base:.2f}"]

        pos = sorted(bd.positives, key=lambda x: -x[1])[:n]
        if pos:
            parts.append("cong: " + ", ".join(
                f"{_VI.get(k, k)}(+{v:.2f})" for k, v in pos))

        neg = sorted(bd.negatives, key=lambda x: -x[1])[:n]
        if neg:
            det = sc.features.detail if sc.features else {}
            items = []
            for k, v in neg:
                d = det.get(k)
                extra = f" [{d[0]}]" if d else ""
                items.append(f"{_VI.get(k, k)}(-{v:.2f}){extra}")
            parts.append("tru: " + ", ".join(items))

        if bd.gate_applied:
            parts.append(f"HARD-NEGATIVE: van ban rat giong (base={bd.base:.2f}) "
                         f"nhung co mau thuan manh "
                         f"({', '.join(_VI.get(k, k) for k in bd.strong_conflicts)})"
                         f" -> chan tran o {sc.final_score:.2f}, khong xep trung lap")
        if sc.rerank_score is not None:
            parts.append(f"reranker={sc.rerank_score:.3f}")
        return "; ".join(parts)

    def decide(self, scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
        for sc in scored:
            sc.label = self.label(sc)
            sc.explanation = self.explain(sc)
        return scored
