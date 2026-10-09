# -*- coding: utf-8 -*-
"""Phase 4b -- gop dac trung thanh mot diem, co giai trinh.

Cong thuc:

    base    = (w_sem * semantic + w_lex * lexical) / (w_sem + w_lex)
    bonus   = SUM  w_i * feature_duong_i
    penalty = SUM  p_j * feature_mau_thuan_j
    score   = clip(base + bonus - penalty, 0, 1)

Hai lua chon thiet ke co chu y:

1. base la trung binh CO TRONG SO, khong phai tong -- de score van nam trong
   [0,1] va nguong co y nghia on dinh khi bat/tat semantic hoac lexical.

2. CONG TAC HARD-NEGATIVE (hard_negative_gate). Rieng nhom cap:

       tuong dong rat cao  +  mau thuan manh
       (TERMINATED vs LOCKED, Save vs Submit, "=" vs "<", 0 vs 1)

   khong duoc phep tu dong len 'duplicate' du penalty co bi trong so keo
   khong du. Nen ngoai phan tru tuyen tinh, con mot TRAN diem cung
   (hard_negative_cap) dat duoi nguong duplicate. Cap loai nay se roi vao
   dai 'related', tuc van hien ra cho ky su, nhung khong bao gio duoc gan
   nhan trung lap tu dong.

   Ly do: false positive dat hon false negative. Mot goi y sai day hai bug
   khac nhau lai gan nhau -- va neu phan hoi do quay lai tap huan luyen thi
   loi duoc nhan doi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import CONFLICT_FEATURES, POSITIVE_FEATURES, PairFeatures

# Mau thuan du manh de kich hoat tran diem hard-negative.
# provider/error_code/identifier KHONG nam trong day: chung chi la penalty
# tuyen tinh, vi khac provider hoac khac ma loi van co the cung root cause.
# entity_conflict CO trong day nhung chi kich hoat khi gia tri = 1.0, tuc
# hai ben deu it field va lech han nhau (requestTime vs accessChannelID);
# van ban dai co hang chuc field chi duoc 0.3 nen khong cham tran.
STRONG_CONFLICTS = (
    "operator_conflict", "numeric_conflict", "negation_conflict",
    "status_conflict", "action_conflict", "actor_conflict",
    "channel_conflict", "entity_conflict", "symptom_conflict",
)

# Dac trung duong khong tinh vao bonus (chung da nam trong base).
_BASE_FEATURES = ("semantic_similarity", "lexical_similarity")


@dataclass
class ScoreBreakdown:
    final: float = 0.0          # diem QUYET DINH (co tru mau thuan)
    rank_score: float = 0.0     # diem XEP HANG (chi bang chung duong)
    base: float = 0.0
    bonus: float = 0.0
    penalty: float = 0.0
    semantic: float = 0.0
    lexical: float = 0.0
    positives: list[tuple[str, float]] = field(default_factory=list)
    negatives: list[tuple[str, float]] = field(default_factory=list)
    strong_conflicts: list[str] = field(default_factory=list)
    gate_applied: bool = False

    def positive_signals(self, top: int = 4) -> list[str]:
        xs = sorted(self.positives, key=lambda x: -x[1])[:top]
        return [f"{n}={v:.2f}" for n, v in xs]

    def negative_signals(self, top: int = 4) -> list[str]:
        xs = sorted(self.negatives, key=lambda x: -x[1])[:top]
        return [f"{n}=-{v:.2f}" for n, v in xs]


@dataclass
class ScoredCandidate:
    """Mot ung vien da co day du bang chung -- don vi di chuyen giua cac phase."""
    idx: int
    key: str
    doc: dict = field(default_factory=dict)
    rep: object = None
    candidate: object = None
    features: PairFeatures | None = None
    breakdown: ScoreBreakdown | None = None
    rerank_score: float | None = None
    final_score: float = 0.0        # diem quyet dinh nhan
    rank_score: float = 0.0         # diem xep hang
    label: str = ""
    explanation: str = ""


class HybridScorer:
    """Weighted scoring. Thay bang model hoc duoc o phase 5, khong sua phase nay."""

    def __init__(self, cfg=None):
        from .config import ScoringConfig
        self.cfg = cfg or ScoringConfig()
        self.w = self.cfg.weights

    # ------------------------------------------------------------- base
    def _base(self, f: PairFeatures) -> tuple[float, float, float]:
        cfg, w = self.cfg, self.w
        sem = f.get("semantic_similarity")
        lex = f.get("lexical_similarity")
        num = den = 0.0
        if cfg.use_semantic:
            num += w.semantic * sem
            den += w.semantic
        if cfg.use_lexical:
            num += w.lexical * lex
            den += w.lexical
        if den == 0:
            # Ca hai deu tat -> roi ve tuong dong tui-tu de van co thu tu.
            return f.get("token_jaccard"), sem, lex
        return num / den, sem, lex

    # ------------------------------------------------------------ score
    def score(self, f: PairFeatures) -> ScoreBreakdown:
        cfg, w = self.cfg, self.w
        base, sem, lex = self._base(f)
        out = ScoreBreakdown(base=round(base, 4), semantic=round(sem, 4),
                             lexical=round(lex, 4))

        bonus = 0.0
        if cfg.use_structured:
            for name in POSITIVE_FEATURES:
                if name in _BASE_FEATURES or name == "token_jaccard":
                    continue
                val = f.get(name)
                if val <= 0:
                    continue
                weight = getattr(w, _weight_name(name), 0.0)
                if weight:
                    c = weight * val
                    bonus += c
                    out.positives.append((name, round(c, 4)))

        penalty = 0.0
        if cfg.use_contradiction:
            for name in CONFLICT_FEATURES:
                val = f.get(name)
                if val <= 0:
                    continue
                weight = getattr(w, name, 0.0)
                if weight:
                    c = weight * val
                    penalty += c
                    out.negatives.append((name, round(c, 4)))
                if name in STRONG_CONFLICTS and val >= 0.99:
                    out.strong_conflicts.append(name)

        # Hai diem, hai muc dich khac nhau -- day la diem thiet ke quan trong.
        #
        # rank_score  chi bang chung DUONG. Quyet dinh cai gi duoc hien ra.
        # final       co tru mau thuan + tran hard-negative. Quyet dinh cai
        #             do duoc GOI LA GI.
        #
        # Do lai tren tap eval held-out: tru mau thuan vao diem xep hang lam
        # recall@10 tut tu 0.502 xuong 0.488. Ly do la tap gold ghi nhan cac
        # cap ky su danh dau "tuong tu", va nhieu cap trong so do CO khac tac
        # nhan/kenh -- vd RO-10006 [EU USSD] va RO-10025 [AGENT USSD] cung
        # loi "System busy". Phat vao xep hang thi day chinh nhung cap do ra
        # khoi top-K, tuc GIAU mat thong tin ky su can.
        #
        # Cach dung cua he thong khong doi hoi phai giau: cap mau thuan van
        # nen hien ra, chi la duoi nhan 'related' thay vi 'duplicate'. Nen
        # mac dinh penalty KHONG vao xep hang. Bat penalty_in_ranking=true
        # neu muon uu tien do sach cua top-K hon do phu.
        bonus = min(bonus, cfg.max_bonus)
        rank_score = max(0.0, min(1.0, base + bonus))
        score = base + bonus - penalty
        score = max(0.0, min(1.0, score))

        if (cfg.use_contradiction and cfg.hard_negative_gate
                and out.strong_conflicts
                and base >= cfg.hard_negative_min_sim
                and score > cfg.hard_negative_cap):
            score = cfg.hard_negative_cap
            out.gate_applied = True

        out.bonus = round(bonus, 4)
        out.penalty = round(penalty, 4)
        out.final = round(score, 4)
        out.rank_score = round(score if cfg.penalty_in_ranking else rank_score, 4)
        return out


def _weight_name(feature: str) -> str:
    """Ten dac trung -> ten truong trong so (mot vai cho viet tat)."""
    return {
        "expected_behavior_similarity": "expected_behavior_sim",
        "observed_behavior_similarity": "observed_behavior_sim",
    }.get(feature, feature)
