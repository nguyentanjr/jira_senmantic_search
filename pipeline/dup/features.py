# -*- coding: utf-8 -*-
"""Phase 4a -- vector dac trung cho MOT CAP bug.

Quy uoc BAT BUOC cua toan bo file nay:

  * moi dac trung nam trong [0, 1];
  * "khac nhau" chi tinh khi CA HAI ben deu co thong tin va lech HAI CHIEU.
    Mot ben thieu du lieu la THIEU BANG CHUNG, khong phai bang chung phu dinh.
    (Neu khong, bug viet ngan se bi phat oan so voi bug viet day du.)
  * dac trung duong va dac trung mau thuan tach hai nhom rieng -- phase 4b
    cong/tru chung voi trong so khac nhau, va reranker hoc duoc tren dung
    vector nay ma khong phai doi gi.

KHONG dac trung nao duoc dung den nhan. Dac biet: KHONG co dac trung kieu
"van ban A co nhac ma issue cua B" -- tap gold duoc dao ra tu chinh cau nhac
do, dung no la ro ri nhan truc tiep.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lexicons as LX
from .extract import BugRepresentation

# Thu tu nay la HOP DONG voi reranker da huan luyen -- chi them vao cuoi.
FEATURE_NAMES = (
    # -- bang chung tuong dong
    "semantic_similarity", "lexical_similarity", "token_jaccard",
    "function_match", "module_match", "domain_match",
    "bank_match", "provider_match", "api_match", "channel_match",
    "entity_overlap", "field_overlap", "action_overlap", "status_match",
    "error_code_match",
    "expected_behavior_similarity", "observed_behavior_similarity",
    # -- bang chung mau thuan
    "operator_conflict", "numeric_conflict", "negation_conflict",
    "status_conflict", "action_conflict", "identifier_conflict",
    "entity_conflict", "actor_conflict", "channel_conflict",
    "provider_conflict", "error_code_conflict",
    # Them o CUOI de khong pha hop dong thu tu voi reranker da huan luyen.
    "symptom_conflict",
)

POSITIVE_FEATURES = FEATURE_NAMES[:17]
CONFLICT_FEATURES = FEATURE_NAMES[17:]


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coef(a: set, b: set) -> float:
    """Do phu chong -- do dai van ban lech nhau nhieu, Jaccard phat oan."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def two_way_diff(a: set, b: set) -> tuple[set, set]:
    """Chenh HAI CHIEU. Mot chieu chi la "ben kia viet day du hon"."""
    da, db = a - b, b - a
    return (da, db) if (da and db) else (set(), set())


@dataclass
class PairFeatures:
    a: str = ""
    b: str = ""
    values: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)     # chi tiet de giai thich

    def get(self, name: str) -> float:
        return float(self.values.get(name, 0.0))

    def vector(self) -> np.ndarray:
        return np.array([self.get(n) for n in FEATURE_NAMES], dtype=np.float32)

    def positives(self) -> list[tuple[str, float]]:
        return [(n, self.get(n)) for n in POSITIVE_FEATURES if self.get(n) > 0]

    def conflicts(self) -> list[tuple[str, float]]:
        return [(n, self.get(n)) for n in CONFLICT_FEATURES if self.get(n) > 0]


class PairFeatureExtractor:
    """(rep_a, rep_b, diem retrieval) -> PairFeatures."""

    def __init__(self, cfg=None):
        from .config import ScoringConfig
        self.cfg = cfg or ScoringConfig()

    def extract(self, a: BugRepresentation, b: BugRepresentation,
                semantic: float = 0.0, lexical: float = 0.0) -> PairFeatures:
        v: dict[str, float] = {}
        d: dict[str, object] = {}

        # ------------------------------------------------ tuong dong tho
        v["semantic_similarity"] = float(max(0.0, min(1.0, semantic)))
        v["lexical_similarity"] = float(max(0.0, min(1.0, lexical)))
        v["token_jaccard"] = jaccard(a.words, b.words)

        # ------------------------------------------------ khop co cau truc
        v["function_match"] = overlap_coef(a.function, b.function)
        v["module_match"] = jaccard(a.module, b.module)
        v["domain_match"] = jaccard(a.domain, b.domain)
        v["bank_match"] = jaccard(a.bank, b.bank)
        v["provider_match"] = jaccard(a.provider, b.provider)
        v["api_match"] = overlap_coef(a.api, b.api)
        v["channel_match"] = jaccard(a.channel, b.channel)
        v["entity_overlap"] = overlap_coef(a.entities, b.entities)
        v["field_overlap"] = overlap_coef(a.fields_, b.fields_)
        v["action_overlap"] = overlap_coef(a.actions, b.actions)
        v["status_match"] = jaccard(a.statuses, b.statuses)
        v["error_code_match"] = overlap_coef(a.error_codes, b.error_codes)
        v["expected_behavior_similarity"] = overlap_coef(a.expected_words,
                                                         b.expected_words)
        v["observed_behavior_similarity"] = overlap_coef(a.observed_words,
                                                         b.observed_words)

        # ------------------------------------------------ mau thuan
        # 1. toan tu:  "From Date = To Date"  vs  "From Date < To Date"
        ops = LX.operator_conflict(a.operators, b.operators)
        v["operator_conflict"] = 1.0 if ops else 0.0
        if ops:
            d["operator_conflict"] = ops

        # 2. con so -- doi chieu theo CUNG MOT TRUONG, khong so hai tui so.
        #    "so tien = 0" vs "so tien = 1" -> mau thuan (cung truong "tien").
        #    "buoc 1 ..."  vs "sau 5 ngay"  -> KHONG, hai con so khac cho.
        num_hits = []
        for kctx, va in (a.number_ctx or {}).items():
            vb = (b.number_ctx or {}).get(kctx)
            if vb and not (va & vb):
                num_hits.append(f"{kctx}: {sorted(va)[:2]}!={sorted(vb)[:2]}")
        v["numeric_conflict"] = 1.0 if num_hits else 0.0
        if num_hits:
            d["numeric_conflict"] = num_hits[:3]

        # 3. phu dinh -- doi chieu theo PHAM VI, khong dem toan van.
        #    Chi tinh mau thuan khi CUNG mot vi ngu xuat hien o hai ben ma
        #    mot ben phu dinh, ben kia khang dinh:
        #        "hien thi thong bao"        <-> "khong hien thi thong bao"
        #    Hai cau paraphrase ("khong dang nhap duoc" / "bi tu choi xac
        #    thuc") KHONG chung vi ngu nen khong bi tinh la mau thuan.
        na_pos = {w for w in a.neg_scope
                  if w in b.raw_bigrams and w not in b.neg_scope}
        nb_pos = {w for w in b.neg_scope
                  if w in a.raw_bigrams and w not in a.neg_scope}
        v["negation_conflict"] = 1.0 if (na_pos or nb_pos) else 0.0
        if na_pos or nb_pos:
            d["negation_conflict"] = [
                "phu dinh lech tren: " + ", ".join(sorted(na_pos | nb_pos)[:3])]

        # 4. trang thai:  "Agent = TERMINATED"  vs  "Agent = LOCKED"
        st = LX.status_conflict(a.statuses, b.statuses)
        v["status_conflict"] = 1.0 if st else 0.0
        if st:
            d["status_conflict"] = st

        # 5. thao tac:  "Bam Save"  vs  "Bam Submit"
        ac = LX.action_conflict(a.actions, b.actions)
        v["action_conflict"] = 1.0 if ac else 0.0
        if ac:
            d["action_conflict"] = ac

        # 6. dinh danh co nghia (funcCode / class) -- KHONG tinh traceId
        ia, ib = two_way_diff(a.identifiers, b.identifiers)
        v["identifier_conflict"] = 1.0 if (ia and ib) else 0.0
        if ia and ib:
            d["identifier_conflict"] = [f"{sorted(ia)[:2]}!={sorted(ib)[:2]}"]

        # 7. field ky thuat:  requestTime  vs  accessChannelID
        fa, fb = two_way_diff(a.fields_, b.fields_)
        if fa and fb:
            # chi tinh khi ben nao cung it field -- van ban dai co hang chuc
            # field, lech nhau la binh thuong, khong phai mau thuan.
            small = len(a.fields_) <= 6 and len(b.fields_) <= 6
            v["entity_conflict"] = 1.0 if small else 0.3
            d["entity_conflict"] = [f"{sorted(fa)[:2]}!={sorted(fb)[:2]}"]
        else:
            v["entity_conflict"] = 0.0

        # 8. vai tro:  EU vs Agent vs Super Agent  (truc phan biet manh)
        ra, rb = two_way_diff(a.actor, b.actor)
        v["actor_conflict"] = 1.0 if (ra and rb) else 0.0
        if ra and rb:
            d["actor_conflict"] = [f"{sorted(ra)}!={sorted(rb)}"]

        # 9. kenh:  USSD vs APP
        ca, cb = two_way_diff(a.channel, b.channel)
        v["channel_conflict"] = 1.0 if (ca and cb) else 0.0
        if ca and cb:
            d["channel_conflict"] = [f"{sorted(ca)}!={sorted(cb)}"]

        # 10. nha cung cap -- PENALTY NHE: khac bank van co the cung root cause
        va_, vb_ = two_way_diff(a.provider, b.provider)
        v["provider_conflict"] = 1.0 if (va_ and vb_) else 0.0
        if va_ and vb_:
            d["provider_conflict"] = [f"{sorted(va_)}!={sorted(vb_)}"]

        # 11. ma loi khac nhau hoan toan
        ea, eb = two_way_diff(a.error_codes, b.error_codes)
        v["error_code_conflict"] = 1.0 if (ea and eb and
                                           not (a.error_codes & b.error_codes)) else 0.0
        if v["error_code_conflict"]:
            d["error_code_conflict"] = [f"{sorted(ea)[:2]}!={sorted(eb)[:2]}"]

        # 12. trieu chung -- "tra ve null" vs "hien thi sai" vs "trung gia tri"
        #     la ba loi khac nhau du cung module va cung ky vong.
        sym = LX.symptom_conflict(a.symptoms, b.symptoms)
        v["symptom_conflict"] = 1.0 if sym else 0.0
        if sym:
            d["symptom_conflict"] = sym[:3]

        return PairFeatures(a=a.key, b=b.key, values=v, detail=d)
