# -*- coding: utf-8 -*-
"""Phase 5 -- reranker cam-vao-duoc.

Hop dong duy nhat:

    Reranker.rerank(query_text, scored: list[ScoredCandidate]) -> list

Ba ban cai san (chon bang rerank.kind trong config, khong sua code):

    feature_weighted   mo hinh tuyen tinh tren dung vector dac trung cua
                       phase 4 -- de doc, de sua trong so, khong can train.
    cross_encoder      BAAI/bge-reranker-base (hoac model khac) tren cap
                       van ban tho.
    sklearn            bat ky uoc luong nao co predict_proba, nap tu joblib
                       (logistic regression / gradient boosting / ...).

QUAN TRONG -- reranker KHONG duoc phep xoa bang chung mau thuan.
Cross-encoder duoc huan luyen tren du lieu chung chung se cham diem rat cao
cho cap "TERMINATED vs LOCKED" vi hai cau gan nhu trung tu. Nen o day diem
reranker chi thay the phan BASE (do giong van ban), roi bonus/penalty va
tran hard-negative cua phase 4 duoc AP LAI. Nho vay bat reranker khong bao
gio lam hong xu ly hard negative.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .scoring import STRONG_CONFLICTS, ScoredCandidate  # noqa: F401


class Reranker:
    name = "identity"

    def rerank(self, query_text: str,
               scored: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
        return list(scored)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.full_like(x, 0.5) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _reapply(sc: ScoredCandidate, new_base: float, cfg_scoring) -> float:
    """Ap lai bonus/penalty/tran hard-negative len mot base moi.

    Cap nhat CA HAI diem, va day la cho de sai:
      rank_score   quyet dinh THU TU  -> neu quen cap nhat thi reranker chay
                   day du, ton thoi gian, ma ket qua khong he doi.
      final_score  quyet dinh NHAN.
    """
    bd = sc.breakdown
    bonus = bd.bonus if cfg_scoring.use_structured else 0.0
    penalty = bd.penalty if cfg_scoring.use_contradiction else 0.0
    rank = max(0.0, min(1.0, new_base + bonus))
    score = max(0.0, min(1.0, new_base + bonus - penalty))
    if (cfg_scoring.use_contradiction and cfg_scoring.hard_negative_gate
            and bd.strong_conflicts
            and new_base >= cfg_scoring.hard_negative_min_sim
            and score > cfg_scoring.hard_negative_cap):
        score = cfg_scoring.hard_negative_cap
        bd.gate_applied = True
    bd.rank_score = round(score if cfg_scoring.penalty_in_ranking else rank, 4)
    bd.final = round(score, 4)
    sc.rank_score = bd.rank_score
    return bd.final


class FeatureWeightedReranker(Reranker):
    """Tuyen tinh tren vector dac trung -- mac dinh trung voi HybridScorer.

    Ton tai de chung minh duong noi: doi sang mot bo he so khac (hoc duoc
    hay chinh tay) chi la doi mang `coef`, khong cham pipeline.
    """
    name = "feature_weighted"

    def __init__(self, cfg_rerank, cfg_scoring, coef: np.ndarray | None = None):
        self.cfg = cfg_rerank
        self.cfg_scoring = cfg_scoring
        self.coef = coef

    def rerank(self, query_text, scored):
        scored = list(scored)
        head = scored[: self.cfg.top_n]
        if self.coef is None:
            for sc in head:
                sc.rerank_score = sc.breakdown.base
            return scored
        for sc in head:
            v = sc.features.vector()
            raw = float(np.dot(self.coef, v))
            new_base = 1.0 / (1.0 + np.exp(-raw))
            sc.rerank_score = round(new_base, 4)
            b = (1 - self.cfg.blend) * sc.breakdown.base + self.cfg.blend * new_base
            sc.final_score = _reapply(sc, b, self.cfg_scoring)
        return scored


class CrossEncoderRerankerAdapter(Reranker):
    """Cross-encoder tren cap van ban tho, chi thay phan BASE."""
    name = "cross_encoder"

    def __init__(self, cfg_rerank, cfg_scoring, text_field: str = "text_dense"):
        from ._compat import CrossEncoderReranker
        self.cfg = cfg_rerank
        self.cfg_scoring = cfg_scoring
        self.text_field = text_field
        try:
            import torch
            torch.set_num_threads(8)
        except Exception:
            pass
        self._ce = CrossEncoderReranker(
            cfg_rerank.model, device=cfg_rerank.device,
            batch_size=cfg_rerank.batch_size,
            max_length=getattr(cfg_rerank, "max_length", 256))

    def rerank(self, query_text, scored):
        scored = list(scored)
        head = scored[: self.cfg.top_n]
        if not head:
            return scored
        docs = [sc.doc.get(self.text_field, "") or sc.doc.get("summary", "")
                for sc in head]
        pairs = [(query_text, d) for d in docs]
        raw = self._ce.model.predict(pairs, batch_size=self.cfg.batch_size,
                                     show_progress_bar=False)
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = _minmax(raw)
        for sc, r, nz in zip(head, raw, norm):
            sc.rerank_score = round(float(r), 4)
            b = (1 - self.cfg.blend) * sc.breakdown.base + self.cfg.blend * float(nz)
            sc.final_score = _reapply(sc, b, self.cfg_scoring)
        return scored


class SklearnPairReranker(Reranker):
    """Bat ky uoc luong sklearn nao co predict_proba tren vector dac trung."""
    name = "sklearn"

    def __init__(self, cfg_rerank, cfg_scoring):
        import joblib
        self.cfg = cfg_rerank
        self.cfg_scoring = cfg_scoring
        self.model = joblib.load(cfg_rerank.model_path)

    def rerank(self, query_text, scored):
        scored = list(scored)
        head = scored[: self.cfg.top_n]
        if not head:
            return scored
        X = np.vstack([sc.features.vector() for sc in head])
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[:, 1]
        else:
            p = _minmax(np.asarray(self.model.decision_function(X)))
        for sc, pi in zip(head, p):
            sc.rerank_score = round(float(pi), 4)
            # Mo hinh hoc TREN vector dac trung -> da nhin thay mau thuan.
            # Van ap lai tran hard-negative: nguyen tac giam false positive
            # khong duoc phu thuoc vao viec model co hoc dung hay khong.
            b = (1 - self.cfg.blend) * sc.breakdown.base + self.cfg.blend * float(pi)
            sc.final_score = _reapply(sc, b, self.cfg_scoring)
        return scored


def build_reranker(cfg_rerank, cfg_scoring, text_field: str = "text_dense"):
    if not cfg_rerank.enabled or cfg_rerank.kind in ("none", ""):
        return Reranker()
    kind = cfg_rerank.kind
    if kind == "feature_weighted":
        return FeatureWeightedReranker(cfg_rerank, cfg_scoring)
    if kind == "cross_encoder":
        return CrossEncoderRerankerAdapter(cfg_rerank, cfg_scoring, text_field)
    if kind == "sklearn":
        return SklearnPairReranker(cfg_rerank, cfg_scoring)
    raise ValueError(f"rerank.kind khong ho tro: {kind}")
