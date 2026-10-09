# -*- coding: utf-8 -*-
"""Lop dieu phoi E2E -- noi bay phase lai voi nhau.

    Jira bug
      -> Preprocessor            (phase 1)
      -> StructuredExtractor     (phase 2)
      -> CandidateGenerator      (phase 3: lexical + semantic + metadata)
      -> PairFeatureExtractor    (phase 4a)
      -> HybridScorer            (phase 4b)
      -> Reranker                (phase 5, tat/bat)
      -> DecisionEngine          (phase 6)
      -> Top-K                   (phase 7)

Lop nay KHONG chua logic tinh diem. No chi noi day chuyen -- doi mot phase
chi la doi mot doi tuong duoc tiem vao.

Dung:
    from pipeline.dup import DuplicatePipeline, load_config
    p = DuplicatePipeline(load_config()).build()
    print(p.search_by_key("RO-10006")["results"][0])
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ._compat import DenseIndex, load_jsonl
from .candidates import CandidateGenerator
from .config import DupConfig
from .decide import DecisionEngine
from .extract import BugRepresentation, StructuredExtractor
from .features import PairFeatureExtractor
from .preprocess import Preprocessor
from .rerank import build_reranker
from .scoring import HybridScorer, ScoredCandidate

E5_QUERY = "query: "
E5_PASSAGE = "passage: "


def flow_plan(cfg: DupConfig, flow: str) -> dict:
    """Cau hinh HIEU LUC cho mot luong. Ham thuan de test duoc.

    issue = goi y tu mot issue co san (search_by_key / rank_keys)
    text  = mo ta tu do (search() khong kem issue)
    rerank.enabled=true ep `rerank.kind` cho moi luong (che do thi nghiem cu);
    nguoc lai lay `rerank.text` / `rerank.issue`. So do: xem config.RerankConfig.
    """
    if flow not in ("issue", "text"):
        raise ValueError(f"luong khong hop le: {flow}")
    r, e = cfg.rerank, cfg.embedding
    kind = r.kind if r.enabled else getattr(r, flow)
    return {"flow": flow,
            "symmetric": e.symmetric if flow == "issue" else e.text_symmetric,
            "rerank": "none" if kind in ("", "none") else kind,
            # nguong nhan duoc chinh tren luong issue voi hieu chuan doi xung
            "labels_measured": flow == "issue"}


@dataclass
class Timing:
    embed_ms: list = field(default_factory=list)
    candidate_ms: list = field(default_factory=list)
    feature_ms: list = field(default_factory=list)
    rerank_ms: list = field(default_factory=list)

    def summary(self) -> dict:
        def p(xs, q):
            return round(float(np.percentile(xs, q)), 2) if xs else 0.0
        return {
            "embed_p50": p(self.embed_ms, 50), "embed_p95": p(self.embed_ms, 95),
            "cand_p50": p(self.candidate_ms, 50), "cand_p95": p(self.candidate_ms, 95),
            "feat_p50": p(self.feature_ms, 50), "feat_p95": p(self.feature_ms, 95),
            "rerank_p50": p(self.rerank_ms, 50), "rerank_p95": p(self.rerank_ms, 95),
        }


class DuplicatePipeline:
    def __init__(self, cfg: DupConfig | None = None, docs: list[dict] | None = None):
        self.cfg = cfg or DupConfig()
        self.docs = docs if docs is not None else load_jsonl(self.cfg.corpus_path)
        self.pre = Preprocessor(self.cfg.preprocess)
        self.extractor = StructuredExtractor(self.cfg.extract, self.pre)
        self.feat = PairFeatureExtractor(self.cfg.scoring)
        self.scorer = HybridScorer(self.cfg.scoring)
        self.decider = DecisionEngine(self.cfg.decision)
        self.reps: list[BugRepresentation] = []
        self.key_to_idx: dict[str, int] = {}
        self.dense: DenseIndex | None = None
        self.gen: CandidateGenerator | None = None
        self._rerankers: dict = {}
        self._text_dense: tuple | None = None   # (DenseIndex, (lo, hi))
        self.timing = Timing()
        self._built = False

    # ------------------------------------------------------------- build
    def build(self, show_progress: bool = True) -> "DuplicatePipeline":
        cfg = self.cfg
        t0 = time.time()
        self.reps = [self.extractor.extract(d) for d in self.docs]
        self.key_to_idx = {d["key"]: i for i, d in enumerate(self.docs)}
        if show_progress:
            print(f"  phase 1-2: {len(self.reps)} bug -> BugRepresentation "
                  f"({time.time()-t0:.1f}s)")

        if cfg.candidates.use_semantic:
            e = cfg.embedding
            pq = pp = ""
            if "e5" in e.model.lower():
                pq = E5_QUERY
                pp = E5_QUERY if e.symmetric else E5_PASSAGE
            self.dense = DenseIndex(e.model, cache_dir=e.cache_dir, device=e.device,
                                    batch_size=e.batch_size,
                                    query_prefix=pq, passage_prefix=pp)

        self.gen = CandidateGenerator(self.docs, self.reps, cfg.candidates,
                                      self.pre, self.dense)
        self.gen.cfg_embed_field = cfg.embedding.text_field
        self.gen.embed_cfg = cfg.embedding
        t0 = time.time()
        self.gen.build(show_progress=show_progress)
        if show_progress:
            print(f"  phase 3: index xong ({time.time()-t0:.1f}s)"
                  + f", hieu chuan: semantic[{self.gen.sem_lo:.3f},"
                    f"{self.gen.sem_hi:.3f}] lexical[{self.gen.lex_lo:.3f},"
                    f"{self.gen.lex_hi:.3f}]")

        self._built = True
        return self

    # ------------------------------------------------------ theo luong
    def _reranker(self, kind: str):
        """Nap reranker khi can lan dau (cross-encoder ~1 GB)."""
        if kind not in self._rerankers:
            rc = replace(self.cfg.rerank, enabled=True, kind=kind)
            self._rerankers[kind] = build_reranker(rc, self.cfg.scoring,
                                                   self.cfg.embedding.text_field)
        return self._rerankers[kind]

    def _dense_for(self, symmetric: bool):
        """-> (dense, sem_range). None = dung chi so chinh da build."""
        e = self.cfg.embedding
        if (self.dense is None or symmetric == e.symmetric
                or "e5" not in e.model.lower()):
            return None, None
        if self._text_dense is None:
            d = DenseIndex(e.model, cache_dir=e.cache_dir, device=e.device,
                           batch_size=e.batch_size, query_prefix=E5_QUERY,
                           passage_prefix=E5_QUERY if symmetric else E5_PASSAGE)
            self.gen.build_semantic_into(d)
            self._text_dense = (d, self.gen.semantic_calibration(d.emb))
        return self._text_dense

    # ------------------------------------------------------------ search
    def search(self, query_text: str, query_rep: BugRepresentation | None = None,
               exclude_keys: set[str] | None = None, top_k: int | None = None,
               query_id: str = "__query__", query_idx: int | None = None,
               flow: str | None = None) -> dict:
        """flow: 'issue' | 'text'. Mac dinh suy ra: co query_rep/query_idx
        (issue co san) -> 'issue', chi co van ban -> 'text'."""
        if not self._built:
            raise RuntimeError("goi .build() truoc khi search")
        cfg = self.cfg
        top_k = top_k or cfg.decision.top_k
        flow = flow or ("issue" if query_rep is not None or query_idx is not None
                        else "text")
        plan = flow_plan(cfg, flow)
        rep_q = query_rep or self.extractor.extract_query(query_text, query_id)
        exclude = {self.key_to_idx[k] for k in (exclude_keys or set())
                   if k in self.key_to_idx}
        dense, sem_range = self._dense_for(plan["symmetric"])

        t0 = time.time()
        cands = self.gen.generate(rep_q, query_text, exclude=exclude,
                                  query_idx=query_idx, dense=dense,
                                  sem_range=sem_range)
        self.timing.candidate_ms.append((time.time() - t0) * 1000)

        t0 = time.time()
        scored: list[ScoredCandidate] = []
        for c in cands:
            rep_b = self.reps[c.idx]
            f = self.feat.extract(rep_q, rep_b, semantic=c.semantic_score,
                                  lexical=c.lexical_score)
            bd = self.scorer.score(f)
            scored.append(ScoredCandidate(
                idx=c.idx, key=c.key, doc=self.docs[c.idx], rep=rep_b,
                candidate=c, features=f, breakdown=bd, final_score=bd.final,
                rank_score=bd.rank_score))
        self.timing.feature_ms.append((time.time() - t0) * 1000)

        # Xep hang theo rank_score (bang chung duong); nhan quyet dinh theo
        # final_score (da tru mau thuan). Xem scoring.HybridScorer.score.
        scored.sort(key=lambda s: -s.rank_score)

        if plan["rerank"] != "none":
            t0 = time.time()
            scored = self._reranker(plan["rerank"]).rerank(query_text, scored)
            scored.sort(key=lambda s: -s.rank_score)
            self.timing.rerank_ms.append((time.time() - t0) * 1000)

        top = scored[:top_k]
        self.decider.decide(top)
        return {"queryId": query_id, "results": [self._fmt(s) for s in top],
                "n_candidates": len(cands), "plan": plan}

    def search_by_key(self, key: str, top_k: int | None = None) -> dict:
        """Tim bug tuong tu voi mot issue da co (loai chinh no khoi ket qua)."""
        i = self.key_to_idx.get(key)
        if i is None:
            raise KeyError(f"khong co issue {key} trong corpus")
        rep = self.reps[i]
        text = self.docs[i].get(self.cfg.embedding.text_field, "") \
            or self.docs[i].get("summary", "")
        return self.search(text, query_rep=rep, exclude_keys={key},
                           top_k=top_k, query_id=key, query_idx=i)

    def rank_keys(self, key: str, top_k: int) -> list[str]:
        """Chi tra danh sach key da xep hang -- dung cho evaluator."""
        i = self.key_to_idx.get(key)
        if i is None:
            return []
        rep = self.reps[i]
        text = self.docs[i].get(self.cfg.embedding.text_field, "") \
            or self.docs[i].get("summary", "")
        res = self.search(text, query_rep=rep, exclude_keys={key},
                          top_k=top_k, query_id=key, query_idx=i)
        return [r["bugId"] for r in res["results"]]

    # ------------------------------------------------------------ format
    def _fmt(self, s: ScoredCandidate) -> dict:
        bd = s.breakdown
        return {
            "bugId": s.key,
            "label": s.label,
            "score": round(s.final_score, 4),
            "rankScore": round(s.rank_score, 4),
            "semanticScore": round(bd.semantic, 4),
            "lexicalScore": round(bd.lexical, 4),
            "rerankScore": s.rerank_score,
            "positiveSignals": bd.positive_signals(
                self.cfg.decision.max_explanation_signals),
            "negativeSignals": bd.negative_signals(
                self.cfg.decision.max_explanation_signals),
            "explanation": s.explanation,
            "sources": s.candidate.sources if s.candidate else [],
            "summary": s.doc.get("summary", "")[:160],
            "url": s.doc.get("url", ""),
        }
