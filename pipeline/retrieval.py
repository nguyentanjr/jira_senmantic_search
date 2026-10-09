# -*- coding: utf-8 -*-
"""
Pipeline tim kiem issue: BM25 + dense + RRF + cross-encoder rerank.

Bon cau hinh trong de tai KHONG phai bon giai doan noi tiep -- chung la bon
ablation cua cung mot he thong:

    keyword         SearchConfig(use_bm25=True,  use_dense=False, rerank=False)
    semantic        SearchConfig(use_bm25=False, use_dense=True,  rerank=False)
    hybrid          SearchConfig(use_bm25=True,  use_dense=True,  rerank=False)
    hybrid+rerank   SearchConfig(use_bm25=True,  use_dense=True,  rerank=True)

Toi uu cho CPU:
  - embedding cache ra .npy, khoa theo (model, noi dung corpus) -> khong encode lai
  - vector chuan hoa san => cosine = dot product, mot phep nhan ma tran
  - BM25 dung inverted index numpy, khong vong lap python tren doc
  - rerank chi chay tren top-N ung vien
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

try:
    from . import embedding_cache
except ImportError:
    import embedding_cache

# ---------------------------------------------------------------- tokenizer

# Giu nguyen dinh danh ky thuat lam MOT token: ER80011, PS0006, trans_id,
# gw-staging.kdqt.vn, /api/v1/mini-app. Tokenizer mac dinh se bam nat chung,
# mat dung nhom tin hieu ma BM25 manh nhat.
_TOKEN_RE = re.compile(r"[A-Za-z0-9À-ỹ][A-Za-z0-9À-ỹ_.\-/]*")
_SPLIT_RE = re.compile(r"[_.\-/]+")
_VN_WORD_RE = re.compile(r"^[a-zà-ỹ]+$")


def strip_diacritics(text: str) -> str:
    """Bo dau tieng Viet nhu nguoi go khong dau: NFD, bo dau ket hop, d/D."""
    t = unicodedata.normalize("NFD", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", t.replace("đ", "d").replace("Đ", "D"))


FOLD_MODES = ("none", "add", "only")
# "auto" chi co o SearchEngine: hai chi muc BM25, chon theo truy van (xem SearchEngine).
ENGINE_FOLD_MODES = FOLD_MODES + ("auto",)


def tokenize(text: str, bigrams: bool = True, fold: str = "none") -> list[str]:
    """Token cho BM25.

    - giu nguyen dinh danh ghep VA phat them cac manh con (tang recall)
    - sinh bigram cho tu thuan tieng Viet: tieng Viet da am tiet, unigram
      am tiet mat ngu nghia ("dang nhap" -> "dang", "nhap")
    - fold: xu ly truy van KHONG DAU (phai dung CUNG mode cho doc va truy van)
        none  giu nguyen (hanh vi cu)
        add   token goc + ban bo dau cua token co dau -> khop ca hai kieu go,
              truy van co dau van duoc them diem khi dau trung
        only  bo dau toan bo ("mất" va "mật" thanh mot token)
    """
    if fold not in FOLD_MODES:
        raise ValueError(f"fold khong hop le: {fold}")
    if fold == "only":
        return tokenize(strip_diacritics(text), bigrams)
    if fold == "add":
        base = tokenize(text, bigrams)
        seen = set(base)
        return base + [t for t in tokenize(strip_diacritics(text), bigrams)
                       if t not in seen]
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    plain: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).strip("._-/")
        if not tok or len(tok) > 80:
            continue
        out.append(tok)
        if _SPLIT_RE.search(tok):
            for part in _SPLIT_RE.split(tok):
                if len(part) > 1:
                    out.append(part)
        if _VN_WORD_RE.match(tok):
            plain.append(tok)
    if bigrams:
        out.extend(a + "_" + b for a, b in zip(plain, plain[1:]))
    return out


# ---------------------------------------------------------------- BM25

class BM25:
    """BM25 Okapi tren inverted index numpy."""

    __slots__ = ("k1", "b", "n_docs", "doc_len", "avgdl", "idf",
                 "postings", "vocab")

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        self.doc_len = np.fromiter((len(d) for d in corpus_tokens),
                                   dtype=np.float32, count=self.n_docs)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        tmp: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(corpus_tokens):
            local: dict[str, int] = {}
            for t in toks:
                local[t] = local.get(t, 0) + 1
            for t, c in local.items():
                tmp.setdefault(t, {})[i] = c

        self.vocab = {t: j for j, t in enumerate(tmp)}
        self.postings: list[tuple[np.ndarray, np.ndarray]] = [None] * len(tmp)
        idf = np.empty(len(tmp), dtype=np.float32)
        N = self.n_docs
        for t, docs in tmp.items():
            j = self.vocab[t]
            ids = np.fromiter(docs.keys(), dtype=np.int32, count=len(docs))
            tfs = np.fromiter(docs.values(), dtype=np.float32, count=len(docs))
            order = np.argsort(ids)
            self.postings[j] = (ids[order], tfs[order])
            df = len(docs)
            idf[j] = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        self.idf = idf

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not query_tokens:
            return scores
        k1, b = self.k1, self.b
        norm = k1 * (1.0 - b + b * self.doc_len / (self.avgdl or 1.0))
        seen: dict[int, int] = {}
        for t in query_tokens:
            j = self.vocab.get(t)
            if j is not None:
                seen[j] = seen.get(j, 0) + 1
        for j, qtf in seen.items():
            ids, tfs = self.postings[j]
            contrib = self.idf[j] * (tfs * (k1 + 1.0)) / (tfs + norm[ids])
            if qtf > 1:
                contrib *= qtf
            scores[ids] += contrib
        return scores


# ---------------------------------------------------------------- dense

class DenseIndex:
    """Bi-encoder + cache. Vector chuan hoa => cosine = dot product."""

    def __init__(self, model_name: str, cache_dir: str = "data/cache",
                 device: str = "cpu", batch_size: int = 32,
                 query_prefix: str = "", passage_prefix: str = "",
                 max_seq_length: int | None = None):
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = None
        self._model_source = None
        self._loaded_model_identity = None
        self._uncached_session = uuid.uuid4().hex
        self._remote_unresolved = False
        self.emb: np.ndarray | None = None
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._loaded_model_identity = self._model_identity()
            self._model = SentenceTransformer(self._model_source or self.model_name,
                                              device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def _model_identity(self):
        if self._remote_unresolved:
            return {"remote": self.model_name, "session": self._uncached_session}
        source = self._model_source or self.model_name
        if not os.path.isdir(source):
            try:
                from huggingface_hub import snapshot_download
                source = snapshot_download(self.model_name, local_files_only=True)
            except (ImportError, OSError, ValueError):
                # An unresolved mutable remote name must never reuse a disk
                # cache from another instance. No network request for identity.
                self._remote_unresolved = True
                return {"remote": self.model_name, "session": self._uncached_session}
        self._model_source = os.path.realpath(source)
        return {"files_sha256": embedding_cache.model_hash(self._model_source)}

    def cache_identity(self, texts: Sequence[str]) -> dict:
        model_identity = self._model_identity()
        if (self._loaded_model_identity is not None
                and model_identity != self._loaded_model_identity):
            raise RuntimeError("Model files changed after loading; create a new DenseIndex")
        return embedding_cache.identity(texts, model_identity, self.query_prefix,
                                        self.passage_prefix, self.max_seq_length)

    def _cache_path(self, texts: Sequence[str], tag: str) -> str:
        return embedding_cache.cache_path(self.cache_dir, tag, self.cache_identity(texts))

    def encode(self, texts: Sequence[str], is_query: bool,
               show_progress: bool = False) -> np.ndarray:
        pre = self.query_prefix if is_query else self.passage_prefix
        payload = [pre + t for t in texts] if pre else list(texts)
        v = self.model.encode(payload, batch_size=self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=show_progress)
        return np.asarray(v, dtype=np.float32)

    def build(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        self.last_build_cache_hit = False
        metadata = self.cache_identity(texts)
        p = embedding_cache.cache_path(self.cache_dir, "corpus", metadata)
        self.emb = embedding_cache.load(p, metadata)
        if self.emb is not None:
            self.last_build_cache_hit = True
            return self.emb
        self.emb = self.encode(texts, is_query=False, show_progress=show_progress)
        # Recheck after inference so changing weights during a build cannot
        # publish a cache under the wrong identity.
        if self.cache_identity(texts) != metadata:
            raise RuntimeError("Embedding inputs changed during build; cache not written")
        embedding_cache.save(p, metadata, self.emb)
        return self.emb

    def score(self, query_vec: np.ndarray) -> np.ndarray:
        return self.emb @ query_vec


# ---------------------------------------------------------------- fusion

def rrf_fuse(rank_lists: Sequence[np.ndarray], n_docs: int, k: float = 60.0,
             weights: Sequence[float] | None = None) -> np.ndarray:
    """Reciprocal Rank Fusion. rank_lists: mang doc-id da xep hang."""
    fused = np.zeros(n_docs, dtype=np.float32)
    w = weights or [1.0] * len(rank_lists)
    for ranked, wi in zip(rank_lists, w):
        if len(ranked) == 0:
            continue
        ranks = np.arange(1, len(ranked) + 1, dtype=np.float32)
        fused[ranked] += wi / (k + ranks)
    return fused


def top_k_idx(scores: np.ndarray, k: int) -> np.ndarray:
    """argsort chi tren top-k (nhanh hon sort toan bo)."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def top_k_allowed(scores: np.ndarray, k: int,
                  allowed: set[int] | np.ndarray | None = None) -> np.ndarray:
    """Top-k chi trong tap tai lieu hop le.

    Day la pre-filter thuc su: BM25/dense lay ung vien tu tap da loc, thay vi
    tim tren toan corpus roi loc top-k (cach sau co the tra rong khi filter hep).
    """
    if allowed is None:
        return top_k_idx(scores, k)
    ids = (np.fromiter(allowed, dtype=np.int64, count=len(allowed))
           if isinstance(allowed, set) else np.asarray(allowed, dtype=np.int64))
    if len(ids) == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < scores.shape[0])]
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64)
    local = top_k_idx(scores[ids], min(k, len(ids)))
    return ids[local]


# ---------------------------------------------------------------- reranker

class CrossEncoderReranker:
    """Cross-encoder rerank top-N.

    Mac dinh dat cho CPU 16GB chay CHUNG tien trinh voi bi-encoder:
    e5-base (~1.1GB) + reranker (~1.1GB) + embedding matrix. Voi
    max_length=512 va batch_size=32 tren 16 luong, bo nho vot qua muc trong
    va tien trinh bi ket thuc KHONG traceback. 256/16 chay 46 cap/s, 1.9GB.
    Van ban rerank (summary + actual + expected) median ~300 ky tu nen 256
    token khong cat mat gi dang ke.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16,
                 max_length: int = 256, fp16: bool = False):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.fp16 = fp16
        self._m = None

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import CrossEncoder
            self._m = CrossEncoder(self.model_name, device=self.device,
                                   max_length=self.max_length)
            if self.fp16 and str(self.device).startswith("cuda"):
                self._m.model.half()
        return self._m

    def rerank(self, query: str, docs: Sequence[str],
               doc_ids: np.ndarray) -> np.ndarray:
        """Xep lai top-N.

        BAT BUOC boc trong inference_mode: goi predict lien tuc hang tram lan
        ma khong tat autograd thi bo nho tang dan theo vong lap va tien trinh
        bi ket thuc GIUA CHUNG, khong traceback (6 truy van chay duoc, 276 thi
        chet). Day la loi da gap that trong du an nay.
        """
        return self.rerank_with_scores(query, docs, doc_ids)[0]

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        """Diem cross-encoder theo DUNG thu tu `docs` dua vao."""
        return self.score_pairs([(query, d) for d in docs])

    def score_pairs(self, pairs: Sequence[tuple[str, str]],
                    show_progress: bool = False) -> np.ndarray:
        """Cham nhieu cap (query, doc) trong mot lan goi -- nhanh hon goi tung truy van."""
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float32)
        import torch
        with torch.inference_mode():
            s = self.model.predict(list(pairs), batch_size=self.batch_size,
                                   show_progress_bar=show_progress)
        return np.asarray(s, dtype=np.float32).reshape(-1)

    def rerank_with_scores(self, query: str, docs: Sequence[str],
                           doc_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nhu rerank, tra kem diem cross-encoder da xep giam dan (de phan tich)."""
        if len(doc_ids) == 0:
            return doc_ids, np.empty(0, dtype=np.float32)
        s = self.score(query, docs)
        order = np.argsort(-s, kind="stable")
        return doc_ids[order], s[order]


# ---------------------------------------------------------------- fusion

FUSIONS = ("replace", "rrf", "blend")


def fuse_rerank(head: np.ndarray, ce: np.ndarray, method: str = "replace",
                weight: float = 0.5, rrf_k: float = 60.0) -> np.ndarray:
    """Xep lai `head` (theo thu tu HYBRID) bang diem cross-encoder `ce` (cung thu tu).

    replace  chi theo cross-encoder (hanh vi cu)
    rrf      weight/(k + hang_ce) + (1-weight)/(k + hang_hybrid)
    blend    weight*minmax(ce) + (1-weight)*(1 - (hang_hybrid-1)/(n-1))
    weight = phan cua cross-encoder. Hoa diem giu thu tu hybrid (sort on dinh).
    Phan tich tung truy van (analyze_rerank_damage.py) cho thay replace day dap
    an hybrid hang 1 xuong o 28% (co dau) / 46% (khong dau) truy van.
    """
    if len(head) == 0:
        return head
    scores = rerank_fusion_scores(ce, method, weight, rrf_k)
    return head[np.argsort(-scores, kind="stable")]


def rerank_fusion_scores(ce: np.ndarray, method: str = "replace",
                         weight: float = 0.5, rrf_k: float = 60.0) -> np.ndarray:
    """Diem noi bo dung de xep hang head theo cach tron da chon."""
    n = len(ce)
    if n == 0 or method == "replace":
        return np.asarray(ce, dtype=np.float64)
    if method not in FUSIONS:
        raise ValueError(f"fusion khong hop le: {method}")
    rank_h = np.arange(1, n + 1, dtype=np.float64)
    if method == "rrf":
        rank_c = np.empty(n)
        rank_c[np.argsort(-ce, kind="stable")] = np.arange(1, n + 1)
        s = weight / (rrf_k + rank_c) + (1 - weight) / (rrf_k + rank_h)
    else:
        lo, hi = float(ce.min()), float(ce.max())
        c = (ce - lo) / (hi - lo) if hi - lo > 1e-9 else np.full(n, 0.5)
        h = 1 - (rank_h - 1) / (n - 1) if n > 1 else np.ones(1)
        s = weight * c + (1 - weight) * h
    return np.asarray(s, dtype=np.float64)


# ---------------------------------------------------------------- engine

@dataclass
class SearchConfig:
    use_bm25: bool = True
    use_dense: bool = True
    rerank: bool = False
    rrf_k: float = 60.0
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    candidate_k: int = 100      # so ung vien moi nhanh dua vao fusion
    rerank_k: int = 30          # chi rerank top-N (CPU: 30 la nguong hop ly)
    rerank_fusion: str = "replace"   # replace | rrf | blend (xem fuse_rerank)
    rerank_weight: float = 0.5
    rerank_rrf_k: float = 60.0
    top_k: int = 10
    name: str = "hybrid"


@dataclass
class Timing:
    embed_ms: list[float] = field(default_factory=list)
    search_ms: list[float] = field(default_factory=list)
    rerank_ms: list[float] = field(default_factory=list)

    def pct(self, xs: list[float], q: float) -> float:
        return float(np.percentile(xs, q)) if xs else 0.0

    def summary(self) -> dict:
        return {
            "embed_p50": round(self.pct(self.embed_ms, 50), 1),
            "embed_p95": round(self.pct(self.embed_ms, 95), 1),
            "search_p50": round(self.pct(self.search_ms, 50), 1),
            "search_p95": round(self.pct(self.search_ms, 95), 1),
            "rerank_p50": round(self.pct(self.rerank_ms, 50), 1),
            "rerank_p95": round(self.pct(self.rerank_ms, 95), 1),
        }


class SearchEngine:
    def __init__(self, docs: Sequence[dict], dense: DenseIndex | None = None,
                 reranker: CrossEncoderReranker | None = None,
                 bm25_field: str = "text_bm25", dense_field: str = "text_dense",
                 rerank_field: str = "text_dense", bm25_fold: str = "none"):
        if bm25_fold not in ENGINE_FOLD_MODES:
            raise ValueError(f"bm25_fold khong hop le: {bm25_fold}")
        self.bm25_fold = bm25_fold
        self.docs = list(docs)
        self.keys = [d["key"] for d in self.docs]
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.n = len(self.docs)
        self.bm25_field = bm25_field
        self.dense_field = dense_field
        self.rerank_field = rerank_field
        self.dense = dense
        self.reranker = reranker
        self._bm25: BM25 | None = None
        self._bm25_add: BM25 | None = None     # chi dung khi bm25_fold="auto"

    # -- build ------------------------------------------------------
    def build_bm25(self):
        """auto: chi muc `none` cho truy van CO DAU + chi muc `add` cho truy van KHONG DAU.

        Vi sao khong dung `add` cho moi truy van: truy van co dau bi dem moi tu
        tieng Viet hai lan (ban goc + ban bo dau). Voi truy van ngan, tu chung
        lan at ma loi / tu tieng Anh: 50 truy van viet tay, keyword 0,620 -> 0,440
        (eval_handwritten.py, 2026-09-14). Truy van dai (paraphrase) khong lo loi.
        """
        fold = "none" if self.bm25_fold == "auto" else self.bm25_fold
        self._bm25 = BM25([tokenize(d.get(self.bm25_field, ""), fold=fold) for d in self.docs])
        if self.bm25_fold == "auto":
            self._bm25_add = BM25([tokenize(d.get(self.bm25_field, ""), fold="add")
                                   for d in self.docs])
        return self._bm25

    def bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25_fold != "auto":
            return self._bm25.score(tokenize(query, fold=self.bm25_fold))
        if strip_diacritics(query) != query:
            return self._bm25.score(tokenize(query, fold="none"))
        return self._bm25_add.score(tokenize(query, fold="add"))

    def build_dense(self, show_progress: bool = True):
        texts = [d.get(self.dense_field, "") for d in self.docs]
        self.dense.build(texts, show_progress=show_progress)

    # -- search -----------------------------------------------------
    def search(self, query: str, cfg: SearchConfig, exclude: set[int] | None = None,
               timing: Timing | None = None,
               allowed: set[int] | np.ndarray | None = None,
               return_scores: bool = False,
               trace: dict | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Tim kiem, co the pre-filter bang doc id va tra diem xep hang noi bo.

        `allowed` duoc ap dung khi tao candidate cho tung retriever va khi fusion,
        nen khac voi post-filter top-k. `return_scores` phuc vu UI pilot; diem chi
        co y nghia tuong doi trong mot luot tim, khong phai xac suat.
        """
        rank_lists, weights = [], []
        source_scores: list[np.ndarray] = []
        if trace is not None:
            trace.clear()
            trace.update({"bm25": [], "dense": [], "union": [],
                          "fused": [], "rerank_head": [], "final": []})

        t0 = time.perf_counter()
        qv = None
        if cfg.use_dense:
            qv = self.dense.encode([query], is_query=True)[0]
        if timing is not None:
            timing.embed_ms.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        if cfg.use_bm25:
            s = self.bm25_scores(query)
            candidates = top_k_allowed(s, cfg.candidate_k, allowed)
            # BM25=0 means no matched token, not weak lexical evidence. Do not
            # give arbitrary padding documents RRF votes (especially for OOV
            # codes and narrow metadata filters). Keep the existing ordering
            # of positive matches; dense scores have different semantics.
            rank_lists.append(candidates[s[candidates] > 0])
            weights.append(cfg.bm25_weight)
            source_scores.append(s)
            if trace is not None:
                trace["bm25"] = rank_lists[-1].tolist()
        if cfg.use_dense:
            s = self.dense.score(qv)
            rank_lists.append(top_k_allowed(s, cfg.candidate_k, allowed))
            weights.append(cfg.dense_weight)
            source_scores.append(s)
            if trace is not None:
                trace["dense"] = rank_lists[-1].tolist()

        if len(rank_lists) == 1:
            ranked = rank_lists[0]
            ranking_scores = source_scores[0][ranked].astype(np.float64)
        else:
            fused = rrf_fuse(rank_lists, self.n, k=cfg.rrf_k, weights=weights)
            ranked = top_k_allowed(fused, cfg.candidate_k, allowed)
            # Fusion may have fewer candidates than candidate_k. Never pad the
            # union with documents which received no vote from any retriever.
            ranked = ranked[fused[ranked] > 0]
            ranking_scores = fused[ranked].astype(np.float64)

        if exclude:
            keep = np.array([i not in exclude for i in ranked], dtype=bool)
            ranked = ranked[keep]
            ranking_scores = ranking_scores[keep]
            # Exclusion must not silently reduce the candidate budget. Preserve
            # existing ranks/RRF votes: renumbering both branches after removing
            # the source changed good top-10 results in the paired audit.
            need = cfg.candidate_k - len(ranked)
            if need > 0:
                eligible = (np.arange(self.n) if allowed is None else
                            np.asarray(sorted(allowed) if isinstance(allowed, set) else allowed))
                spare_ids = eligible[~np.isin(eligible, list(exclude) + ranked.tolist())]
                scores = source_scores[0] if len(rank_lists) == 1 else fused
                spare = top_k_allowed(scores, need, spare_ids)
                if len(rank_lists) != 1 or cfg.use_bm25:
                    spare = spare[scores[spare] > 0]
                # Usually the existing hybrid union already contains enough
                # spare documents. If not, extend raw branch tails, retaining
                # their original prefixes and never treating BM25=0 as evidence.
                if len(spare) < need and len(rank_lists) > 1:
                    extended = []
                    for branch, raw in zip(rank_lists, source_scores):
                        extra = top_k_allowed(raw, cfg.candidate_k + len(exclude), allowed)
                        extra = extra[~np.isin(extra, branch)]
                        if cfg.use_bm25 and len(extended) == 0:
                            extra = extra[raw[extra] > 0]
                        extended.append(np.concatenate([branch, extra]))
                    rank_lists = extended
                    scores = rrf_fuse(extended, self.n, k=cfg.rrf_k, weights=weights)
                    spare = top_k_allowed(scores, need, spare_ids)
                    spare = spare[scores[spare] > 0]
                ranked = np.concatenate([ranked, spare])
                ranking_scores = np.concatenate([ranking_scores, scores[spare]])
        if trace is not None:
            branches = [[int(i) for i in ranks if not exclude or i not in exclude]
                        for ranks in rank_lists]
            # Single retrievers can backfill directly from their raw scores.
            if len(branches) == 1:
                branches[0] = ranked.tolist()
            branch_index = 0
            if cfg.use_bm25:
                trace["bm25"] = branches[branch_index]
                branch_index += 1
            if cfg.use_dense:
                trace["dense"] = branches[branch_index]
            trace["union"] = sorted({i for ranks in branches for i in ranks})
            trace["fused"] = ranked.tolist()
        if timing is not None:
            timing.search_ms.append((time.perf_counter() - t1) * 1000)

        if cfg.rerank and self.reranker is not None and len(ranked):
            t2 = time.perf_counter()
            head = ranked[: cfg.rerank_k]
            if trace is not None:
                trace["rerank_head"] = head.tolist()
            tail = ranked[cfg.rerank_k:]
            tail_scores = ranking_scores[cfg.rerank_k:]
            texts = [self.docs[i].get(self.rerank_field, "") for i in head]
            ce = self.reranker.score(query, texts)
            head_scores = rerank_fusion_scores(
                ce, cfg.rerank_fusion, cfg.rerank_weight, cfg.rerank_rrf_k)
            order = np.argsort(-head_scores, kind="stable")
            head = head[order]
            head_scores = head_scores[order]
            ranked = np.concatenate([head, tail])
            ranking_scores = np.concatenate([head_scores, tail_scores])
            if timing is not None:
                timing.rerank_ms.append((time.perf_counter() - t2) * 1000)

        ranked, ranking_scores = ranked[: cfg.top_k], ranking_scores[: cfg.top_k]
        if trace is not None:
            trace["final"] = ranked.tolist()
        return (ranked, ranking_scores) if return_scores else ranked


# ---------------------------------------------------------------- metrics

def evaluate(engine: SearchEngine, queries: Sequence[dict], cfg: SearchConfig,
             ks: Sequence[int] = (1, 5, 10), timing: Timing | None = None) -> dict:
    """queries: [{"query": str, "relevant": [key,...], "exclude": [key,...]}]"""
    maxk = max(ks)
    cfg_run = SearchConfig(**{**cfg.__dict__, "top_k": max(maxk, cfg.top_k)})
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg.update({f"hit@{k}": 0.0 for k in ks})
    agg.update({f"precision@{k}": 0.0 for k in ks})
    agg.update({f"ndcg@{k}": 0.0 for k in ks})
    agg[f"mrr@{maxk}"] = 0.0
    n = 0

    for q in queries:
        rel = {engine.key_to_idx[k] for k in q["relevant"] if k in engine.key_to_idx}
        if not rel:
            continue
        exc = {engine.key_to_idx[k] for k in q.get("exclude", ())
               if k in engine.key_to_idx}
        ranked = engine.search(q["query"], cfg_run, exclude=exc, timing=timing)
        n += 1
        hits = [1.0 if i in rel else 0.0 for i in ranked[:maxk]]
        for k in ks:
            h = hits[:k]
            got = sum(h)
            agg[f"recall@{k}"] += got / len(rel)
            agg[f"hit@{k}"] += 1.0 if got else 0.0
            agg[f"precision@{k}"] += got / k
            dcg = sum(v / np.log2(r + 2.0) for r, v in enumerate(h))
            idcg = sum(1.0 / np.log2(r + 2.0) for r in range(min(len(rel), k)))
            agg[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
        rr = 0.0
        for r, v in enumerate(hits):
            if v:
                rr = 1.0 / (r + 1)
                break
        agg[f"mrr@{maxk}"] += rr

    if n:
        for k in agg:
            agg[k] = round(agg[k] / n, 4)
    agg["queries"] = n
    return agg


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
