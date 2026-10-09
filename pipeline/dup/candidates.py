# -*- coding: utf-8 -*-
"""Phase 3 -- sinh ung vien tu NHIEU nguon, khong so sanh O(N^2).

Ba nguon doc lap, hop nhat bang RRF:

    lexical   BM25 tren token da giu toan tu/so/phu dinh
    semantic  bi-encoder (cosine tren vector chuan hoa)
    metadata  inverted index tren error_code / api / function / provider /
              module -- CHI de THEM ung vien.

Vi sao metadata khong duoc hard-filter: mot bug co the duoc viet o ngu canh
khac (khac module, khac partner) nhung cung root cause. Loc cung se lam mat
han nhom do. O day metadata chi bo phieu THEM, khong bao gio loai.

Diem tra ve cho phase 4 la diem THAT (cosine, BM25 chuan hoa), khong phai
hang RRF -- de scoring lam viec tren dai lien tuc.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._compat import BM25, DenseIndex, rrf_fuse, top_k_idx
from .extract import BugRepresentation


@dataclass
class Candidate:
    idx: int
    key: str
    sources: list[str] = field(default_factory=list)
    # -- diem dung cho SCORING (bi chan, so sanh duoc giua cac truy van)
    lexical_score: float = 0.0        # cosine tren vector token co trong so idf
    semantic_score: float = 0.0       # cosine da hieu chuan theo nen corpus
    # -- diem tho, chi dung de XEP UNG VIEN va de debug
    bm25_score: float = 0.0           # BM25 chia cho max cua truy van
    semantic_raw: float = 0.0         # cosine goc
    metadata_hits: list[str] = field(default_factory=list)
    fusion_score: float = 0.0


class CandidateGenerator:
    """Xay index mot lan, phuc vu nhieu truy van."""

    def __init__(self, docs: Sequence[dict], reps: Sequence[BugRepresentation],
                 cfg, pre, dense: DenseIndex | None = None):
        self.docs = list(docs)
        self.reps = list(reps)
        self.cfg = cfg
        self.pre = pre
        self.dense = dense
        self.keys = [d["key"] for d in self.docs]
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.n = len(self.docs)
        self._bm25: BM25 | None = None
        self._meta_index: dict[str, list[int]] = {}
        self._meta_idf: dict[str, float] = {}

    # ------------------------------------------------------------- build
    def build_lexical(self) -> None:
        toks = [self.pre.tokenize(self.pre.mask_issue_refs(
            d.get("text_bm25") or d.get("summary", ""))) for d in self.docs]
        self._bm25 = BM25(toks)
        self._build_tfidf(toks)

    def _build_tfidf(self, toks: Sequence[Sequence[str]]) -> None:
        """Vector token co trong so idf -- dung cho DIEM tu vung.

        Vi sao khong dung thang BM25 lam diem tu vung cho phase 4: BM25 khong
        bi chan tren va phu thuoc do dai truy van, nen phai chia cho max cua
        chinh truy van do -> tai lieu dung dau LUC NAO cung duoc 1.0, ke ca
        khi no khong lien quan. Nhu vay diem het so sanh duoc giua cac truy
        van va nguong tuyet doi mat y nghia.

        Cosine tren vector idf thi doi xung, bi chan [0,1], va van giu duoc
        uu the cua token hiem (ma loi, ten API) -- dung cho nhom lexical.
        """
        idf, vocab = self._bm25.idf, self._bm25.vocab
        self._vec: list[dict[int, float]] = []
        self._norm = np.zeros(self.n, dtype=np.float32)
        for i, ts in enumerate(toks):
            v: dict[int, float] = {}
            for t in set(ts):
                j = vocab.get(t)
                if j is not None:
                    v[j] = float(idf[j])
            self._vec.append(v)
            self._norm[i] = math.sqrt(sum(w * w for w in v.values())) or 1.0

    def _query_vec(self, tokens: Sequence[str]) -> tuple[dict[int, float], float]:
        if self._bm25 is None:
            return {}, 1.0
        idf, vocab = self._bm25.idf, self._bm25.vocab
        v: dict[int, float] = {}
        for t in set(tokens):
            j = vocab.get(t)
            if j is not None:
                v[j] = float(idf[j])
        return v, (math.sqrt(sum(w * w for w in v.values())) or 1.0)

    def tfidf_sim(self, qv: dict[int, float], qn: float, i: int) -> float:
        dv = self._vec[i]
        if not qv or not dv:
            return 0.0
        if len(qv) > len(dv):
            qv, dv = dv, qv
        dot = sum(w * dv[j] for j, w in qv.items() if j in dv)
        return float(dot / (qn * self._norm[i]))

    def build_semantic(self, show_progress: bool = True) -> None:
        if self.dense is None:
            return
        self.build_semantic_into(self.dense, show_progress)

    def build_semantic_into(self, dense: DenseIndex, show_progress: bool = True) -> None:
        """Nhung corpus vao mot DenseIndex bat ky (cung van ban, khac tien to)."""
        texts = [d.get(self.cfg_embed_field, "") or d.get("summary", "")
                 for d in self.docs]
        dense.build(texts, show_progress=show_progress)

    cfg_embed_field = "text_dense"
    # Hieu chuan cosine: mac dinh khong doi (se do lai o _fit_calibration).
    sem_lo: float = 0.0
    sem_hi: float = 1.0
    embed_cfg = None

    def _fit_calibration(self) -> None:
        """Do NEN va DINH cua diem tuong dong tren chinh corpus nay.

        Cung mot dinh nghia cho ca hai tin hieu, de chung nam CUNG mot thang
        va trung binh co trong so moi co nghia:

            lo = trung vi diem cua cap NGAU NHIEN   -> muc "khong lien quan"
            hi = phan vi 95 cua diem lang gieng gan nhat cua mot mau truy van
                 -> muc "rat giong"

        Vi sao bat buoc: cosine e5 tren corpus nay co nen 0.85 con cosine
        tf-idf co nen ~0.02. De nguyen thi 0.6*sem + 0.4*lex khong phai la
        "60% ngu nghia" ma la mot phep tron hai thuoc do khac don vi -- do
        chinh la ly do hybrid tung do TE HON ca semantic lan lexical rieng le.
        """
        e = self.embed_cfg
        if e is None or not getattr(e, "calibrate", True):
            return
        m = int(getattr(e, "calibration_sample", 20000))
        n_probe = min(200, self.n)

        # -- semantic
        emb = self.dense.emb if self.dense is not None else None
        if emb is not None and emb.shape[0] >= 50:
            self.sem_lo, self.sem_hi = self.semantic_calibration(emb)

        # -- lexical (tf-idf cosine). Truoc day hai buoc dung chung mot rng;
        # tua lai dung so luot rut cua buoc semantic de hieu chuan lexical
        # giu nguyen y het cac lan do cu.
        rng = np.random.RandomState(0)
        if emb is not None and emb.shape[0] >= 50 and not e.sem_lo:
            rng.randint(0, self.n, m), rng.randint(0, self.n, m)
            rng.choice(self.n, n_probe, replace=False)
        if self._bm25 is not None:
            i, j = rng.randint(0, self.n, m // 4), rng.randint(0, self.n, m // 4)
            vals = [self.tfidf_sim(self._vec[x], float(self._norm[x]), int(y))
                    for x, y in zip(i, j) if x != y]
            self.lex_lo = float(np.median(vals)) if vals else 0.0
            tops = []
            for q in rng.choice(self.n, n_probe, replace=False):
                s = self._bm25.score(
                    self.pre.tokenize(self.docs[q].get("text_bm25", "")))
                s[q] = -1
                cand = top_k_idx(s, 20)
                best = max((self.tfidf_sim(self._vec[q], float(self._norm[q]), int(c))
                            for c in cand), default=0.0)
                tops.append(best)
            self.lex_hi = float(np.percentile(tops, 95)) if tops else 1.0

    def semantic_calibration(self, emb: np.ndarray) -> tuple[float, float]:
        """(lo, hi) cho mot ma tran embedding corpus -- dung cho moi bo tien to."""
        e = self.embed_cfg
        if e is not None and e.sem_lo:
            return e.sem_lo, e.sem_hi
        rng = np.random.RandomState(0)
        m = int(getattr(e, "calibration_sample", 20000))
        n_probe = min(200, self.n)
        i, j = rng.randint(0, self.n, m), rng.randint(0, self.n, m)
        keep = i != j
        lo = float(np.median((emb[i[keep]] * emb[j[keep]]).sum(1)))
        probe = rng.choice(self.n, n_probe, replace=False)
        S = emb[probe] @ emb.T
        S[np.arange(n_probe), probe] = -1.0
        return lo, float(np.percentile(S.max(1), 95))

    lex_lo: float = 0.0
    lex_hi: float = 1.0

    @staticmethod
    def _rescale(x, lo: float, hi: float):
        if hi <= lo:
            return x
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    def calibrate(self, cos, sem_range: tuple[float, float] | None = None):
        lo, hi = sem_range or (self.sem_lo, self.sem_hi)
        return self._rescale(cos, lo, hi)

    def calibrate_lexical(self, sim):
        return self._rescale(sim, self.lex_lo, self.lex_hi)

    def build_metadata(self) -> None:
        """Inverted index tren cac khoa co cau truc."""
        inv: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.reps):
            for k in self._meta_keys(r):
                inv[k].append(i)
        n = max(1, self.n)
        self._meta_index = dict(inv)
        self._meta_idf = {k: math.log(n / len(v)) for k, v in inv.items()}

    @staticmethod
    def _meta_keys(r: BugRepresentation) -> list[str]:
        out = []
        out += ["err:" + e for e in r.error_codes]
        out += ["api:" + a for a in r.api]
        out += ["fn:" + f for f in r.function]
        out += ["mod:" + m for m in r.module]
        out += ["prov:" + p for p in r.provider]
        out += ["ent:" + e for e in list(r.fields_)[:20]]
        out += ["dom:" + d for d in r.domain if d]
        return out

    def build(self, show_progress: bool = True) -> "CandidateGenerator":
        if self.cfg.use_lexical:
            self.build_lexical()
        if self.cfg.use_semantic:
            self.build_semantic(show_progress=show_progress)
        if self.cfg.use_metadata:
            self.build_metadata()
        self._fit_calibration()
        return self

    # ------------------------------------------------------------ scores
    def lexical_scores(self, query_text: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(self.n, dtype=np.float32)
        s = self._bm25.score(self.pre.tokenize(query_text))
        m = float(s.max()) if s.size else 0.0
        return s / m if m > 0 else s          # chuan hoa theo truy van

    def semantic_scores(self, query_text: str,
                        query_vec: np.ndarray | None = None,
                        dense: DenseIndex | None = None) -> np.ndarray:
        dense = dense or self.dense
        if dense is None or dense.emb is None:
            return np.zeros(self.n, dtype=np.float32)
        if query_vec is None:
            query_vec = dense.encode([query_text], is_query=True)[0]
        return dense.score(query_vec)

    def metadata_scores(self, rep: BugRepresentation) -> tuple[np.ndarray, dict]:
        """Diem blocking = tong idf cua cac khoa chung (chuan hoa)."""
        s = np.zeros(self.n, dtype=np.float32)
        hits: dict[int, list[str]] = defaultdict(list)
        if not self._meta_index:
            return s, hits
        for k in self._meta_keys(rep):
            idf = self._meta_idf.get(k)
            if idf is None or idf < self.cfg.metadata_min_idf:
                continue
            docs = self._meta_index[k]
            if len(docs) > 300:            # khoa qua pho bien -> bo qua
                continue
            for i in docs:
                s[i] += idf
                if len(hits[i]) < 6:
                    hits[i].append(k)
        m = float(s.max()) if s.size else 0.0
        if m > 0:
            s = s / m
        return s, hits

    # --------------------------------------------------------- generate
    def generate(self, rep: BugRepresentation, query_text: str | None = None,
                 exclude: set[int] | None = None,
                 query_idx: int | None = None,
                 dense: DenseIndex | None = None,
                 sem_range: tuple[float, float] | None = None) -> list[Candidate]:
        """query_idx: chi so cua truy van neu no la mot issue CO SAN trong
        corpus -- khi do dung lai vector da encode thay vi encode lai. Vua
        nhanh hon nhieu lan, vua bao dam cung mot van ban cho ra cung mot
        vector. Chi hop le o che do doi xung (hai phia cung tien to).

        dense / sem_range: chi so dense khac (vd tien to 'passage: ' cho luong
        mo ta tu do) va khoang hieu chuan cua chinh no. Mac dinh: self.dense."""
        cfg = self.cfg
        exclude = exclude or set()
        qtext = query_text if query_text is not None else rep.text
        dense = dense or self.dense

        qvec = None
        if (query_idx is not None and dense is not None
                and dense.emb is not None
                and dense.query_prefix == dense.passage_prefix):
            qvec = dense.emb[query_idx]

        lex = self.lexical_scores(qtext) if cfg.use_lexical \
            else np.zeros(self.n, dtype=np.float32)
        sem = self.semantic_scores(qtext, qvec, dense) if cfg.use_semantic \
            else np.zeros(self.n, dtype=np.float32)
        meta, meta_hits = (self.metadata_scores(rep) if cfg.use_metadata
                           else (np.zeros(self.n, dtype=np.float32), {}))

        rank_lists, weights, src_of = [], [], []
        if cfg.use_lexical:
            rank_lists.append(top_k_idx(lex, cfg.lexical_k))
            weights.append(cfg.lexical_weight)
            src_of.append("lexical")
        if cfg.use_semantic:
            rank_lists.append(top_k_idx(sem, cfg.semantic_k))
            weights.append(cfg.semantic_weight)
            src_of.append("semantic")
        if cfg.use_metadata and meta.any():
            rank_lists.append(top_k_idx(meta, cfg.metadata_k))
            weights.append(cfg.metadata_weight)
            src_of.append("metadata")

        if not rank_lists:
            return []

        fused = rrf_fuse(rank_lists, self.n, k=cfg.rrf_k, weights=weights)
        pool = top_k_idx(fused, cfg.pool_k + len(exclude))

        src_sets = {name: set(rl.tolist()) for name, rl in zip(src_of, rank_lists)}
        qv, qn = self._query_vec(self.pre.tokenize(qtext)) if cfg.use_lexical \
            else ({}, 1.0)
        out: list[Candidate] = []
        for i in pool:
            i = int(i)
            if i in exclude:
                continue
            raw = float(sem[i])
            out.append(Candidate(
                idx=i, key=self.keys[i],
                sources=[nm for nm in src_of if i in src_sets[nm]],
                lexical_score=float(self.calibrate_lexical(
                    self.tfidf_sim(qv, qn, i))) if cfg.use_lexical else 0.0,
                semantic_score=float(self.calibrate(max(0.0, raw), sem_range)),
                bm25_score=float(lex[i]),
                semantic_raw=raw,
                metadata_hits=list(meta_hits.get(i, [])),
                fusion_score=float(fused[i]),
            ))
            if len(out) >= cfg.pool_k:
                break
        return out
