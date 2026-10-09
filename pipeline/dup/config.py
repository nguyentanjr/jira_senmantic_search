# -*- coding: utf-8 -*-
"""Cau hinh thi nghiem -- MOI thu anh huong ket qua deu phai o day.

Muc tieu: benchmark tung phase ma KHONG sua source.

    python pipeline/run_dup_e2e.py --mode ablation
    python pipeline/run_dup_e2e.py --mode eval --set scoring.use_structured=false
    python pipeline/run_dup_e2e.py --mode eval --config configs/dup_default.json

Toan bo config serialize duoc ra JSON -> ghi kem ket qua de tai lap.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


# --------------------------------------------------------------- phase 1-2
@dataclass
class PreprocessConfig:
    lowercase: bool = True
    keep_operators: bool = True          # "=" / "<" thanh token rieng
    keep_numbers: bool = True            # "0" / "1" khong duoc gop
    keep_negation: bool = True
    bigrams: bool = True                 # tieng Viet da am tiet
    mask_issue_refs: bool = True         # chong ro ri nhan (xem docs/leakage)
    max_token_len: int = 80


@dataclass
class ExtractConfig:
    enabled: bool = True
    use_tags: bool = True                # [USSD EU][Show balance]
    use_sections: bool = True            # actual / expected
    use_signals: bool = True             # error_code / api_path / exception
    use_facets: bool = True              # team / root_cause / defect_type
    max_entities: int = 40


# ----------------------------------------------------------------- phase 3
@dataclass
class CandidateConfig:
    use_lexical: bool = True
    use_semantic: bool = True
    use_metadata: bool = True
    lexical_k: int = 100
    semantic_k: int = 100
    metadata_k: int = 50
    pool_k: int = 150                    # tong ung vien dua vao scoring
    rrf_k: float = 60.0
    lexical_weight: float = 1.0
    semantic_weight: float = 1.0
    metadata_weight: float = 0.5
    # blocking KHONG duoc hard-filter: chi dung de THEM ung vien
    metadata_min_idf: float = 2.0        # bo qua khoa qua pho bien


@dataclass
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-base"
    device: str = "cpu"
    batch_size: int = 32
    # bai toan bug<->bug la DOI XUNG -> cung tien to hai phia
    symmetric: bool = True
    # Luong "mo ta tu do" (search() khong kem issue): mo ta ngan -> issue dai
    # -> BAT DOI XUNG. Do 2026-09-12 (eval_dup_flows, paraphrase dev 500):
    # R@10 sym 0.524 vs asym 0.550, sym - asym -0.026 co y nghia.
    text_symmetric: bool = False
    text_field: str = "text_dense"
    cache_dir: str = "data/cache"
    # HIEU CHUAN COSINE -- bat buoc voi e5.
    # Do tren chinh corpus nay: cosine cua 20k cap NGAU NHIEN co
    # mean = p50 = 0.852. Tuc hai bug khong lien quan gi van dat 0.85.
    # De nguyen thi moi nguong tuyet doi deu vo nghia (tung do: top-10 cua
    # bat ky truy van nao cung >= 0.78 -> tat ca deu bi gan 'duplicate').
    # Chuan hoa tuyen tinh ve [0,1] theo nen corpus:
    #     sem = clip((cos - sem_lo) / (sem_hi - sem_lo), 0, 1)
    # Doi chieu sau khi chuan hoa: cap gold p50 = 0.947 -> 0.65,
    # cap ngau nhien p50 = 0.852 -> 0.00.
    calibrate: bool = True
    calibration_sample: int = 20000
    sem_lo: float = 0.0                  # 0 = tu do tu corpus khi build
    sem_hi: float = 1.0


# ----------------------------------------------------------------- phase 4
@dataclass
class ScoringWeights:
    """Trong so cua HybridScorer.

    base   = semantic/lexical (bang chung "hai van ban noi cung mot chuyen")
    bonus  = bang chung DUONG co cau truc
    penalty= bang chung AM (mau thuan)

    Nguyen tac: penalty cua tin hieu mau thuan THAT (operator/number/negation/
    status/action) phai du lon de keo mot cap semantic 0.95 xuong duoi nguong
    duplicate; penalty cua metadata (bank/provider/channel) phai NHO vi khac
    provider van co the cung root cause.
    """
    # Chon bang `--mode tune` TREN TAP TRAIN (cac cum trung lap rieng biet
    # voi tap eval), khong phai tren tap dung de bao cao. Luoi 0.0..1.0:
    # w_sem=0.8 cho mrr@10 train = 0.461 (0.0 -> 0.362, 0.5 -> 0.429,
    # 1.0 -> 0.450). Xem data/processed/dup_tune.json.
    semantic: float = 0.80
    lexical: float = 0.20

    # -- bang chung duong
    function_match: float = 0.05
    module_match: float = 0.02
    domain_match: float = 0.02
    api_match: float = 0.05
    error_code_match: float = 0.08
    entity_overlap: float = 0.04
    field_overlap: float = 0.03
    action_overlap: float = 0.03
    provider_match: float = 0.02
    channel_match: float = 0.02
    expected_behavior_sim: float = 0.05
    observed_behavior_sim: float = 0.05

    # -- bang chung am (tru vao diem)
    operator_conflict: float = 0.30
    numeric_conflict: float = 0.22
    negation_conflict: float = 0.25
    status_conflict: float = 0.28
    action_conflict: float = 0.26
    identifier_conflict: float = 0.12
    entity_conflict: float = 0.16
    actor_conflict: float = 0.20         # EU / Agent / Super Agent
    channel_conflict: float = 0.15       # USSD / APP
    provider_conflict: float = 0.06      # NHE: co the cung root cause
    error_code_conflict: float = 0.10
    # Hien tuong khac han nhau. Dat = 0 de tro ve hanh vi E2E v1.
    symptom_conflict: float = 0.14


@dataclass
class ScoringConfig:
    use_semantic: bool = True
    use_lexical: bool = True
    use_structured: bool = True          # tat -> chi con semantic + lexical
    use_contradiction: bool = True       # tat -> bo toan bo penalty
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    # Cong tac hard-negative: cap co semantic cao MA co mau thuan manh
    # bi chan tran diem, khong bao gio tu dong len 'duplicate'.
    # TRAN cua tong bang chung duong. Bang chung co cau truc la de CHUNG
    # THUC, khong duoc tu no tao ra ket luan trung lap.
    # Loi tim thay khi soat tap eval: RO-10517 va RO-10520 cung module
    # [EU APP][Buy Luku-History] nhung khac hanh vi ("token trung
    # referenceNumber" vs "sms gui token sai"). base chi 0.75, nhung cong
    # them chuc nang + module + field + thao tac thi vuot 1.0 -> bi gan
    # 'duplicate' voi diem 1.000. Chan tong bonus lai thi no khong the mot
    # minh dua mot cap chi giong vua phai len nguong trung lap.
    max_bonus: float = 0.15
    # Mau thuan co tru vao diem XEP HANG khong, hay chi doi NHAN?
    # Mac dinh: chi doi nhan. Xem giai thich trong scoring.HybridScorer.score.
    penalty_in_ranking: bool = False
    hard_negative_gate: bool = True
    hard_negative_min_sim: float = 0.70  # tren muc nay moi coi la "rat giong"
    hard_negative_cap: float = 0.72      # tran diem sau khi chan


# ----------------------------------------------------------------- phase 5
@dataclass
class RerankConfig:
    # enabled=true: ep dung `kind` cho MOI luong (che do thi nghiem cu,
    # vd ablation `hybrid_structured_rerank`). Mac dinh false -> theo luong.
    enabled: bool = False
    kind: str = "feature_weighted"       # none | feature_weighted | cross_encoder | sklearn
    # Reranker THEO LUONG (dung khi enabled=false). Do 2026-09-12, eval_dup_flows:
    #   text  : paraphrase dev 500, asym + cross_encoder R@10 0.550 -> 0.608,
    #           +0.058 [+0.034, +0.084]
    #   issue : gold train nDCG@10 -0.020 [-0.048, +0.007] (chon: tat);
    #           gold eval R@10 -0.049 [-0.094, -0.014]
    text: str = "cross_encoder"
    issue: str = "none"
    model: str = "BAAI/bge-reranker-base"
    model_path: str = "data/models/reranker.joblib"
    top_n: int = 30                      # chi rerank top-N ung vien
    blend: float = 0.5                   # 0 = giu hybrid, 1 = chi reranker
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 256                # cat bot de cross-encoder chay noi tren CPU


# ----------------------------------------------------------------- phase 6
@dataclass
class DecisionConfig:
    t_duplicate: float = 0.78
    t_related: float = 0.55
    # Bien "khong chac": nam sat nguong duplicate VA co tin hieu mau thuan
    # -> tra ve 'review' thay vi 'duplicate'. False positive dat hon FN.
    review_margin: float = 0.06
    top_k: int = 10
    max_explanation_signals: int = 4


# ------------------------------------------------------------------- root
@dataclass
class DupConfig:
    corpus_path: str = "data/processed/corpus.jsonl"
    gold_path: str = "data/processed/dup_candidates_from_text.jsonl"
    name: str = "hybrid_structured"
    seed: int = 13
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # ------------------------------------------------------------ tien ich
    def to_dict(self) -> dict:
        return asdict(self)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=1)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps() + "\n")

    def set_path(self, dotted: str, value: Any) -> None:
        """cfg.set_path("scoring.weights.semantic", 0.7) -- dung cho CLI --set."""
        parts = dotted.split(".")
        obj = self
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise KeyError(f"khong co truong cau hinh: {dotted}")
            obj = getattr(obj, p)
        last = parts[-1]
        if not hasattr(obj, last):
            raise KeyError(f"khong co truong cau hinh: {dotted}")
        cur = getattr(obj, last)
        setattr(obj, last, _coerce(value, type(cur)))


def _coerce(value: Any, want: type) -> Any:
    if isinstance(value, str):
        if want is bool:
            v = value.strip().lower()
            if v in ("true", "1", "yes", "on"):
                return True
            if v in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"gia tri bool khong hop le: {value}")
        if want is int:
            return int(value)
        if want is float:
            return float(value)
    return value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> DupConfig:
    """Doc config tu JSON (neu co) roi ap cac --set key=value."""
    cfg = DupConfig()
    if path:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = _merge(cfg, data)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set can dang key=value, nhan duoc: {ov}")
        k, v = ov.split("=", 1)
        cfg.set_path(k.strip(), v.strip())
    return cfg


def _merge(obj, data: dict):
    """Ghi de tung truong len dataclass da co, giu nguyen truong khong khai bao."""
    for k, v in data.items():
        if not hasattr(obj, k):
            raise KeyError(f"khong biet truong cau hinh: {k}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, _coerce(v, type(cur)) if cur is not None else v)
    return obj


# ------------------------------------------------------- cau hinh ablation
# Dung cho --mode ablation: moi preset TAT dan mot phase de do dong gop.
def ablation_presets(base: DupConfig | None = None) -> dict[str, DupConfig]:
    def mk(name: str, **muts) -> DupConfig:
        c = load_config()
        if base is not None:
            c = _merge(load_config(), json.loads(base.dumps()))
        c.name = name
        for k, v in muts.items():
            c.set_path(k, v)
        return c

    return {
        "lexical_only": mk(
            "lexical_only",
            **{"candidates.use_semantic": False, "scoring.use_semantic": False,
               "scoring.use_structured": False, "scoring.use_contradiction": False,
               "rerank.enabled": False}),
        "semantic_only": mk(
            "semantic_only",
            **{"candidates.use_lexical": False, "scoring.use_lexical": False,
               "scoring.use_structured": False, "scoring.use_contradiction": False,
               "rerank.enabled": False}),
        "hybrid": mk(
            "hybrid",
            **{"scoring.use_structured": False, "scoring.use_contradiction": False,
               "rerank.enabled": False}),
        "hybrid_structured": mk(
            "hybrid_structured", **{"rerank.enabled": False}),
        # Bien the: tru mau thuan vao ca DIEM XEP HANG (khong chi doi nhan).
        # Do de biet cai gia phai tra ve do phu khi lam sach top-K.
        "hybrid_structured_penalised_rank": mk(
            "hybrid_structured_penalised_rank",
            **{"rerank.enabled": False, "scoring.penalty_in_ranking": True}),
        "hybrid_structured_rerank": mk(
            "hybrid_structured_rerank",
            **{"rerank.enabled": True, "rerank.kind": "cross_encoder"}),
    }
