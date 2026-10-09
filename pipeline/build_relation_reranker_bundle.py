"""Build a self-contained, leakage-checked Colab bundle for relation reranking.

The script consumes only explicitly reviewed pairs.  It never treats an
unjudged corpus item as a negative.  Five anchors from the newer independent
family pool are selected by a label-blind stable hash and held out before
preferences are constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_cache
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl


SEED = 20260923
OLD = Path("data/processed/relation_codex_review_20260923/judgments.jsonl")
NEW = Path("data/processed/disagreement_review_pool_20260923/judgments.jsonl")
CORPUS = Path("data/processed/corpus.jsonl")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_hard_negative(row: dict) -> bool:
    adaptive = bool(row.get("adaptive_to_source_model") or row.get("adaptive_to_compared_models"))
    return int(row["grade"]) == 0 and (
        str(row.get("same_context", "")).lower() == "yes" or adaptive)


def make_preferences(rows: list[dict], docs: dict[str, dict]) -> list[dict]:
    """Use hard grade-0 negatives plus the ordinal grade-2 > grade-1 signal."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_key"]].append(row)
    result: list[dict] = []
    for query_key, judged in sorted(by_query.items()):
        pairs: list[tuple[dict, dict, float, float, str]] = []
        positives = [r for r in judged if int(r["grade"]) > 0]
        hard = [r for r in judged if is_hard_negative(r)]
        for positive in positives:
            for negative in hard:
                if int(positive["grade"]) == 2:
                    margin, importance = 1.0, 2.0
                else:
                    margin, importance = 0.5, 1.0
                pairs.append((positive, negative, margin, importance, "positive_over_hard_zero"))
        for positive in (r for r in judged if int(r["grade"]) == 2):
            for negative in (r for r in judged if int(r["grade"]) == 1):
                pairs.append((positive, negative, 0.75, 2.0, "grade2_over_grade1"))
        if not pairs:
            continue
        total_importance = sum(p[3] for p in pairs)
        for positive, negative, margin, importance, kind in pairs:
            result.append({
                "query_key": query_key,
                "positive_key": positive["candidate_key"],
                "negative_key": negative["candidate_key"],
                "query_text": docs[query_key]["text_dense"],
                "positive_text": docs[positive["candidate_key"]]["text_dense"],
                "negative_text": docs[negative["candidate_key"]]["text_dense"],
                "positive_grade": int(positive["grade"]),
                "negative_grade": int(negative["grade"]),
                "margin": margin,
                "query_weight": importance / total_importance,
                "preference_kind": kind,
                "label_source": "codex_ai_development_review",
                "human_verified": False,
            })
    return result


def make_pair_rows(rows: list[dict], docs: dict[str, dict], split: str) -> list[dict]:
    result = []
    for row in sorted(rows, key=lambda r: (r["query_key"], r["candidate_key"])):
        result.append({
            "split": split,
            "query_key": row["query_key"],
            "candidate_key": row["candidate_key"],
            "query_text": docs[row["query_key"]]["text_dense"],
            "candidate_text": docs[row["candidate_key"]]["text_dense"],
            "grade": int(row["grade"]),
            "same_context": str(row.get("same_context", "")),
            "reason": row.get("reason", ""),
            "confidence": row.get("confidence", ""),
            "family": row.get("anchor_family") or row.get("group_id") or row["query_key"],
            "label_source": row.get("label_source", "codex_ai_review"),
            "human_verified": bool(row.get("human_verified", False)),
        })
    return result


class FrozenDense:
    def __init__(self, emb: np.ndarray, texts: list[str]):
        self.emb = emb
        self.lookup = {text: i for i, text in enumerate(texts)}

    def encode(self, texts, is_query):
        if not is_query:
            raise AssertionError("FrozenDense is only used for cached issue queries")
        return self.emb[[self.lookup[text] for text in texts]]

    def score(self, query):
        return self.emb @ query


def candidate_pools(dev_rows: list[dict], docs_list: list[dict]) -> tuple[list[dict], dict]:
    texts = [d.get("text_dense", "") for d in docs_list]
    index = {d["key"]: i for i, d in enumerate(docs_list)}
    dense = DenseIndex("models/e5-base-task2", query_prefix="query: ", passage_prefix="query: ")
    identity = dense.cache_identity(texts)
    cache = embedding_cache.cache_path("data/cache", "corpus", identity)
    emb = embedding_cache.load(cache, identity)
    if emb is None:
        raise RuntimeError("Verified issue-flow embedding cache is missing; refusing silent recompute")
    engine = SearchEngine(docs_list, dense=FrozenDense(emb, texts), bm25_fold="none")
    engine.build_bm25()
    cfg = SearchConfig(top_k=100, candidate_k=100, name="hybrid")
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in dev_rows:
        labels[row["query_key"]][row["candidate_key"]] = int(row["grade"])
    pools = []
    for query_key in sorted(labels):
        qi = index[query_key]
        ranking = engine.search(texts[qi], cfg, exclude={qi})
        candidates = [{"key": docs_list[i]["key"], "text": texts[i]} for i in ranking]
        pools.append({
            "query_key": query_key,
            "query_text": texts[qi],
            "baseline_ranking": [c["key"] for c in candidates],
            "candidates": candidates,
            "judgments": labels[query_key],
        })
    return pools, {"identity": identity, "path": str(cache), "sha256": embedding_cache.file_hash(cache)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/relation_reranker_colab_20260923")
    ap.add_argument("--dev-anchors", type=int, default=5)
    ap.add_argument("--dev-offset", type=int, default=0)
    ap.add_argument("--extra-judgments", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output already exists: {out}")

    old = load_jsonl(str(OLD))
    new = load_jsonl(str(NEW))
    docs_list = load_jsonl(str(CORPUS))
    docs = {d["key"]: d for d in docs_list}
    new_queries = sorted({r["query_key"] for r in new},
                         key=lambda q: hashlib.sha256(f"{SEED}:{q}".encode()).hexdigest())
    dev_queries = set(new_queries[args.dev_offset:args.dev_offset + args.dev_anchors])
    if len(dev_queries) != args.dev_anchors:
        raise ValueError("dev-offset/dev-anchors exceed the independent-family pool")
    extra = [r for path in args.extra_judgments for r in load_jsonl(path)]
    if any(r["query_key"] in dev_queries for r in extra):
        raise AssertionError("Adaptive extra judgments may not enter the new dev split")
    train_rows = old + [r for r in new if r["query_key"] not in dev_queries] + extra
    dev_rows = [r for r in new if r["query_key"] in dev_queries]

    train_keys = {k for r in train_rows for k in (r["query_key"], r["candidate_key"])}
    dev_keys = {k for r in dev_rows for k in (r["query_key"], r["candidate_key"])}
    overlap = train_keys & dev_keys
    if overlap:
        raise AssertionError(f"Document leakage across train/dev: {sorted(overlap)[:5]}")
    for row in train_rows + dev_rows:
        if row["query_key"] not in docs or row["candidate_key"] not in docs:
            raise AssertionError(f"Missing corpus document in {row.get('review_id')}")

    train_preferences = make_preferences(train_rows, docs)
    dev_preferences = make_preferences(dev_rows, docs)
    pools, cache_info = candidate_pools(dev_rows, docs_list)
    out.mkdir(parents=True)
    dump_jsonl(out / "train_pairs.jsonl", make_pair_rows(train_rows, docs, "train"))
    dump_jsonl(out / "dev_pairs.jsonl", make_pair_rows(dev_rows, docs, "dev"))
    dump_jsonl(out / "train_preferences.jsonl", train_preferences)
    dump_jsonl(out / "dev_preferences.jsonl", dev_preferences)
    dump_jsonl(out / "dev_candidate_pools.jsonl", pools)

    train_queries = {r["query_key"] for r in train_rows}
    manifest = {
        "seed": SEED,
        "split_rule": (f"anchors [{args.dev_offset}:{args.dev_offset + args.dev_anchors}] after sorting "
                       "by sha256('20260923:'+query_key) in the independent-family pool"),
        "train_queries": len(train_queries),
        "dev_queries": len(dev_queries),
        "dev_query_keys": sorted(dev_queries),
        "train_judged_pairs": len(train_rows),
        "dev_judged_pairs": len(dev_rows),
        "train_preferences": len(train_preferences),
        "dev_preferences": len(dev_preferences),
        "train_preference_queries": len({r["query_key"] for r in train_preferences}),
        "dev_preference_queries": len({r["query_key"] for r in dev_preferences}),
        "train_grade_counts": dict(sorted(Counter(int(r["grade"]) for r in train_rows).items())),
        "dev_grade_counts": dict(sorted(Counter(int(r["grade"]) for r in dev_rows).items())),
        "train_hard_negatives": sum(is_hard_negative(r) for r in train_rows),
        "dev_hard_negatives": sum(is_hard_negative(r) for r in dev_rows),
        "document_overlap_train_dev": [],
        "unjudged_used_as_negative": False,
        "human_verified": False,
        "extra_adaptive_judgments": len(extra),
        "intended_use": "development training and held-out research gate only; not independent test evidence",
        "base_model": "BAAI/bge-reranker-base",
        "retrieval_cache": cache_info,
        "sources_sha256": {str(p): sha256(Path(p)) for p in
                            (OLD, NEW, CORPUS, Path(__file__), *map(Path, args.extra_judgments))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
