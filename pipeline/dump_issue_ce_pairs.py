"""Dong goi top-100 hybrid cua MOI issue trong corpus de cham cross-encoder offline.

Luong issue: truy van la mot issue CO SAN trong corpus, nen diem CE cua
(issue, ung vien top-100) tinh truoc duoc mot lan tren GPU (Colab), khi tim chi
tra bang -- thay cho >=13 s/truy van khi cham 100 cap tren CPU.

    python pipeline/dump_issue_ce_pairs.py --out colab/issue_ce_offline_bundle/data

Ung vien = dung top-100 hybrid production cua luong issue (e5-task2, 'query: '
hai phia, BM25 khong bo dau, RRF), cung cach `candidate_pools` da dung cho danh
gia 10 family (build_relation_reranker_bundle.py). Van ban cap = text_dense ca
hai phia, giong SearchEngine.rerank_field.

Dau ra:
  docs.jsonl      {"key", "text", "text_sha"} theo dung thu tu corpus
  pools.npy       int32 [n_docs, 100], chi so doc ung vien theo thu tu hybrid, -1 = trong
  manifest.json   van tay corpus/cache embedding/cau hinh truy xuat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embedding_cache                                              # noqa: E402
from ce_offline import text_sha                                     # noqa: E402
from build_relation_reranker_bundle import FrozenDense             # noqa: E402
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl  # noqa: E402

POOL_K = 100
MODEL = "models/e5-base-task2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out", default="colab/issue_ce_offline_bundle/data")
    ap.add_argument("--limit", type=int, default=0, help="chi dump N truy van dau (thu nhanh)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output exists: {out}")

    docs = load_jsonl(args.corpus)
    texts = [d.get("text_dense", "") for d in docs]
    dense = DenseIndex(MODEL, query_prefix="query: ", passage_prefix="query: ")
    identity = dense.cache_identity(texts)
    cache = embedding_cache.cache_path("data/cache", "corpus", identity)
    emb = embedding_cache.load(cache, identity)
    if emb is None:
        raise SystemExit("Thieu cache embedding luong issue da xac minh; khong tu nhung lai")
    engine = SearchEngine(docs, dense=FrozenDense(emb, texts), bm25_fold="none")
    engine.build_bm25()
    cfg = SearchConfig(top_k=POOL_K, candidate_k=POOL_K, name="hybrid")

    n_q = args.limit or len(docs)
    pools = np.full((n_q, POOL_K), -1, dtype=np.int32)
    t0 = time.perf_counter()
    for qi in range(n_q):
        ranked = engine.search(texts[qi], cfg, exclude={qi})
        pools[qi, :len(ranked)] = ranked
        if (qi + 1) % 1000 == 0:
            print(f"{qi + 1}/{n_q}  {time.perf_counter() - t0:.0f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "docs.jsonl", "w", encoding="utf-8") as f:
        for d, t in zip(docs, texts):
            f.write(json.dumps({"key": d["key"], "text": t, "text_sha": text_sha(t)},
                               ensure_ascii=False) + "\n")
    np.save(out / "pools.npy", pools)
    manifest = {
        "purpose": "issue-flow base cross-encoder scores, precomputed offline",
        "corpus": args.corpus,
        "corpus_sha256": embedding_cache.file_hash(args.corpus),
        "n_docs": len(docs),
        "n_queries": n_q,
        "pool_k": POOL_K,
        "pairs": int((pools >= 0).sum()),
        "short_pools": int((pools < 0).any(axis=1).sum()),
        "retrieval": {"model": MODEL, "prefix": "query: / query:", "bm25_fold": "none",
                      "fusion": "rrf", "rrf_k": cfg.rrf_k, "candidate_k": cfg.candidate_k,
                      "exclude_self": True},
        "embedding_cache": {"path": str(cache), "sha256": embedding_cache.file_hash(cache)},
        "pair_text": "text_dense (query) x text_dense (candidate)",
        "cross_encoder": {"model": "BAAI/bge-reranker-base", "max_length": 256,
                          "autocast": "float16 on cuda"},
        "dump_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
