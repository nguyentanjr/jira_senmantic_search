# -*- coding: utf-8 -*-
"""
Ghi diem cross-encoder cua top-k HYBRID theo tung truy van (co dau + khong dau),
de thu cac cach tron rerank OFFLINE (select_rerank_fusion.py) ma khong chay lai model.

    python pipeline/dump_rerank_scores.py --split dev  --device cuda
    python pipeline/dump_rerank_scores.py --split test --device cuda

Cau hinh = luong text hien tai cua search.py: e5 bat doi xung, BM25 fold="add",
hybrid top-100, bge-reranker-base cho top-30, max_length 256.

Moi dong: key, qv (goc|khong_dau), relevant, jac_bin, lexical_group,
hybrid (100 key theo thu tu hybrid), ce (30 diem, CUNG thu tu hybrid[:30]).

File test duoc ghi o day nhung select_rerank_fusion.py chi doc no SAU KHI da
ghi file lua chon tu dev.

Dau ra: data/processed/rerank_scores_<split>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_rerankers import make_split                                    # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,   # noqa: E402
                       SearchEngine, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS, E5_QUERY                               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--bm25-fold", default="add")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or f"data/processed/rerank_scores_{args.split}.jsonl"

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        wanted = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])[args.split])
    queries = [q for q in load_jsonl(args.queries) if q["key"] in wanted]
    if args.limit:
        queries = queries[: args.limit]

    eng = SearchEngine(load_jsonl(args.corpus), bm25_fold=args.bm25_fold,
                       dense=DenseIndex(args.model, device=args.device, batch_size=64,
                                        query_prefix=E5_QUERY, passage_prefix="passage: "))
    eng.build_bm25()
    eng.build_dense(show_progress=True)
    rr = CrossEncoderReranker(args.reranker, device=args.device, batch_size=args.batch_size)
    cfg = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])

    t0, n = time.perf_counter(), 0
    with open(out, "w", encoding="utf-8") as f:
        for j, q in enumerate(queries):
            rel = [k for k in q["relevant"] if k in eng.key_to_idx]
            if not rel:
                continue
            for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                hy = eng.search(text, cfg)
                head = hy[: args.rerank_k]
                ce = rr.score(text, [eng.docs[i].get(eng.rerank_field, "") for i in head])
                f.write(json.dumps({
                    "key": q["key"], "qv": qv, "relevant": rel,
                    "jac_bin": q.get("jac_bin"), "lexical_group": q.get("lexical_group"),
                    "hybrid": [eng.keys[i] for i in hy],
                    "ce": [round(float(s), 5) for s in ce],
                }, ensure_ascii=False) + "\n")
            n += 1
            if (j + 1) % 250 == 0:
                print(f"    {j + 1}/{len(queries)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"{args.split}: {n} truy van x 2 ban, rerank_k {args.rerank_k}, fold {args.bm25_fold} -> {out}")


if __name__ == "__main__":
    main()
