# -*- coding: utf-8 -*-
"""
Do anh huong cua truy van KHONG DAU len luong mo ta tu do (engine chinh).

    python pipeline/eval_undiacritized.py                 # dev 500, CPU ~75 phut
    python pipeline/eval_undiacritized.py --limit 20      # chay thu

VI SAO
Ky su hay go khong dau ("khong dang nhap duoc"). Moi truy van paraphrase deu co
dau nen chua bo do nao phu truong hop nay. Chay thu pipeline dup cho thay e5
tra ve ket qua vo nghia voi truy van khong dau.

THIET KE
- Cung 500 truy van DEV, hai ban: goc va bo dau (`strip_diacritics`). Tai lieu
  giu nguyen (issue trong Jira van co dau).
- So cap doi TREN CUNG TRUY VAN: khong_dau - goc, cho keyword / semantic /
  hybrid / hybrid+rerank, tong the va jaccard <0.2.
- Chi dung DEV. Tap test 2.675 khong duoc cham.
- Chi DO. Khong dua cach sua nao vao day.

Dau ra: data/processed/undiacritized_dev.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_paraphrase import ci, paired                                  # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked, rerank_head  # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,   # noqa: E402
                       SearchEngine, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS, E5_QUERY                               # noqa: E402

METRICS = ("recall@10", "mrr@10", "ndcg@10")
CFGS = ("keyword", "semantic", "hybrid", "hybrid+rerank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="data/processed/undiacritized_dev.json")
    args = ap.parse_args()

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        dev = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])["dev"])
    corpus = load_jsonl(args.corpus)
    queries = [q for q in load_jsonl(args.queries) if q["key"] in dev]
    if args.limit:
        queries = random.Random(13).sample(queries, min(args.limit, len(queries)))

    eng = SearchEngine(corpus, dense=DenseIndex(args.model, device=args.device, batch_size=64,
                                                query_prefix=E5_QUERY,
                                                passage_prefix="passage: "))
    eng.build_bm25()
    eng.build_dense(show_progress=False)
    rr = CrossEncoderReranker(args.reranker, device=args.device)
    items = [(q, {eng.key_to_idx[k] for k in q["relevant"] if k in eng.key_to_idx})
             for q in queries]
    items = [(q, rel) for q, rel in items if rel]
    changed = sum(strip_diacritics(q["query"]) != q["query"] for q, _ in items)
    print(f"dev: {len(items)} truy van, {changed} truy van doi khi bo dau")

    scores = {v: {c: {m: [] for m in METRICS} for c in CFGS} for v in ("goc", "khong_dau")}
    cfg_hy = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])
    t0 = time.perf_counter()
    for j, (q, rel) in enumerate(items):
        for variant, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
            runs = {c: eng.search(text, SearchConfig(name=c, top_k=10, **CONFIGS[c]))
                    for c in ("keyword", "semantic")}
            hy = eng.search(text, cfg_hy)
            runs["hybrid"] = hy
            runs["hybrid+rerank"] = rerank_head(eng, rr, text, hy, args.rerank_k)
            for c, ranked in runs.items():
                for m, v in metrics_from_ranked(ranked, rel).items():
                    scores[variant][c][m].append(v)
        if (j + 1) % 50 == 0:
            print(f"    {j + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    arr = {v: {c: {m: np.asarray(x) for m, x in d.items()} for c, d in vv.items()}
           for v, vv in scores.items()}
    low = np.asarray([i for i, (q, _) in enumerate(items) if q.get("jac_bin") == "<0.2"])

    def block(idx):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        return {
            "n": len(items) if idx is None else int(len(idx)),
            "ci": {v: {c: {m: ci(sel(arr[v][c][m]), args.n_boot) for m in METRICS}
                       for c in CFGS} for v in arr},
            "khong_dau - goc": {c: {m: paired(sel(arr["khong_dau"][c][m]),
                                              sel(arr["goc"][c][m]), args.n_boot)
                                    for m in METRICS} for c in CFGS},
            "khong_dau: cau hinh - keyword": {
                c: {m: paired(sel(arr["khong_dau"][c][m]), sel(arr["khong_dau"]["keyword"][m]),
                              args.n_boot) for m in METRICS}
                for c in CFGS if c != "keyword"},
        }

    result = {"split": "dev", "n_queries": len(items), "n_changed": changed,
              "limit": args.limit, "model": args.model, "reranker": args.reranker,
              "rerank_k": args.rerank_k, "overall": block(None)}
    if len(low):
        result["jac_lt_0.2"] = block(low)

    for key in ("overall", "jac_lt_0.2"):
        if key not in result:
            continue
        b = result[key]
        print(f"\n== {key} (n={b['n']}) ==  R@10 / MRR@10 / nDCG@10")
        for c in CFGS:
            g = b["ci"]["goc"][c]
            k = b["ci"]["khong_dau"][c]
            print(f"  {c:<14} goc " + " ".join(f"{g[m]['mean']:.3f}" for m in METRICS)
                  + " | khong dau " + " ".join(f"{k[m]['mean']:.3f}" for m in METRICS))
        for title in ("khong_dau - goc", "khong_dau: cau hinh - keyword"):
            print(f"  -- {title}")
            for c, d in b[title].items():
                print(f"    {c:<14}" + "".join(
                    f"{x['delta']:+.4f}[{x['lo']:+.3f},{x['hi']:+.3f}]{'*' if x['significant'] else ' '} "
                    for x in d.values()))

    if args.limit:
        print("\n--limit: khong ghi file")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
