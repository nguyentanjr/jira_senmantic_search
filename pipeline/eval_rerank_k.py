# -*- coding: utf-8 -*-
"""So sanh rerank_k=30/20/10 tren DEV: chat luong va latency CPU.

Mac dinh dung dung model/cach tron cua pilot text. Diem CE top-30 duoc cache
theo tung query, nen chay lai se tiep tuc thay vi lam lai. Script KHONG sua
``pipeline/search.py``; output chi la bang de quyet dinh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paraphrase import ci, paired
from eval_rerankers import make_split, metrics_from_ranked
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig, SearchEngine,
                       fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS


SETS = {
    "paraphrase": ("data/processed/corpus_holdout.jsonl",
                   "data/processed/paraphrase_queries.jsonl"),
    "title": ("data/processed/corpus_title_holdout.jsonl",
              "data/processed/title_queries.jsonl"),
}
METRICS = ("recall@10", "mrr@10", "ndcg@10")


def load_dev_keys(clean_path: str, sample_path: str) -> set[str]:
    with open(clean_path, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(sample_path, encoding="utf-8") as f:
        sample = json.load(f)["paraphrase_clean_keys"]
    return set(make_split(clean, sample)["dev"])


def read_cache(path: str) -> dict[tuple[str, str, str], dict]:
    return {(r["set"], r["key"], r["qv"]): r for r in load_jsonl(path)} if os.path.exists(path) else {}


def percentile(values: list[float]) -> dict:
    return {"p50": round(float(np.percentile(values, 50)), 1),
            "p95": round(float(np.percentile(values, 95)), 1),
            "mean": round(float(np.mean(values)), 1)} if values else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--ks", nargs="+", type=int, default=[30, 20, 10])
    ap.add_argument("--model", default="models/e5-base-task2")
    ap.add_argument("--reranker", default="models/bge-reranker-base-task")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--cache", default="data/processed/rerank_k_dev_scores.jsonl")
    ap.add_argument("--out", default="data/processed/rerank_k_dev.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: so query moi set")
    ap.add_argument("--latency-queries", type=int, default=10)
    ap.add_argument("--ce-batch", type=int, default=64)
    ap.add_argument("--score-chunk", type=int, default=20,
                    help="so query-version gom vao mot lan CE de giam overhead CPU")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    ks = sorted(set(args.ks), reverse=True)
    if not ks or max(ks) > 30 or min(ks) < 1:
        raise SystemExit("--ks phai nam trong 1..30")

    wanted = load_dev_keys(args.clean_keys, args.dev_keys)
    cached = read_cache(args.cache)
    data = {}
    rr = CrossEncoderReranker(args.reranker, device=args.device,
                              batch_size=args.ce_batch, fp16=args.device == "cuda")
    shared_dense_model = None
    for set_index, set_name in enumerate(args.sets):
        corpus_path, query_path = SETS[set_name]
        corpus = load_jsonl(corpus_path)
        corpus_keys = {d["key"] for d in corpus}
        queries = [q for q in load_jsonl(query_path) if q["key"] in wanted]
        if args.limit:
            queries = queries[:args.limit]
        expected = [(set_name, q["key"], qv) for q in queries
                    for qv in ("goc", "khong_dau")]
        need_engine = (set_index == 0 and args.latency_queries > 0) or any(
            key not in cached for key in expected)
        eng = dense = None
        if need_engine:
            dense = DenseIndex(args.model, device=args.device, batch_size=64,
                               query_prefix="query: ", passage_prefix="passage: ")
            if shared_dense_model is not None:
                dense._model = shared_dense_model
            eng = SearchEngine(corpus, bm25_fold="auto", dense=dense,
                               reranker=rr)
            eng.build_bm25(); eng.build_dense(show_progress=False)
        cfg_hybrid = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])
        specs, pending = [], []
        for q in queries:
            rel = {k for k in q["relevant"] if k in corpus_keys}
            if not rel:
                continue
            for qv, text in (("goc", q["query"]),
                             ("khong_dau", strip_diacritics(q["query"]))):
                cache_key = (set_name, q["key"], qv)
                row = cached.get(cache_key)
                if row is None:
                    assert eng is not None
                    hybrid = eng.search(text, cfg_hybrid)
                    head = hybrid[:max(ks)]
                    row = {"set": set_name, "key": q["key"], "qv": qv,
                           "query": text, "relevant": sorted(rel),
                           "hybrid": [eng.keys[i] for i in hybrid],
                           "ce": [], "_head_docs": [corpus[i].get("text_dense", "") for i in head]}
                    pending.append((cache_key, row))
                specs.append((cache_key, rel))
        if dense is not None and dense._model is not None:
            shared_dense_model = dense._model
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        for start in range(0, len(pending), args.score_chunk):
            chunk = pending[start:start + args.score_chunk]
            pairs = [(row["query"], doc) for _, row in chunk for doc in row["_head_docs"]]
            scores = rr.score_pairs(pairs, show_progress=False)
            offset = 0
            with open(args.cache, "a", encoding="utf-8") as f:
                for cache_key, row in chunk:
                    n = len(row.pop("_head_docs"))
                    row["ce"] = [round(float(x), 6) for x in scores[offset:offset + n]]
                    offset += n
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cached[cache_key] = row
            print(f"cache {len(cached)} rows", flush=True)
        items = [(cached[cache_key], rel) for cache_key, rel in specs]
        data[set_name] = {"corpus": corpus, "engine": eng, "items": items,
                          "queries": queries}
        print(f"{set_name}: {len(items)//2} query x 2 ban", flush=True)

    arrays = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for set_name, block in data.items():
        for row, rel_keys in block["items"]:
            hybrid = np.asarray(row["hybrid"], dtype=object)
            ce = np.asarray(row["ce"], dtype=float)
            sl = f"{set_name}:{row['qv']}"
            for k in ks:
                ranked = np.concatenate([
                    fuse_rerank(hybrid[:k], ce[:k], "blend", 0.3), hybrid[k:]
                ])
                for metric, value in metrics_from_ranked(ranked, rel_keys).items():
                    arrays[sl][str(k)][metric].append(value)
    arrays = {sl: {k: {m: np.asarray(v) for m, v in metrics.items()}
                   for k, metrics in by_k.items()} for sl, by_k in arrays.items()}

    baseline = str(max(ks))
    quality, paired_vs_30 = {}, {}
    for sl, by_k in arrays.items():
        quality[sl] = {k: {m: ci(v, args.n_boot) for m, v in metrics.items()}
                       for k, metrics in by_k.items()}
        paired_vs_30[sl] = {
            k: {m: paired(metrics[m], by_k[baseline][m], args.n_boot) for m in METRICS}
            for k, metrics in by_k.items() if k != baseline
        }

    # Latency request-like: chi paraphrase:goc, thu tu k dao moi vong de giam bias warm cache.
    latency = {str(k): [] for k in ks}
    first = data[args.sets[0]]
    # Giong benchmark pilot: loai chi phi nap trong so/model khoi request dau.
    if args.latency_queries:
        first["engine"].dense.encode(["khởi động"], is_query=True)
        rr.score("khởi động", ["khởi động"])
    candidates = [x for x in first["items"] if x[0]["qv"] == "goc"][:args.latency_queries]
    rng = random.Random(20260915)
    for row, _ in candidates:
        order = ks[:]; rng.shuffle(order)
        for k in order:
            cfg = SearchConfig(name=f"hybrid+rerank-k{k}", top_k=10, rerank_k=k,
                               rerank_fusion="blend", rerank_weight=0.3,
                               **CONFIGS["hybrid+rerank"])
            start = time.perf_counter()
            first["engine"].search(row["query"], cfg)
            latency[str(k)].append((time.perf_counter() - start) * 1000)
            print(f"latency k={k}: {len(latency[str(k)])}/{len(candidates)}", flush=True)

    # Quy tac goi y dat truoc: chon k nho nhat co point delta R@10 va MRR@10
    # khong thap hon 0.01 tren bat ky slice nao. Van bao CI de nguoi dung quyet.
    eligible = {}
    for k in map(str, sorted(ks)):
        if k == baseline:
            eligible[k] = True
        else:
            eligible[k] = all(
                paired_vs_30[sl][k][m]["delta"] >= -0.01
                for sl in paired_vs_30 for m in ("recall@10", "mrr@10"))
    recommended = next((k for k in map(str, sorted(ks)) if eligible[k]), baseline)
    result = {"split": "dev", "sets": args.sets, "ks": ks,
              "model": args.model, "reranker": args.reranker,
              "fusion": "blend@0.3", "limit": args.limit,
              "n_queries": {s: len(d["items"]) // 2 for s, d in data.items()},
              "quality": quality, "paired_vs_k30": paired_vs_30,
              "latency_ms": {k: percentile(v) for k, v in latency.items()},
              "selection_rule": "smallest k with R@10 and MRR@10 point delta >= -0.01 on every dev slice",
              "eligible": eligible, "recommended": int(recommended),
              "changes_pilot_config": False}
    out = args.out if not args.limit else args.out.replace(".json", f"_smoke{args.limit}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for sl in quality:
        print(f"\n{sl}")
        for k in map(str, ks):
            x = quality[sl][k]
            print(f"  k={k}: " + " ".join(f"{m}={x[m]['mean']:.4f}" for m in METRICS))
    print(f"latency: {result['latency_ms']}")
    print(f"goi y DEV: rerank_k={recommended}; KHONG tu dong doi pilot")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
