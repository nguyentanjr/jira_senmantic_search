# -*- coding: utf-8 -*-
"""
Khoang tin cay bootstrap + kiem dinh cap doi cho bang so sanh cau hinh.

    python pipeline/significance.py --query-field summary

Vi sao can:
  Bo eval hien tai co 138 CAP gold -> 276 truy van (a->b va b->a).
  Hai truy van cua cung mot cap KHONG doc lap. Neu bootstrap theo truy van
  se danh gia qua cao do chac chan. Script nay resample theo CAP (cluster
  bootstrap), dung don vi doc lap that.

  Voi n nho, chenh lech vai diem phan tram rat de nam trong nhieu. Bao cao
  kem CI la cach duy nhat trung thuc -- va la thu hoi dong se hoi.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,
                       SearchEngine, load_jsonl)
from run_ablation import CONFIGS, E5_QUERY, build_pair_queries


def per_query_scores(engine, queries, cfg, ks=(1, 5, 10)):
    """-> {metric: np.array(per-query)} + mang id cap de resample theo cum."""
    maxk = max(ks)
    run = SearchConfig(**{**cfg.__dict__, "top_k": maxk})
    out = defaultdict(list)
    pair_ids = []
    for q in queries:
        rel = {engine.key_to_idx[k] for k in q["relevant"]
               if k in engine.key_to_idx}
        if not rel:
            continue
        exc = {engine.key_to_idx[k] for k in q.get("exclude", ())
               if k in engine.key_to_idx}
        ranked = engine.search(q["query"], run, exclude=exc)
        hits = [1.0 if i in rel else 0.0 for i in ranked[:maxk]]
        for k in ks:
            got = sum(hits[:k])
            out[f"recall@{k}"].append(got / len(rel))
            out[f"hit@{k}"].append(1.0 if got else 0.0)
            dcg = sum(v / np.log2(r + 2.0) for r, v in enumerate(hits[:k]))
            idcg = sum(1.0 / np.log2(r + 2.0) for r in range(min(len(rel), k)))
            out[f"ndcg@{k}"].append(dcg / idcg if idcg else 0.0)
        rr = 0.0
        for r, v in enumerate(hits):
            if v:
                rr = 1.0 / (r + 1)
                break
        out[f"mrr@{maxk}"].append(rr)
        pair_ids.append(q["pair_id"])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, \
        np.asarray(pair_ids)


def cluster_bootstrap(values: np.ndarray, pair_ids: np.ndarray, n_boot=5000,
                      seed=17):
    """Resample theo CAP, khong theo truy van."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pair_ids)
    idx_by_pair = {p: np.where(pair_ids == p)[0] for p in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_pair[p] for p in pick])
        means[b] = values[sel].mean()
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--pairs", default="data/processed/dup_candidates_from_text.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--query-field", default="text_dense")
    ap.add_argument("--configs", nargs="*", default=["keyword", "semantic", "hybrid"])
    ap.add_argument("--asymmetric", action="store_true",
                    help="query:/passage: -- dung khi truy van NGAN, doc DAI")
    ap.add_argument("--metrics", nargs="*",
                    default=["recall@10", "mrr@10", "ndcg@10"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="data/processed/significance.json")
    a = ap.parse_args()

    corpus = load_jsonl(a.corpus)
    queries = build_pair_queries(corpus, a.pairs, query_field=a.query_field)
    for i, q in enumerate(queries):
        q["pair_id"] = i // 2          # a->b va b->a cung mot cap
    print(f"corpus {len(corpus)} | {len(queries)} truy van "
          f"| {len(queries)//2} cap doc lap | query_field={a.query_field}")

    # Truy van ngan -> doc dai la bai toan BAT DOI XUNG (query:/passage:).
    # Truy van dai <-> doc dai la DOI XUNG (query: ca hai phia).
    # Dung sai khong bao loi, chi lam tut 7-19% -- nen chon tu dong.
    asym = a.asymmetric or a.query_field == "summary"
    passage_prefix = "passage: " if asym else E5_QUERY
    print(f"che do e5: {'BAT DOI XUNG' if asym else 'DOI XUNG'} "
          f"(query='{E5_QUERY}' passage='{passage_prefix}')")
    dense = DenseIndex(a.model, device="cpu", batch_size=32,
                       query_prefix=E5_QUERY, passage_prefix=passage_prefix)
    eng = SearchEngine(corpus, dense=dense)
    eng.build_bm25()
    if any(CONFIGS[c]["use_dense"] for c in a.configs):
        eng.build_dense(show_progress=False)

    scores, pair_ids = {}, None
    for name in a.configs:
        cfg = SearchConfig(name=name, top_k=10, **CONFIGS[name])
        scores[name], pair_ids = per_query_scores(eng, queries, cfg)
        print(f"  chay xong {name}")

    result = {"query_field": a.query_field, "n_pairs": int(len(np.unique(pair_ids))),
              "ci": {}, "paired": {}}

    print("\n" + "=" * 78)
    print("KHOANG TIN CAY 95%% (bootstrap theo cap, %d lan)" % a.n_boot)
    print("=" * 78)
    print(f"{'config':12s} " + " ".join(m.ljust(26) for m in a.metrics))
    for name in a.configs:
        cells = []
        for m in a.metrics:
            v = scores[name][m]
            bs = cluster_bootstrap(v, pair_ids, a.n_boot)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            result["ci"].setdefault(name, {})[m] = {
                "mean": round(float(v.mean()), 4),
                "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
            cells.append(f"{v.mean():.4f} [{lo:.4f},{hi:.4f}]".ljust(26))
        print(f"{name:12s} " + " ".join(cells))

    base = a.configs[0]
    print("\n" + "=" * 78)
    print(f"KIEM DINH CAP DOI so voi '{base}'  (chenh lech + CI + p hai phia)")
    print("=" * 78)
    for name in a.configs[1:]:
        for m in a.metrics:
            d = scores[name][m] - scores[base][m]
            bs = cluster_bootstrap(d, pair_ids, a.n_boot)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
            sig = "CO Y NGHIA" if lo > 0 or hi < 0 else "khong phan biet duoc"
            print(f"  {name:10s} {m:10s} {d.mean():+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]  p={p:.3f}  {sig}")
            result["paired"].setdefault(name, {})[m] = {
                "delta": round(float(d.mean()), 4),
                "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "p": round(float(p), 4), "significant": bool(lo > 0 or hi < 0)}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
