# -*- coding: utf-8 -*-
"""
Chay 4 cau hinh cua de tai tren corpus Viettel va in bang so sanh.

    python pipeline/run_ablation.py                      # ca 4 cau hinh
    python pipeline/run_ablation.py --configs keyword semantic
    python pipeline/run_ablation.py --limit 50           # chay thu nhanh

Bo danh gia:
  pairs  -- 199 cap "trung/tuong tu" ky su tu ghi trong comment (nhan THAT).
            Moi cap sinh 2 truy van (a->b, b->a), loai chinh no khoi ket qua.
            Day la bo do dung muc tieu cua de tai: hai issue KHAC NHAU cung
            ban chat ky thuat, do hai nguoi viet bang hai bo tu vung.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Gioi han luong + tat song song tokenizer: hai model cung mot tien trinh
# tren CPU rat de vot RAM va bi ket thuc khong traceback.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,
                       SearchEngine, Timing, evaluate, load_jsonl)

E5_QUERY = "query: "
E5_PASSAGE = "passage: "


def build_pair_queries(corpus: list[dict], pairs_path: str,
                       query_field: str = "text_dense") -> list[dict]:
    """Moi cap gold sinh 2 truy van (a->b, b->a), loai chinh no khoi ket qua.

    query_field quyet dinh KIEU truy van, va hai kieu do hai thu khac nhau:
      text_dense -- ca noi dung issue. Bai toan "phat hien trung lap".
                    Truy van dai => nhieu token hiem => co loi cho BM25.
      summary    -- mot dong mo ta. Dung luong "ky su mo ta van de dang gap"
                    trong de tai, va la phep thu that cho ngu nghia.
    """
    by = {d["key"]: d for d in corpus}
    out = []
    if not os.path.exists(pairs_path):
        return out
    for p in load_jsonl(pairs_path):
        a, b = p.get("a"), p.get("b")
        if a in by and b in by:
            for src, tgt in ((a, b), (b, a)):
                txt = by[src].get(query_field, "").strip()
                if txt:
                    out.append({"query": txt, "relevant": [tgt],
                                "exclude": [src], "cue": p.get("cue", ""),
                                "lexical_group": bool(by[tgt].get("lexical_group"))})
    return out


def report_by_group(engine, queries, cfg, ks=(1, 5, 10)):
    """Tach ket qua theo nhom truy van -- de tai cam ket phan tich nay."""
    from retrieval import evaluate as _ev
    groups = {
        "lexical (co ma loi/API/class)": [q for q in queries if q["lexical_group"]],
        "semantic (thuan nghiep vu)":    [q for q in queries if not q["lexical_group"]],
    }
    out = {}
    for name, qs in groups.items():
        if qs:
            out[name] = _ev(engine, qs, cfg, ks=ks)
    return out


def report_by_distance(engine, queries, cfg, ks=(1, 5, 10), near=200, far=1000):
    """Tach theo khoang cach so key giua truy van va dap an.

    Vi sao can: 79.7% cap gold co |dkey| < 200, tuc phan lon tap danh gia la
    cac issue tao gan nhau (cung dot test). Neu he thong chi gioi o nhom do
    thi no CHUA giai duoc bai toan that -- muc tieu de tai la tim TIEN LE CU
    ("loi nay da tung xay ra chua"), tuc cac cap cach nhau rat xa.
    """
    from retrieval import evaluate as _ev
    num = lambda k: int(k.split("-")[1])
    buckets = {f"gan (|dkey| < {near})": [], f"vua ({near}-{far})": [],
               f"xa (> {far})": []}
    for q in queries:
        try:
            d = abs(num(q["exclude"][0]) - num(q["relevant"][0]))
        except (IndexError, ValueError):
            continue
        key = (f"gan (|dkey| < {near})" if d < near
               else f"vua ({near}-{far})" if d <= far else f"xa (> {far})")
        buckets[key].append(q)
    return {k: _ev(engine, v, cfg, ks=ks) for k, v in buckets.items() if v}


CONFIGS = {
    "keyword":       dict(use_bm25=True,  use_dense=False, rerank=False),
    "semantic":      dict(use_bm25=False, use_dense=True,  rerank=False),
    "hybrid":        dict(use_bm25=True,  use_dense=True,  rerank=False),
    "hybrid+rerank": dict(use_bm25=True,  use_dense=True,  rerank=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--pairs", default="data/processed/dup_candidates_from_text.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--rerank-max-length", type=int, default=256)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0, help="gioi han so truy van")
    ap.add_argument("--query-field", default="text_dense",
                    choices=["text_dense", "summary", "text_business"],
                    help="text_dense = phat hien trung lap; summary = tra cuu mot dong")
    ap.add_argument("--by-group", action="store_true",
                    help="tach ket qua theo nhom lexical / semantic")
    ap.add_argument("--by-distance", action="store_true",
                    help="tach theo khoang cach so key (gan/vua/xa)")
    # e5: bai toan issue<->issue la DOI XUNG -> dung 'query: ' cho ca hai phia.
    # Cap 'query:'/'passage:' bat doi xung chi dung cho truy van ngan -> doan dai.
    ap.add_argument("--symmetric", action="store_true", default=True,
                    help="dung cung tien to hai phia (mac dinh: bat)")
    ap.add_argument("--asymmetric", dest="symmetric", action="store_false",
                    help="dung query:/passage: (cho truy van ngan -> doan dai)")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--out", default="data/processed/ablation.json")
    a = ap.parse_args()

    corpus = load_jsonl(a.corpus)
    print(f"corpus: {len(corpus)} issue")

    queries = build_pair_queries(corpus, a.pairs, query_field=a.query_field)
    print(f"kieu truy van: --query-field={a.query_field}")
    if a.limit:
        queries = queries[: a.limit]
    print(f"truy van (bo 'pairs'): {len(queries)}")
    if not queries:
        sys.exit("khong dung duoc truy van nao -- kiem tra file pairs")

    need_dense = any(CONFIGS[c]["use_dense"] for c in a.configs)
    need_rerank = any(CONFIGS[c]["rerank"] for c in a.configs)

    dense = None
    if need_dense:
        prefix_q, prefix_p = ("", "")
        if "e5" in a.model.lower():
            prefix_q = E5_QUERY
            prefix_p = E5_QUERY if a.symmetric else E5_PASSAGE
            print(f"  (tien to e5: query='{prefix_q}' passage='{prefix_p}'"
                  f" -- che do {'DOI XUNG' if a.symmetric else 'BAT DOI XUNG'})")
        dense = DenseIndex(a.model, device=a.device, batch_size=a.batch_size,
                           query_prefix=prefix_q, passage_prefix=prefix_p)

    reranker = None
    if need_rerank:
        try:
            import torch
            torch.set_num_threads(8)
        except Exception:
            pass
        reranker = CrossEncoderReranker(a.reranker, device=a.device,
                                        batch_size=min(a.batch_size, 16),
                                        max_length=a.rerank_max_length)

    eng = SearchEngine(corpus, dense=dense, reranker=reranker)

    t = time.time()
    print("build BM25 ...", end="", flush=True)
    eng.build_bm25()
    print(f" {time.time()-t:.1f}s  ({len(eng._bm25.vocab)} token)")

    if need_dense:
        t = time.time()
        print("build dense index (lan dau se lau, sau do doc cache) ...")
        eng.build_dense(show_progress=True)
        print(f"  xong {time.time()-t:.0f}s  shape={eng.dense.emb.shape}")

    rows, timings, groups, dists = [], {}, {}, {}
    for name in a.configs:
        cfg = SearchConfig(name=name, rrf_k=a.rrf_k, candidate_k=a.candidate_k,
                           rerank_k=a.rerank_k, top_k=10, **CONFIGS[name])
        tm = Timing()
        t = time.time()
        m = evaluate(eng, queries, cfg, ks=(1, 5, 10), timing=tm)
        m["config"] = name
        m["wall_s"] = round(time.time() - t, 1)
        rows.append(m)
        timings[name] = tm.summary()
        if a.by_group:
            groups[name] = report_by_group(eng, queries, cfg)
        if a.by_distance:
            dists[name] = report_by_distance(eng, queries, cfg)
        print(f"  {name:15s} recall@10={m['recall@10']:.4f} "
              f"mrr@10={m['mrr@10']:.4f}  ({m['wall_s']}s)")

    cols = ["config", "queries", "recall@1", "recall@5", "recall@10",
            "precision@10", "mrr@10", "ndcg@10"]
    w = [15, 7, 9, 9, 10, 12, 8, 8]
    line = "-" * (sum(w) + len(w))
    print("\n" + line)
    print("BANG SO SANH 4 CAU HINH")
    print(line)
    print("".join(c.ljust(x) for c, x in zip(cols, w)))
    print(line)
    base = rows[0] if rows else None
    for r in rows:
        print("".join(str(r.get(c, "")).ljust(x) for c, x in zip(cols, w)))
    print(line)
    if base and len(rows) > 1:
        print("\nMuc cai thien so voi cau hinh dau (%s):" % base["config"])
        for r in rows[1:]:
            for k in ("recall@10", "mrr@10", "ndcg@10"):
                d = r[k] - base[k]
                rel = (100 * d / base[k]) if base[k] else 0.0
                print(f"   {r['config']:15s} {k:10s} {d:+.4f} ({rel:+.1f}%)")

    print("\nLATENCY (ms, tach embedding / search / rerank)")
    print(f"  {'config':15s} {'emb p50':>8s} {'emb p95':>8s} {'srch p50':>9s} "
          f"{'srch p95':>9s} {'rrk p50':>8s} {'rrk p95':>8s}")
    for name in a.configs:
        s = timings[name]
        print(f"  {name:15s} {s['embed_p50']:8.1f} {s['embed_p95']:8.1f} "
              f"{s['search_p50']:9.2f} {s['search_p95']:9.2f} "
              f"{s['rerank_p50']:8.1f} {s['rerank_p95']:8.1f}")

    if groups:
        print("\nTHEO NHOM TRUY VAN (recall@10 / mrr@10)")
        gnames = list(next(iter(groups.values())))
        print(f"  {'config':15s} " + " ".join(g[:30].ljust(32) for g in gnames))
        for name in a.configs:
            cells = []
            for g in gnames:
                m = groups[name].get(g)
                cells.append((f"{m['recall@10']:.3f} / {m['mrr@10']:.3f} "
                              f"(n={m['queries']})").ljust(32) if m else "-".ljust(32))
            print(f"  {name:15s} " + " ".join(cells))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "reranker": a.reranker,
                   "query_field": a.query_field, "symmetric": a.symmetric,
                   "n_corpus": len(corpus), "n_queries": len(queries),
                   "results": rows, "latency_ms": timings,
                   "by_group": groups, "by_distance": dists}, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
