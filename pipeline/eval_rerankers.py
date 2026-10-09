# -*- coding: utf-8 -*-
"""
So sanh cac cross-encoder rerank tren bo PARAPHRASE, tach DEV / TEST.

    # 1. chon model tren dev (500 query da dung de chot luong rerank)
    python pipeline/eval_rerankers.py --split dev \
        --rerankers BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 --device cuda
    # 2. chay MOT lan tren test voi cau hinh da chot
    python pipeline/eval_rerankers.py --split test \
        --rerankers BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 --device cuda

VI SAO TACH DEV / TEST
500 query mau (`paraphrase_rerank_sample500.json`) da duoc dung de quyet dinh
"mo ta tu do -> hybrid+rerank". Neu tiep tuc dung chung de chon reranker thi so
tren 500 query do bi lac quan. Vi vay:
    dev  = 500 query mau        -> chon model / max_length
    test = 3.175 - 500 = 2.675  -> chay mot lan, lay so bao cao
Lan chay test ghi file ket qua; chay lai se bi tu choi tru khi --force.

HIEU QUA
keyword va hybrid chi chay mot lan; moi reranker chi xep lai top-k cua hybrid
(giong het `SearchEngine.search` voi rerank=True), nen so giua cac reranker la
so cap doi tren cung ung vien.

LOC DU LIEU TRAIN CROSS-ENCODER
`train_cross_encoder_clean.jsonl` chi loai issue cua gold eval va pooling. No
van chua 1.452/3.175 issue paraphrase sach (chu yeu easy_neg/hard_neg); van ban
cua cac issue do chua `actual` = chinh truy van. `filter_train_rows` bo moi
dong cham vao bat ky key dev/test nao.

Dau ra: data/processed/rerank_eval_<split>.json
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

from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,
                       SearchEngine, load_jsonl)
from run_ablation import CONFIGS, E5_QUERY
from eval_paraphrase import ci, paired

METRICS = ("recall@10", "mrr@10", "ndcg@10")


# ---------------------------------------------------------------- split

def make_split(clean_keys, dev_keys) -> dict:
    """dev = mau 500 da dung; test = phan con lai cua bo sach. Roi nhau."""
    clean = list(dict.fromkeys(clean_keys))
    dev = [k for k in dict.fromkeys(dev_keys)]
    missing = set(dev) - set(clean)
    if missing:
        raise ValueError(f"{len(missing)} key dev khong nam trong bo sach")
    dev_set = set(dev)
    test = [k for k in clean if k not in dev_set]
    return {"dev": sorted(dev), "test": sorted(test)}


def filter_train_rows(rows, eval_keys) -> list[dict]:
    """Bo dong train co a_key hoac b_key thuoc tap danh gia."""
    ek = set(eval_keys)
    return [r for r in rows if r["a_key"] not in ek and r["b_key"] not in ek]


# ---------------------------------------------------------------- metrics

def metrics_from_ranked(ranked, rel: set) -> dict:
    hits = [1.0 if i in rel else 0.0 for i in ranked[:10]]
    got = sum(hits)
    dcg = sum(v / np.log2(r + 2.0) for r, v in enumerate(hits))
    idcg = sum(1.0 / np.log2(r + 2.0) for r in range(min(len(rel), 10)))
    rr = next((1.0 / (r + 1) for r, v in enumerate(hits) if v), 0.0)
    return {"recall@10": got / len(rel), "mrr@10": rr,
            "ndcg@10": dcg / idcg if idcg else 0.0}


def rerank_head(engine, reranker, query, ranked, k):
    head, tail = ranked[:k], ranked[k:]
    texts = [engine.docs[i].get(engine.rerank_field, "") for i in head]
    return np.concatenate([reranker.rerank(query, texts, head), tail])


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--rerankers", nargs="+", required=True,
                    help="model dau tien la moc so sanh (thuong bge-reranker-base)")
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="chay thu; khong ghi file test")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true",
                    help="cho phep chay lai tren test (ghi ro ly do trong bao cao)")
    args = ap.parse_args()

    out = args.out or f"data/processed/rerank_eval_{args.split}.json"
    if args.split == "test" and os.path.exists(out) and not args.force and not args.limit:
        sys.exit(f"{out} da ton tai: test chi chay mot lan. Dung --force neu that su can.")

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        dev = json.load(f)["paraphrase_clean_keys"]
    split = make_split(clean, dev)
    wanted = set(split[args.split])

    corpus = load_jsonl(args.corpus)
    queries = [q for q in load_jsonl(args.queries) if q["key"] in wanted]
    if args.limit:
        queries = random.Random(13).sample(queries, min(args.limit, len(queries)))
    print(f"split {args.split}: {len(queries)} truy van | corpus {len(corpus)} | "
          f"dev {len(split['dev'])} / test {len(split['test'])}")

    dense = DenseIndex(args.model, device=args.device, batch_size=64,
                       query_prefix=E5_QUERY, passage_prefix="passage: ")
    eng = SearchEngine(corpus, dense=dense)
    eng.build_bm25()
    print("  nhung corpus (doc cache neu co)...", flush=True)
    eng.build_dense(show_progress=True)

    cfg_kw = SearchConfig(name="keyword", top_k=100, **CONFIGS["keyword"])
    cfg_hy = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])

    items = []
    for q in queries:
        rel = {eng.key_to_idx[k] for k in q["relevant"] if k in eng.key_to_idx}
        if rel:
            items.append((q, rel, eng.search(q["query"], cfg_kw),
                          eng.search(q["query"], cfg_hy)))
    names = ["keyword", "hybrid"] + [f"rerank:{m}" for m in args.rerankers]
    scores = {n: {m: [] for m in METRICS} for n in names}
    for q, rel, kw, hy in items:
        for n, r in (("keyword", kw), ("hybrid", hy)):
            for m, v in metrics_from_ranked(r, rel).items():
                scores[n][m].append(v)

    speed = {}
    for model in args.rerankers:
        rr = CrossEncoderReranker(model, device=args.device,
                                  batch_size=args.batch_size,
                                  max_length=args.max_length)
        _ = rr.model
        t0, n = time.perf_counter(), f"rerank:{model}"
        for j, (q, rel, kw, hy) in enumerate(items):
            ranked = rerank_head(eng, rr, q["query"], hy, args.rerank_k)
            for m, v in metrics_from_ranked(ranked, rel).items():
                scores[n][m].append(v)
            if (j + 1) % 250 == 0:
                print(f"    {model}: {j + 1}/{len(items)}", flush=True)
        dt = time.perf_counter() - t0
        speed[model] = round(len(items) * args.rerank_k / dt, 1)
        print(f"  xong {model}: {speed[model]} cap/giay", flush=True)
        del rr

    scores = {n: {m: np.asarray(v) for m, v in d.items()} for n, d in scores.items()}
    kept = [q for q, *_ in items]
    low = np.asarray([i for i, q in enumerate(kept) if q.get("jac_bin") == "<0.2"])
    base_rr = f"rerank:{args.rerankers[0]}"
    comparisons = [(n, "keyword") for n in names if n != "keyword"]
    comparisons += [(n, "hybrid") for n in names if n.startswith("rerank:")]
    comparisons += [(n, base_rr) for n in names[3:]]

    def block(idx):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        return {
            "n": len(kept) if idx is None else int(len(idx)),
            "ci": {n: {m: ci(sel(scores[n][m]), args.n_boot) for m in METRICS}
                   for n in names},
            "paired": {f"{a} - {b}": {m: paired(sel(scores[a][m]), sel(scores[b][m]),
                                                args.n_boot) for m in METRICS}
                       for a, b in comparisons},
        }

    result = {"split": args.split, "n_queries": len(kept), "limit": args.limit,
              "model": args.model, "rerankers": args.rerankers,
              "rerank_k": args.rerank_k, "max_length": args.max_length,
              "device": args.device, "pairs_per_s": speed,
              "overall": block(None), "jac_lt_0.2": block(low if len(low) else None)}

    for title, key in (("TONG THE", "overall"), ("TRUNG TU <0.2", "jac_lt_0.2")):
        b = result[key]
        print(f"\n== {title} (n={b['n']}) ==")
        for n in names:
            print(f"  {n:<40}" + "".join(
                f"{m} {b['ci'][n][m]['mean']:.4f}  " for m in METRICS))
        for c, d in b["paired"].items():
            print(f"  {c:<60}" + "".join(
                f"{v['delta']:+.4f}[{v['lo']:+.3f},{v['hi']:+.3f}]{'*' if v['significant'] else ' '} "
                for v in d.values()))
    print("  (* = CI 95% khong chua 0; thu tu cot: " + ", ".join(METRICS) + ")")

    if args.limit and args.split == "test":
        print("\n--limit tren test: khong ghi file")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
