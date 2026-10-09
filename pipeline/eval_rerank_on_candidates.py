# -*- coding: utf-8 -*-
"""Cham mot reranker tren bo ung vien DA DONG BANG (`dump_candidates.py`).

Moi reranker deu duoc cham tren dung mot bo ung vien, nen chenh lech giua hai
reranker khong con lan voi chenh lech cua tang truy xuat.

Ngoai nDCG/recall, script bao cao CUU va PHA: bao nhieu tai lieu dung duoc keo
tu ngoai top-10 vao trong, va bao nhieu tai lieu dung dang o trong bi day ra.
Hai con so nay so sanh noi bo tren tai lieu DA CO NHAN nen khong dinh thien lech
cua phan chua ai cham - khac voi nDCG@10 tuyet doi, von bi uoc luong thap vi 98%
ung vien trong top-100 cua gold chua duoc cham. Xem [[issue-error-buckets]].
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import CrossEncoderReranker, load_jsonl  # noqa: F401
from eval_similar import ci, paired


def ndcg(keys: list, grade: dict, k: int = 10) -> float:
    dcg = sum((2 ** grade.get(x, 0) - 1) / math.log2(i + 2) for i, x in enumerate(keys[:k]))
    ideal = sorted(grade.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def metrics(keys: list, grade: dict, n_rel: int) -> dict:
    hit10 = sum(1 for x in keys[:10] if grade.get(x, 0) >= 1)
    return {"ndcg@10": ndcg(keys, grade),
            "recall@10": hit10 / n_rel if n_rel else 0.0,
            "precision@10": hit10 / 10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/processed/candidates_issue_dev.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--set", default="gold", help="gold | pooling_v4 | all")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--scores", default="",
                    help="JSONL cache diem CE: {query_key, doc_key, score}. Co thi "
                         "dung lai, khong thi cham roi ghi ra day.")
    ap.add_argument("--rerank-k", type=int, default=100,
                    help="cham lai top-N ung vien dau; phan con lai giu nguyen thu tu")
    ap.add_argument("--fusion", default="replace", choices=["replace", "blend"])
    ap.add_argument("--weight", type=float, default=0.5, help="chi dung khi fusion=blend")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="data/processed/rerank_on_candidates.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.candidates, encoding="utf-8") if l.strip()]
    if args.set != "all":
        rows = [r for r in rows if r["set"] == args.set]
    docs = {d["key"].upper(): d for d in load_jsonl(args.corpus)}
    print(f"{len(rows)} truy van | bo={args.set} | reranker={args.reranker}", flush=True)

    cache: dict[tuple[str, str], float] = {}
    if args.scores and os.path.exists(args.scores):
        for line in open(args.scores, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[(r["query_key"], r["doc_key"])] = float(r["score"])
        print(f"dung lai {len(cache)} diem CE tu {args.scores}", flush=True)

    need = [(r["query_key"], c["key"]) for r in rows
            for c in r["candidates"][:args.rerank_k] if (r["query_key"], c["key"]) not in cache]
    if need:
        ce = CrossEncoderReranker(args.reranker, device=args.device,
                                  max_length=args.max_length)
        print(f"can cham {len(need)} cap...", flush=True)
        done = 0
        fh = open(args.scores, "a", encoding="utf-8") if args.scores else None
        for r in rows:
            qtext = r["query_text"]
            todo = [c["key"] for c in r["candidates"][:args.rerank_k]
                    if (r["query_key"], c["key"]) not in cache]
            if not todo:
                continue
            texts = [docs[k].get("text_dense") or docs[k].get("summary", "") for k in todo]
            scores = ce.score(qtext, texts)
            for k, s in zip(todo, scores):
                cache[(r["query_key"], k)] = float(s)
                if fh:
                    fh.write(json.dumps({"query_key": r["query_key"], "doc_key": k,
                                         "score": float(s)}) + "\n")
            done += len(todo)
            if done % 2000 < len(todo):
                print(f"  {done}/{len(need)}", flush=True)
        if fh:
            fh.close()

    base_m: dict[str, list] = collections.defaultdict(list)
    new_m: dict[str, list] = collections.defaultdict(list)
    rescued = broken = 0
    rescued_rows, broken_rows = [], []
    for r in rows:
        grade = {c["key"]: (c["grade"] or 0) for c in r["candidates"]}
        n_rel = r["n_rel"]
        base_keys = [c["key"] for c in r["candidates"]]
        head = base_keys[:args.rerank_k]
        tail = base_keys[args.rerank_k:]
        if args.fusion == "replace":
            order = sorted(head, key=lambda k: -cache[(r["query_key"], k)])
        else:
            rank_of = {k: i for i, k in enumerate(head)}
            ce_order = sorted(head, key=lambda k: -cache[(r["query_key"], k)])
            ce_rank = {k: i for i, k in enumerate(ce_order)}
            w = args.weight
            order = sorted(head, key=lambda k: (1 - w) * rank_of[k] + w * ce_rank[k])
        new_keys = order + tail
        for name, keys, acc in (("base", base_keys, base_m), ("new", new_keys, new_m)):
            for k, v in metrics(keys, grade, n_rel).items():
                acc[k].append(v)
        b10, n10 = set(base_keys[:10]), set(new_keys[:10])
        for k in n10 - b10:
            if grade.get(k, 0) >= 1:
                rescued += 1
                rescued_rows.append({"query_key": r["query_key"], "doc_key": k,
                                     "grade": grade[k],
                                     "old_rank": base_keys.index(k) + 1,
                                     "new_rank": new_keys.index(k) + 1})
        for k in b10 - n10:
            if grade.get(k, 0) >= 1:
                broken += 1
                broken_rows.append({"query_key": r["query_key"], "doc_key": k,
                                    "grade": grade[k],
                                    "old_rank": base_keys.index(k) + 1,
                                    "new_rank": new_keys.index(k) + 1})

    out = {"set": args.set, "reranker": args.reranker, "fusion": args.fusion,
           "weight": args.weight, "rerank_k": args.rerank_k,
           "n_queries": len(rows), "metrics": {},
           "rescued": rescued, "broken": broken,
           "rescued_rows": rescued_rows, "broken_rows": broken_rows}
    print(f"\n{'chi so':14s} {'nen':>8s} {'sau rerank':>11s} {'delta':>22s}")
    for k in ("ndcg@10", "recall@10", "precision@10"):
        a = np.asarray(new_m[k]); b = np.asarray(base_m[k])
        mean, lo, hi = ci(a, args.n_boot)
        d, dlo, dhi, sig = paired(a, b, args.n_boot)
        out["metrics"][k] = {"base": round(float(b.mean()), 4), "new": round(mean, 4),
                             "delta": round(d, 4), "delta_lo": round(dlo, 4),
                             "delta_hi": round(dhi, 4), "significant": sig}
        print(f"{k:14s} {b.mean():8.4f} {mean:11.4f}   "
              f"{d:+.4f} [{dlo:+.4f},{dhi:+.4f}]{' *' if sig else ''}")
    print(f"\nCUU (dap an dung tu ngoai top-10 vao trong): {rescued}")
    print(f"PHA (dap an dung dang trong top-10 bi day ra): {broken}")
    print(f"can bang rong: {rescued - broken:+d}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
