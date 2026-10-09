# -*- coding: utf-8 -*-
"""Dong bang tang truy xuat: luu top-K ung vien cho tung truy van phat trien.

Moi reranker sau nay deu duoc cham tren DUNG bo ung vien nay, nen chenh lech giua
hai reranker khong con lan voi chenh lech cua tang truy xuat. Day cung la tran
cung cua reranker: no chi sap xep lai, khong keo them tai lieu nao vao.

Moi dong JSONL la mot truy van:
    {"query_key", "query_text", "set", "n_rel",
     "candidates": [{"key", "rank", "score", "bm25", "dense", "grade"}]}

`grade` = nhan da co (gold: 1/2; pooling: 0/1/2), hoac null neu CHUA AI CHAM.
Tai lieu grade=null khong duoc coi la khong lien quan - xem
[[issue-error-buckets]]: pooling v4 chi gop top-10 cua 4 cau hinh cu.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl
from search import FLOWS


def load_gold(path: str) -> dict:
    rel = collections.defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            a, b, g = r["a"].upper(), r["b"].upper(), int(r["grade"])
            rel[a][b] = max(rel[a].get(b, 0), g)
            rel[b][a] = max(rel[b].get(a, 0), g)
    return rel


def load_pooling(path: str) -> dict:
    rel = collections.defaultdict(dict)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = (r.get("grade") or "").strip()
            if g in {"0", "1", "2"}:
                rel[r["query_key"].upper()][r["doc_key"].upper()] = int(g)
    return rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="data/processed/pooling_judgments_v4_llm.csv")
    ap.add_argument("--flow", default="issue", choices=["issue", "text"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="data/processed/candidates_issue_dev.jsonl")
    args = ap.parse_args()

    docs_list = load_jsonl(args.corpus)
    docs = {d["key"].upper(): d for d in docs_list}
    sets = {}
    if args.gold:
        sets["gold"] = load_gold(args.gold)
    if args.pooling:
        sets["pooling_v4"] = load_pooling(args.pooling)

    flow = FLOWS[args.flow]
    dense = DenseIndex(flow["model"], device=args.device, batch_size=args.batch_size,
                       query_prefix=flow["query_prefix"],
                       passage_prefix=flow["passage_prefix"])
    eng = SearchEngine(docs_list, dense=dense, reranker=None, bm25_fold=flow["bm25_fold"])
    eng.build_bm25()
    eng.build_dense(show_progress=False)
    cfg = SearchConfig(use_bm25=True, use_dense=True, rerank=False,
                       candidate_k=args.top_k, top_k=args.top_k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_rows = 0
    stats = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for set_name, rel_map in sets.items():
            for q, rel in rel_map.items():
                if q not in docs:
                    continue
                qtext = docs[q].get("text_dense") or docs[q].get("summary", "")
                ex = {eng.key_to_idx[q]} if q in eng.key_to_idx else set()
                ranked = eng.search(qtext, cfg, exclude=ex)
                # Diem tung retriever, de reranker sau co the tron lai.
                bm = eng.bm25_scores(qtext)
                dn = eng.dense.score(eng.dense.encode([qtext], is_query=True)[0])
                cands = []
                for pos, idx in enumerate(ranked, 1):
                    key = eng.keys[idx]
                    cands.append({"key": key, "rank": pos,
                                  "bm25": round(float(bm[idx]), 6),
                                  "dense": round(float(dn[idx]), 6),
                                  "grade": rel.get(key)})
                n_rel = sum(1 for g in rel.values() if g >= 1)
                hit = sum(1 for c in cands if (c["grade"] or 0) >= 1)
                stats[set_name + "/queries"] += 1
                stats[set_name + "/rel_total"] += n_rel
                stats[set_name + "/rel_in_topk"] += hit
                stats[set_name + "/unjudged_in_topk"] += sum(
                    1 for c in cands if c["grade"] is None)
                out.write(json.dumps(
                    {"query_key": q, "set": set_name, "query_text": qtext,
                     "n_rel": n_rel, "candidates": cands}, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"-> {args.out}: {n_rows} truy van, top-{args.top_k}")
    for s in sets:
        nq = stats[s + "/queries"]
        rel_t, rel_k = stats[s + "/rel_total"], stats[s + "/rel_in_topk"]
        unj = stats[s + "/unjudged_in_topk"]
        print(f"  {s:11s} {nq:3d} truy van | dap an dung lot top-{args.top_k}: "
              f"{rel_k}/{rel_t} = {rel_k / max(1, rel_t):.1%}  <- TRAN CUNG cua reranker")
        print(f"              tai lieu CHUA AI CHAM trong top-{args.top_k}: "
              f"{unj}/{nq * args.top_k} = {unj / max(1, nq * args.top_k):.1%}")


if __name__ == "__main__":
    main()
