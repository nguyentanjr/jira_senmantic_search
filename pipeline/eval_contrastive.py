# -*- coding: utf-8 -*-
"""Do bo Jira contrastive (cung chu de, trieu chung nguoc chieu).

Chi cac dong ``approved=yes`` moi duoc do. Moi family phai nam tron trong mot
split; ``validate_contrastive.py`` kiem tra dieu nay. Mac dinh do ba baseline
va cau hinh dang deploy cua tung luong.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import CrossEncoderReranker, DenseIndex, SearchConfig, SearchEngine, load_jsonl
from run_ablation import CONFIGS
from search import FLOWS
from direction_filter import demote, doc_labels, llm_labels, load_mined
from validate_contrastive import TRAIN_PAIRS, YES, training_keys, validate


def approved_rows(path: str, split: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows
            if (r.get("approved") or "").strip().lower() in YES
            and (split == "all" or (r.get("split") or "").strip().lower() == split)]


def directions(rows: list[dict], flow: str, docs: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        for side, other in (("a", "b"), ("b", "a")):
            preferred = row[f"issue_{side}"].strip().upper()
            contrast = row[f"issue_{other}"].strip().upper()
            source = (row.get(f"source_{side}") or "").strip().upper()
            if flow == "text":
                query = (row.get(f"query_{side}") or "").strip()
                if not query:
                    continue
                exclude = []
            else:
                if not source:
                    continue
                query = docs[source].get("text_dense") or docs[source].get("summary", "")
                exclude = [source]
            out.append({
                "family_id": row.get("family_id", ""), "group_id": row["group_id"],
                "direction": side.upper(), "flow": flow, "query": query,
                "source_key": source, "preferred": preferred, "contrast": contrast,
                "exclude": exclude,
            })
    return out


def score_rankings(items: list[dict]) -> tuple[dict, list[dict]]:
    """Aggregate precomputed rows. Each row has ranked_keys besides direction fields."""
    details, group_ok = [], defaultdict(list)
    for item in items:
        ranked = item["ranked_keys"]
        positions = {key: i + 1 for i, key in enumerate(ranked)}
        pr, cr = positions.get(item["preferred"]), positions.get(item["contrast"])
        correct = pr is not None and (cr is None or pr < cr)
        both_visible = pr is not None and cr is not None
        row = {k: item.get(k, "") for k in
               ("family_id", "group_id", "direction", "flow", "source_key",
                "preferred", "contrast")}
        row.update({"preferred_rank": pr, "contrast_rank": cr,
                    "pairwise_correct": correct, "both_candidates_top100": both_visible,
                    "preferred_top10": pr is not None and pr <= 10})
        details.append(row)
        group_ok[item["group_id"]].append(correct)
    n = len(details)
    summary = {
        "directions": n,
        "pairwise_accuracy": round(sum(r["pairwise_correct"] for r in details) / n, 4) if n else None,
        "preferred_hit@100": round(sum(r["preferred_rank"] is not None for r in details) / n, 4) if n else None,
        "preferred_hit@10": round(sum(r["preferred_top10"] for r in details) / n, 4) if n else None,
        "both_candidates_top100": round(sum(r["both_candidates_top100"] for r in details) / n, 4) if n else None,
        "two_way_group_accuracy": (round(sum(all(v) for v in group_ok.values()) / len(group_ok), 4)
                                   if group_ok else None),
        "groups": len(group_ok),
    }
    return summary, details


def config_for(name: str, flow: str) -> SearchConfig:
    f = FLOWS[flow]
    actual = f["config"] if name == "deployed" else name
    if actual not in CONFIGS:
        raise ValueError(f"cau hinh khong hop le: {name}")
    return SearchConfig(name=name, top_k=100, rerank_k=30,
                        rerank_fusion=f["rerank_fusion"],
                        rerank_weight=f["rerank_weight"], **CONFIGS[actual])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="data/processed/contrastive_jira_review.csv")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--train-pairs", nargs="+", default=list(TRAIN_PAIRS))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--flows", nargs="+", choices=["text", "issue"], default=["text", "issue"])
    ap.add_argument("--configs", nargs="+",
                    default=["keyword", "semantic", "hybrid", "deployed"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/processed/contrastive_eval.json")
    ap.add_argument("--direction-filter", action="store_true",
                    help="ha diem ung vien nguoc chieu trong top --df-window")
    ap.add_argument("--df-window", type=int, default=10)
    ap.add_argument("--df-mined", default="",
                    help="tu dien chi chieu hoc tu nhan LLM, dung cho phia truy van")
    ap.add_argument("--df-labels", default="",
                    help="file nhan chieu do LLM gan; bo trong = dung luat tu vung")
    args = ap.parse_args()

    docs_list = load_jsonl(args.corpus)
    docs = {d["key"].upper(): d for d in docs_list}
    with open(args.review, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    errors = validate(all_rows, set(docs), training_keys(args.train_pairs), docs)
    if errors:
        raise SystemExit("Bo contrastive khong hop le:\n" + "\n".join(errors))
    rows = approved_rows(args.review, args.split)
    if not rows:
        raise SystemExit("Khong co nhom approved=yes trong split da chon; can ky su duyet truoc.")

    mined = load_mined(args.df_mined) if args.df_mined else None
    if not args.direction_filter:
        labels = {}
    elif args.df_labels:
        labels = llm_labels(args.df_labels)
        print(f"nhan chieu tu {args.df_labels}: {len(labels)} issue")
    else:
        labels = doc_labels(docs_list)
    output = {"review": args.review, "direction_filter": args.direction_filter, "split": args.split, "approved_groups": len(rows),
              "model": FLOWS["text"]["model"], "results": {}}
    shared_model = None
    for flow in args.flows:
        f = FLOWS[flow]
        needs_reranker = any(config_for(n, flow).rerank for n in args.configs)
        dense = DenseIndex(f["model"], device=args.device, batch_size=32,
                           query_prefix=f["query_prefix"], passage_prefix=f["passage_prefix"])
        if shared_model is not None:
            dense._model = shared_model
        reranker = (CrossEncoderReranker(f["reranker"], device=args.device)
                    if needs_reranker else None)
        engine = SearchEngine(docs_list, dense=dense, reranker=reranker,
                              bm25_fold=f["bm25_fold"])
        engine.build_bm25()
        if any(config_for(n, flow).use_dense for n in args.configs):
            engine.build_dense(show_progress=False)
            shared_model = dense._model
        flow_dirs = directions(rows, flow, docs)
        output["results"][flow] = {}
        for name in args.configs:
            cfg = config_for(name, flow)
            items = []
            for direction in flow_dirs:
                excluded = {engine.key_to_idx[k] for k in direction["exclude"] if k in engine.key_to_idx}
                ranked = engine.search(direction["query"], cfg, exclude=excluded)
                keys = [engine.keys[i] for i in ranked]
                if args.direction_filter:
                    src = direction.get("source_key") or ""
                    keys = demote(direction["query"], keys, labels,
                                  window=args.df_window,
                                  source_labels=labels.get(src.upper()) if src else None,
                                  mined=mined)
                items.append({**direction, "ranked_keys": keys})
            summary, details = score_rankings(items)
            output["results"][flow][name] = {"summary": summary, "details": details}
            print(f"{flow:<6} {name:<10} {summary}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
