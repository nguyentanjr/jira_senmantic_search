# -*- coding: utf-8 -*-
"""
Phan tich THEO NHOM TRUY VAN cho luong issue (goi y tu mot issue) -- mo ta, KHONG chon.

    python pipeline/group_analysis_issue.py

Thiet ke chot truoc khi chay: HANDOFF.md §7 "Follow-up analyses" (muc A).

BO DO: gold 134 (nhan cue, thien lech lexical) va pooling v4 (39 truy van co diem,
nhan LLM; 40 truy van da lay mau phan tang 20/20 theo lexical_group).
CAU HINH: keyword | semantic, hybrid voi e5 goc | semantic, hybrid voi e5-task2
(hybrid e5-task2 = dang chay). Tien to 'query: ' hai phia, BM25 fold none.
NHOM: truong `lexical_group` cua issue truy van (group_stats.py).

POOLING: pool v4 gom top-10 cua keyword, semantic/hybrid e5 goc va hybrid
e5-task2. semantic e5-task2 KHONG duoc gop pool -> in ty le chua cham; > 1% thi
con so cua no khong dung duoc (liet ke trong `unjudged_over_1pct`).

Dau ra: data/processed/group_issue.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_similar as es                                                 # noqa: E402
import resolution                                                         # noqa: E402
from group_stats import GROUPS, group_label, group_report, print_report  # noqa: E402
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl  # noqa: E402
from run_ablation import CONFIGS, E5_QUERY                                # noqa: E402

D = "data/processed"
MODELS = {"e5": "intfloat/multilingual-e5-base", "e5-task2": "models/e5-base-task2"}
METRICS = ("ndcg@10", "recall@10", "mrr@10", "resolution_hit@10")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=f"{D}/corpus.jsonl")
    ap.add_argument("--gold", default=f"{D}/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default=f"{D}/pooling_judgments_v4_llm.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=f"{D}/group_issue.json")
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    by_key = {d["key"]: d for d in docs}
    has_res = {d["key"]: resolution.has(d) for d in docs}
    sets = {"gold": es.queries_from_gold(args.gold),
            "pooling_v4": es.queries_from_pooling(args.pooling)}

    bm25 = SearchEngine(docs)
    bm25.build_bm25()
    engines = {}
    for name, path in MODELS.items():
        dense = DenseIndex(path, device="cpu", batch_size=32,
                           query_prefix=E5_QUERY, passage_prefix=E5_QUERY)
        eng = SearchEngine(docs, dense=dense)
        eng._bm25 = bm25._bm25
        eng.build_dense(show_progress=False)
        engines[name] = eng
    systems = {"keyword": (engines["e5"], "keyword")}
    for name in MODELS:
        for cfg in ("semantic", "hybrid"):
            systems[f"{cfg}|{name}"] = (engines[name], cfg)

    result = {"groups": list(GROUPS), "models": MODELS, "n_boot": args.n_boot,
              "group_field": "lexical_group", "sets": {}}
    for sname, queries in sets.items():
        scores, keys, unjudged = {}, None, {}
        for sys_name, (eng, cfg_name) in systems.items():
            cfg = SearchConfig(name=cfg_name, top_k=10, **CONFIGS[cfg_name])
            kept: list[str] = []
            s = es.per_query(eng, queries, cfg, has_res, kept=kept)
            if keys is None:
                keys = kept
            elif kept != keys:
                sys.exit(f"{sname}/{sys_name}: tap truy van khac nhau giua cac he")
            scores[sys_name] = {m: s[m] for m in METRICS}
            if sname.startswith("pooling"):
                unjudged[sys_name] = round(es.unjudged_rate(eng, queries, cfg), 4)
        labels = [group_label(bool(by_key[k].get("lexical_group"))) for k in keys]
        rep = group_report(scores, labels, "keyword", METRICS, args.n_boot)
        if unjudged:
            rep["unjudged_top10"] = unjudged
            rep["unjudged_over_1pct"] = sorted(k for k, v in unjudged.items() if v > 0.01)
        result["sets"][sname] = rep
        print_report(sname, rep, metrics=("ndcg@10", "recall@10"))
        if unjudged:
            print(f"  top-10 chua cham: {unjudged}  | >1% (khong dung duoc): {rep['unjudged_over_1pct']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
