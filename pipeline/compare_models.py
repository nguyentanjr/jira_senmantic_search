# -*- coding: utf-8 -*-
"""
So sanh CAP DOI hai model dense tren CUNG tap truy van (vd model fine-tune vs goc).

    python pipeline/compare_models.py --task gold
    python pipeline/compare_models.py --task paraphrase \
        --keys-file data/processed/finetune_split_clean.json [--symmetric]
    python pipeline/compare_models.py --task pooling

VI SAO CAN
eval_similar.py / eval_paraphrase.py chi so cac cau hinh BEN TRONG mot model voi
keyword. Hai file ket qua cua hai model khong so truc tiep duoc: "hybrid vs
keyword" doi tu +0,031 (goc) sang +0,052 (fine-tune) KHONG phai la kiem dinh
model moi tot hon. Script nay chay ca hai model tren cung truy van, lay hieu
tung truy van (B - A) va bootstrap tren hieu do -- dung ham paired() cua
eval_paraphrase.py de cung cach tinh CI.

keyword khong phu thuoc model nen khong co trong so sanh.

TIEN TO e5
  gold / pooling : 'query: ' ca hai phia (nhu eval_similar.py va nhu luc fine-tune)
  paraphrase     : mac dinh bat doi xung nhu eval_paraphrase.py ('passage: ' cho
                   tai lieu); --symmetric de dung 'query: ' ca hai phia. Model
                   fine-tune bang colab/finetune_e5_colab.ipynb chi duoc train
                   voi 'query: ', nen phai do ca hai che do truoc khi ket luan.

POOLING
Model moi tim ra tai lieu CHUA CHAM, eval tinh la 0 -> so sanh thien vi chong lai
model moi cho den khi pool duoc bo sung. Script in ty le top-10 chua co nhan cua
tung model; ty le do lech nhau nhieu thi con so pooling chua dung duoc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_paraphrase as ep                                          # noqa: E402
import eval_similar as es                                             # noqa: E402
import resolution                                                     # noqa: E402
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl  # noqa: E402
from run_ablation import CONFIGS, E5_QUERY                            # noqa: E402

D = "data/processed"


def build_engine(docs, model: str, query_prefix: str, passage_prefix: str):
    dense = DenseIndex(model, device="cpu", batch_size=32,
                       query_prefix=query_prefix, passage_prefix=passage_prefix)
    eng = SearchEngine(docs, dense=dense)
    eng.build_bm25()
    eng.build_dense(show_progress=False)
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gold", "pooling", "paraphrase"])
    ap.add_argument("--model-a", default="intfloat/multilingual-e5-base",
                    help="model moc (A)")
    ap.add_argument("--model-b", default="data/models/e5-base-jira-gold_weak",
                    help="model can so (B); bao cao B - A")
    ap.add_argument("--configs", nargs="*", default=["semantic", "hybrid"])
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--gold", default=f"{D}/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default=f"{D}/pooling_judgments_v3_llm.csv")
    ap.add_argument("--queries", default=f"{D}/paraphrase_queries.jsonl")
    ap.add_argument("--keys-file", default="")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if any(not CONFIGS[c]["use_dense"] for c in args.configs):
        sys.exit("cau hinh khong dung dense (vd keyword) giong het nhau o hai model")

    if args.task == "paraphrase":
        docs = load_jsonl(f"{D}/corpus_holdout.jsonl")
        queries = load_jsonl(args.queries)
        if args.keys_file:
            with open(args.keys_file, encoding="utf-8") as f:
                ks = json.load(f)
            ks = set(ks["paraphrase_clean_keys"] if isinstance(ks, dict) else ks)
            queries = [q for q in queries if q["key"] in ks]
        qp, pp = E5_QUERY, (E5_QUERY if args.symmetric else "passage: ")
        metrics = args.metrics or ["recall@10", "mrr@10", "ndcg@10"]

        def score(eng, cfg):
            return ep.per_query(eng, queries, cfg)
    else:
        docs = load_jsonl(f"{D}/corpus.jsonl")
        queries = (es.queries_from_gold(args.gold) if args.task == "gold"
                   else es.queries_from_pooling(args.pooling))
        has_res = {d["key"]: resolution.has(d) for d in docs}
        qp = pp = E5_QUERY
        metrics = args.metrics or ["ndcg@10", "recall@10", "resolution_hit@10"]

        def score(eng, cfg):
            return es.per_query(eng, queries, cfg, has_res), None
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        sys.exit("khong co truy van")

    print(f"task={args.task} | {len(queries)} truy van | prefix query='{qp}' passage='{pp}'")
    print(f"A = {args.model_a}\nB = {args.model_b}")
    eng_a = build_engine(docs, args.model_a, qp, pp)
    eng_b = build_engine(docs, args.model_b, qp, pp)

    result = {"task": args.task, "model_a": args.model_a, "model_b": args.model_b,
              "query_prefix": qp, "passage_prefix": pp, "n_queries_input": len(queries),
              "keys_file": args.keys_file, "n_boot": args.n_boot, "configs": {}}
    print(f"\n{'cau hinh':<10}{'chi so':<20}{'A':>8}{'B':>8}   B - A [95% CI]")
    for name in args.configs:
        cfg = SearchConfig(name=name, top_k=10, **CONFIGS[name])
        sa, ka = score(eng_a, cfg)
        sb, kb = score(eng_b, cfg)
        if ka is not None and ka != kb:
            sys.exit(f"{name}: tap truy van giu lai khac nhau giua hai model")
        result["configs"][name] = {}
        for m in metrics:
            a, b = sa.get(m), sb.get(m)
            if a is None or b is None or len(a) != len(b) or not len(a):
                continue
            pr = ep.paired(b, a, args.n_boot)
            result["configs"][name][m] = {"n": int(len(a)),
                                          "a": round(float(a.mean()), 4),
                                          "b": round(float(b.mean()), 4), **pr}
            star = "*" if pr["significant"] else " "
            print(f"{name:<10}{m:<20}{a.mean():>8.4f}{b.mean():>8.4f}   "
                  f"{pr['delta']:+.4f} [{pr['lo']:+.3f},{pr['hi']:+.3f}]{star} p={pr['p']}")
        if args.task == "pooling":
            ua, ub = es.unjudged_rate(eng_a, queries, cfg), es.unjudged_rate(eng_b, queries, cfg)
            result["configs"][name]["unjudged_top10"] = {"a": round(ua, 4), "b": round(ub, 4)}
            print(f"{name:<10}{'top-10 chua cham':<20}{ua:>8.1%}{ub:>8.1%}"
                  "   (B cao hon -> pooling thien vi chong lai B)")
    print("  (* = khoang tin cay 95% cua B - A khong chua 0)")

    out = args.out or f"{D}/compare_models_{args.task}{'_sym' if args.symmetric else ''}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
