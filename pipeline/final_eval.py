# -*- coding: utf-8 -*-
"""
Lan thu CUOI cho luong mo ta tu do: chon tren DEV, kiem MOT lan tren TEST.

    python pipeline/final_eval.py --split dev  --device cuda
    python pipeline/final_eval.py --split test --device cuda     # can final_choice.json

LUOI -- CHOT TRUOC KHI DO (2026-09-14). BM25 fold="add" co dinh, rerank top-30.
    dense    : e5 (intfloat/multilingual-e5-base, hien tai)
               bge-m3 (BAAI/bge-m3, dung nguyen, khong tien to, seq 512)
               e5-task (models/e5-base-task, fine-tune tren task_pairs)
    reranker : bge (BAAI/bge-reranker-base, hien tai)
               ce-task (models/bge-reranker-base-task)
    tron     : blend@0.3, blend@0.5, replace (= CE thay han)
Ung vien = dense|reranker|tron. Kem `dense|hybrid` (khong rerank) va `keyword`
de bao cao, khong duoc chon. Model nao chua co file thi bo qua (ghi vao ket qua).
Hien tai = e5|bge|0.3.

QUY TAC CHON -- CHOT TRUOC KHI DO
    Du dieu kien: o CA HAI ban truy van (co dau, khong dau), R@10 khong thap hon
    `hien tai` co y nghia (CI 95% cua hieu co hi >= 0).
    Chon ung vien du dieu kien co trung binh MRR@10 cua hai ban cao nhat.
    Hoa hoac khong ai hon -> giu hien tai.

TEST: hien tai, lua chon, keyword, hybrid cua dense duoc chon va cua e5.
File ket qua test khong ghi de tru khi --force.

Dau ra: final_eval_<split>.json, final_choice.json (dev)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paraphrase import ci, paired                                     # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked                 # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,     # noqa: E402
                       SearchEngine, fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS                                           # noqa: E402

DENSE = {"e5": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
         "bge-m3": ("BAAI/bge-m3", "", ""),
         "e5-task": ("models/e5-base-task", "query: ", "passage: ")}
RERANK = {"bge": "BAAI/bge-reranker-base", "ce-task": "models/bge-reranker-base-task"}
FUSIONS = ("0.3", "0.5", "replace")
CURRENT = "e5|bge|0.3"
QVERS = ("goc", "khong_dau")
METRICS = ("recall@10", "mrr@10", "ndcg@10")
RERANK_K = 30


def available(path: str) -> bool:
    return "/" in path and not path.startswith("models/") or os.path.isdir(path)


def fused(head, ce, fusion: str):
    if fusion == "replace":
        return fuse_rerank(head, ce, "replace")
    return fuse_rerank(head, ce, "blend", float(fusion))


def choose_final(ci_block: dict, paired_block: dict, names) -> dict:
    """ci_block[qv][name][metric]['mean'], paired_block[qv][name vs CURRENT][metric]."""
    checks = {}
    for n in names:
        ok = n == CURRENT or all(paired_block[qv][n]["recall@10"]["hi"] >= 0 for qv in QVERS)
        mrr = float(np.mean([ci_block[qv][n]["mrr@10"]["mean"] for qv in QVERS]))
        checks[n] = {"eligible": ok, "mean_mrr": round(mrr, 4)}
    best = CURRENT
    for n in names:
        if checks[n]["eligible"] and checks[n]["mean_mrr"] > checks[best]["mean_mrr"]:
            best = n
    return {"chosen": best, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--dense", nargs="*", default=list(DENSE))
    ap.add_argument("--rerankers", nargs="*", default=list(RERANK))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--bm25-fold", default="add", choices=["none", "add", "only", "auto"],
                    help="lan chon/test goc dung add; auto = ban sua loi (2026-09-14)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--choice", default="data/processed/final_choice.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out or f"data/processed/final_eval_{args.split}.json"

    dense_names = [d for d in args.dense if available(DENSE[d][0])]
    rr_names = [r for r in args.rerankers if available(RERANK[r])]
    if args.split == "test":
        if not os.path.exists(args.choice):
            sys.exit(f"chua co {args.choice}: chay --split dev truoc")
        if os.path.exists(out) and not args.force and not args.limit:
            sys.exit(f"{out} da ton tai: test chi chay mot lan. Dung --force neu that su can.")
        with open(args.choice, encoding="utf-8") as f:
            chosen = json.load(f)["chosen"]
        need = {CURRENT, chosen}
        dense_names = sorted({n.split("|")[0] for n in need})
        rr_names = sorted({n.split("|")[1] for n in need})
    skipped = sorted(set(args.dense) - set(dense_names)) + sorted(set(args.rerankers) - set(rr_names))
    print(f"dense {dense_names} | reranker {rr_names} | bo qua (chua co model): {skipped}")

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        wanted = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])[args.split])
    corpus = load_jsonl(args.corpus)
    queries = [q for q in load_jsonl(args.queries) if q["key"] in wanted]
    if args.limit:
        queries = queries[: args.limit]

    base = SearchEngine(corpus, bm25_fold=args.bm25_fold)
    base.build_bm25()
    items = []
    for q in queries:
        rel = {base.key_to_idx[k] for k in q["relevant"] if k in base.key_to_idx}
        if rel:
            for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                items.append({"key": q["key"], "qv": qv, "text": text, "rel": rel,
                              "jac_bin": q.get("jac_bin")})
    print(f"{args.split}: {len(items) // 2} truy van x 2 ban")

    ranked: dict[str, list] = {}
    cfg_kw = SearchConfig(name="keyword", top_k=100, **CONFIGS["keyword"])
    ranked["keyword"] = [base.search(it["text"], cfg_kw) for it in items]
    hybrids: dict[str, list] = {}
    for dn in dense_names:
        path, qp, pp = DENSE[dn]
        t0 = time.perf_counter()
        eng = SearchEngine(corpus, bm25_fold=args.bm25_fold,
                           dense=DenseIndex(path, device=args.device, batch_size=64,
                                            query_prefix=qp, passage_prefix=pp,
                                            max_seq_length=512 if dn == "bge-m3" else None))
        eng._bm25, eng._bm25_add = base._bm25, base._bm25_add
        eng.build_dense(show_progress=True)
        cfg = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])
        hybrids[dn] = [eng.search(it["text"], cfg) for it in items]
        ranked[f"{dn}|hybrid"] = hybrids[dn]
        print(f"  hybrid {dn}: {time.perf_counter() - t0:.0f}s", flush=True)
        del eng
        gc.collect()

    for rn in rr_names:
        t0 = time.perf_counter()
        rr = CrossEncoderReranker(RERANK[rn], device=args.device, batch_size=args.ce_batch,
                                  fp16=True)
        need = sorted({(i, int(d)) for dn in dense_names for i, h in enumerate(hybrids[dn])
                       for d in h[:RERANK_K]})
        s = rr.score_pairs([(items[i]["text"], corpus[d].get("text_dense", "")) for i, d in need],
                           show_progress=True)
        cache = dict(zip(need, s.tolist()))
        for dn in dense_names:
            for fu in FUSIONS:
                name = f"{dn}|{rn}|{fu}"
                if args.split == "test" and name not in (CURRENT, chosen):
                    continue
                lst = []
                for i, h in enumerate(hybrids[dn]):
                    head = h[:RERANK_K]
                    ce = np.asarray([cache[(i, int(d))] for d in head])
                    lst.append(np.concatenate([fused(head, ce, fu), h[RERANK_K:]]))
                ranked[name] = lst
        print(f"  reranker {rn}: {len(need)} cap, {time.perf_counter() - t0:.0f}s", flush=True)
        del rr
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    names = list(ranked)
    arr = {qv: {n: {m: [] for m in METRICS} for n in names} for qv in QVERS}
    for n in names:
        for it, r in zip(items, ranked[n]):
            for m, v in metrics_from_ranked(r, it["rel"]).items():
                arr[it["qv"]][n][m].append(v)
    arr = {qv: {n: {m: np.asarray(v) for m, v in d.items()} for n, d in dd.items()}
           for qv, dd in arr.items()}
    low = {qv: np.asarray([j for j, it in enumerate([x for x in items if x["qv"] == qv])
                           if it["jac_bin"] == "<0.2"]) for qv in QVERS}

    def block(sub: bool):
        res = {"ci": {}, "vs_current": {}, "vs_keyword": {}}
        for qv in QVERS:
            sel = (lambda a, qv=qv: a[low[qv]]) if sub else (lambda a: a)
            res["ci"][qv] = {n: {m: ci(sel(arr[qv][n][m]), args.n_boot) for m in METRICS} for n in names}
            if CURRENT in names:
                res["vs_current"][qv] = {n: {m: paired(sel(arr[qv][n][m]), sel(arr[qv][CURRENT][m]),
                                                       args.n_boot) for m in METRICS}
                                         for n in names if n != CURRENT}
            res["vs_keyword"][qv] = {n: {m: paired(sel(arr[qv][n][m]), sel(arr[qv]["keyword"][m]),
                                                   args.n_boot) for m in METRICS}
                                     for n in names if n != "keyword"}
        return res

    result = {"split": args.split, "bm25_fold": args.bm25_fold,
              "n_queries": len(items) // 2, "dense": dense_names,
              "rerankers": rr_names, "skipped_missing": skipped, "limit": args.limit,
              "overall": block(False), "jac_lt_0.2": block(True)}

    b = result["overall"]
    print("\n== overall ==  R@10 MRR@10 nDCG@10 | vs hien tai (R@10, MRR@10)")
    for qv in QVERS:
        for n in names:
            x = b["ci"][qv][n]
            line = f"  {qv:<10}{n:<22}" + " ".join(f"{x[m]['mean']:.3f}" for m in METRICS)
            if n in b["vs_current"].get(qv, {}):
                d = b["vs_current"][qv][n]
                line += " | " + " ".join(
                    f"{d[m]['delta']:+.4f}[{d[m]['lo']:+.3f},{d[m]['hi']:+.3f}]{'*' if d[m]['significant'] else ' '}"
                    for m in ("recall@10", "mrr@10"))
            print(line)

    if args.split == "dev":
        cands = [n for n in names if n.count("|") == 2]
        if CURRENT not in cands:
            sys.exit("khong co cau hinh hien tai trong luoi -- khong chon duoc")
        ch = choose_final(b["ci"], b["vs_current"], cands)
        result["choice"] = ch
        if not args.limit:
            with open(args.choice, "w", encoding="utf-8") as f:
                json.dump({"chosen": ch["chosen"], "checks": ch["checks"],
                           "skipped_missing": skipped}, f, ensure_ascii=False, indent=1)
        print(f"\nCHON THEO QUY TAC: {ch['chosen']}")
        for n, c in sorted(ch["checks"].items(), key=lambda kv: -kv[1]["mean_mrr"]):
            print(f"  {n:<22} {c}")
    else:
        result["chosen"] = chosen

    if args.limit and args.split == "test":
        print("--limit tren test: khong ghi file")
        return
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
