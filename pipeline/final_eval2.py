# -*- coding: utf-8 -*-
"""
Vong 2 cho luong mo ta tu do: chon tren DEV cua HAI bo do, kiem MOT lan tren TEST.

    python pipeline/final_eval2.py --split dev  --device cuda
    python pipeline/final_eval2.py --split test --device cuda   # can final2_choice.json

BO DO (cung tach dev/test theo key)
    paraphrase : actual -> issue da tru actual   (corpus_holdout)
    title      : tieu de -> issue da tru tieu de (corpus_title_holdout, build_title_eval.py)
Moi truy van hai ban (co dau / khong dau) -> 4 "lat": paraphrase:goc,
paraphrase:khong_dau, title:goc, title:khong_dau. BM25 fold="auto", rerank top-30.

LUOI -- CHOT TRUOC KHI DO (2026-09-14)
    dense    : e5-task (v1, hien tai) | e5-task2 (e5-base, train v2) | e5-large-task2 (e5-large, train v2)
    reranker : ce-task (v1, hien tai) | ce-task2 (bge-reranker-base, train v2)
    tron     : blend@0.3 | blend@0.5
Hien tai = e5-task|ce-task|0.3. Kem keyword va dense|hybrid de bao cao.
Tren Colab, model v1 duoc TRAIN LAI bang cung script/seed (ban goc chi co o may local).

QUY TAC CHON -- CHOT TRUOC KHI DO
    Du dieu kien: o CA 4 lat, R@10 khong thap hon `hien tai` co y nghia (hi >= 0).
    Chon ung vien du dieu kien co MRR@10 trung binh 4 lat cao nhat; khong ai hon -> giu.

Dau ra: final2_eval_<split>.json, final2_choice.json
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

SETS = {"paraphrase": ("data/processed/corpus_holdout.jsonl", "data/processed/paraphrase_queries.jsonl"),
        "title": ("data/processed/corpus_title_holdout.jsonl", "data/processed/title_queries.jsonl")}
DENSE = {"e5-task": "models/e5-base-task", "e5-task2": "models/e5-base-task2",
         "e5-large-task2": "models/e5-large-task2"}
RERANK = {"ce-task": "models/bge-reranker-base-task", "ce-task2": "models/bge-reranker-base-task2"}
FUSIONS = ("0.3", "0.5")
CURRENT = "e5-task|ce-task|0.3"
QVERS = ("goc", "khong_dau")
METRICS = ("recall@10", "mrr@10", "ndcg@10")
RERANK_K = 30


def choose2(ci_block: dict, paired_block: dict, names, slices) -> dict:
    checks = {}
    for n in names:
        ok = n == CURRENT or all(paired_block[s][n]["recall@10"]["hi"] >= 0 for s in slices)
        checks[n] = {"eligible": ok,
                     "mean_mrr": round(float(np.mean([ci_block[s][n]["mrr@10"]["mean"] for s in slices])), 4)}
    best = CURRENT
    for n in names:
        if checks[n]["eligible"] and checks[n]["mean_mrr"] > checks[best]["mean_mrr"]:
            best = n
    return {"chosen": best, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--dense", nargs="*", default=list(DENSE))
    ap.add_argument("--rerankers", nargs="*", default=list(RERANK))
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--choice", default="data/processed/final2_choice.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out or f"data/processed/final2_eval_{args.split}.json"

    dense_names = [d for d in args.dense if os.path.isdir(DENSE[d])]
    rr_names = [r for r in args.rerankers if os.path.isdir(RERANK[r])]
    chosen = None
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
    print(f"bo do {args.sets} | dense {dense_names} | reranker {rr_names} | bo qua: {skipped}")

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        wanted = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])[args.split])

    data = {}
    for s in args.sets:
        cpath, qpath = SETS[s]
        corpus = load_jsonl(cpath)
        qs = [q for q in load_jsonl(qpath) if q["key"] in wanted]
        if args.limit:
            qs = qs[: args.limit]
        base = SearchEngine(corpus, bm25_fold="auto")
        base.build_bm25()
        items = []
        for q in qs:
            rel = {base.key_to_idx[k] for k in q["relevant"] if k in base.key_to_idx}
            if rel:
                for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                    items.append({"qv": qv, "text": text, "rel": rel, "jac_bin": q.get("jac_bin")})
        cfg_kw = SearchConfig(name="keyword", top_k=100, **CONFIGS["keyword"])
        data[s] = {"corpus": corpus, "base": base, "items": items,
                   "ranked": {"keyword": [base.search(it["text"], cfg_kw) for it in items]},
                   "hybrid": {}}
        print(f"{s}: {len(items) // 2} truy van x 2 ban", flush=True)

    cfg_hy = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])
    for dn in dense_names:
        for s, D in data.items():
            t0 = time.perf_counter()
            eng = SearchEngine(D["corpus"], bm25_fold="auto",
                               dense=DenseIndex(DENSE[dn], device=args.device, batch_size=64,
                                                query_prefix="query: ", passage_prefix="passage: "))
            eng._bm25, eng._bm25_add = D["base"]._bm25, D["base"]._bm25_add
            eng.build_dense(show_progress=True)
            D["hybrid"][dn] = [eng.search(it["text"], cfg_hy) for it in D["items"]]
            D["ranked"][f"{dn}|hybrid"] = D["hybrid"][dn]
            print(f"  hybrid {dn} / {s}: {time.perf_counter() - t0:.0f}s", flush=True)
            del eng
            gc.collect()

    for rn in rr_names:
        t0 = time.perf_counter()
        rr = CrossEncoderReranker(RERANK[rn], device=args.device, batch_size=args.ce_batch, fp16=True)
        for s, D in data.items():
            need = sorted({(i, int(d)) for dn in dense_names for i, h in enumerate(D["hybrid"][dn])
                           for d in h[:RERANK_K]})
            sc = rr.score_pairs([(D["items"][i]["text"], D["corpus"][d].get("text_dense", ""))
                                 for i, d in need], show_progress=True)
            cache = dict(zip(need, sc.tolist()))
            for dn in dense_names:
                for fu in FUSIONS:
                    name = f"{dn}|{rn}|{fu}"
                    if chosen and name not in (CURRENT, chosen):
                        continue
                    D["ranked"][name] = [
                        np.concatenate([fuse_rerank(h[:RERANK_K],
                                                    np.asarray([cache[(i, int(d))] for d in h[:RERANK_K]]),
                                                    "blend", float(fu)), h[RERANK_K:]])
                        for i, h in enumerate(D["hybrid"][dn])]
        print(f"  reranker {rn}: {time.perf_counter() - t0:.0f}s", flush=True)
        del rr
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    names = list(next(iter(data.values()))["ranked"])
    slices = [f"{s}:{qv}" for s in data for qv in QVERS]
    arr = {sl: {n: {m: [] for m in METRICS} for n in names} for sl in slices}
    low = {sl: [] for sl in slices}
    for s, D in data.items():
        counters = {qv: 0 for qv in QVERS}
        for i, it in enumerate(D["items"]):
            sl = f"{s}:{it['qv']}"
            for n in names:
                for m, v in metrics_from_ranked(D["ranked"][n][i], it["rel"]).items():
                    arr[sl][n][m].append(v)
            if it["jac_bin"] == "<0.2":
                low[sl].append(counters[it["qv"]])
            counters[it["qv"]] += 1
    arr = {sl: {n: {m: np.asarray(v) for m, v in d.items()} for n, d in dd.items()} for sl, dd in arr.items()}

    def block(sub: bool):
        res = {"ci": {}, "vs_current": {}, "vs_keyword": {}}
        for sl in slices:
            idx = np.asarray(low[sl], dtype=int)
            sel = (lambda a, idx=idx: a[idx]) if sub else (lambda a: a)
            if sub and not len(idx):
                continue
            res["ci"][sl] = {n: {m: ci(sel(arr[sl][n][m]), args.n_boot) for m in METRICS} for n in names}
            if CURRENT in names:
                res["vs_current"][sl] = {n: {m: paired(sel(arr[sl][n][m]), sel(arr[sl][CURRENT][m]), args.n_boot)
                                             for m in METRICS} for n in names if n != CURRENT}
            res["vs_keyword"][sl] = {n: {m: paired(sel(arr[sl][n][m]), sel(arr[sl]["keyword"][m]), args.n_boot)
                                         for m in METRICS} for n in names if n != "keyword"}
        return res

    result = {"split": args.split, "sets": list(data), "dense": dense_names, "rerankers": rr_names,
              "skipped_missing": skipped, "limit": args.limit, "bm25_fold": "auto",
              "n_queries": {s: len(D["items"]) // 2 for s, D in data.items()},
              "overall": block(False), "jac_lt_0.2": block(True)}

    b = result["overall"]
    for sl in slices:
        print(f"\n== {sl} ==  R@10 MRR@10 nDCG@10 | vs hien tai (R@10, MRR@10)")
        for n in names:
            x = b["ci"][sl][n]
            line = f"  {n:<26}" + " ".join(f"{x[m]['mean']:.3f}" for m in METRICS)
            if n in b["vs_current"].get(sl, {}):
                d = b["vs_current"][sl][n]
                line += " | " + " ".join(
                    f"{d[m]['delta']:+.4f}[{d[m]['lo']:+.3f},{d[m]['hi']:+.3f}]{'*' if d[m]['significant'] else ' '}"
                    for m in ("recall@10", "mrr@10"))
            print(line)

    if args.split == "dev":
        cands = [n for n in names if n.count("|") == 2]
        if CURRENT not in cands:
            sys.exit("khong co cau hinh hien tai trong luoi -- khong chon duoc")
        ch = choose2(b["ci"], b["vs_current"], cands, slices)
        result["choice"] = ch
        if not args.limit:
            with open(args.choice, "w", encoding="utf-8") as f:
                json.dump({"chosen": ch["chosen"], "checks": ch["checks"], "slices": slices,
                           "skipped_missing": skipped}, f, ensure_ascii=False, indent=1)
        print(f"\nCHON THEO QUY TAC: {ch['chosen']}")
        for n, c in sorted(ch["checks"].items(), key=lambda kv: -kv[1]["mean_mrr"]):
            print(f"  {n:<26} {c}")
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
