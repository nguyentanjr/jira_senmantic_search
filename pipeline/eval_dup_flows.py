# -*- coding: utf-8 -*-
"""
Do reranker cua PIPELINE DUP (pipeline/dup) theo TUNG LUONG, truoc khi dinh tuyen.

    # luong issue: chon tren train, bao cao tren eval (gold cap tu comment)
    python pipeline/eval_dup_flows.py --flow issue --split train
    python pipeline/eval_dup_flows.py --flow issue --split eval
    # luong text: chon tren dev (500), test (2.675) chay MOT lan
    python pipeline/eval_dup_flows.py --flow text --split dev
    python pipeline/eval_dup_flows.py --flow text --split test --variants asym asym+rerank

VI SAO KHONG SUY TU retrieval.SearchEngine
Reranker o day KHAC: diem cross-encoder (minmax trong top-30) chi thay mot nua
phan base (blend 0.5), roi bonus/penalty/tran hard-negative duoc ap lai
(dup/rerank.py). Ung vien cung khac (co nguon metadata, pool 150). Nen phai do
rieng.

BIEN THE
    issue:  norerank | rerank            (tien to doi xung, nhu da do)
    text:   sym | sym+rerank | asym | asym+rerank
            sym  = 'query: ' hai phia (hanh vi hien tai cua --query)
            asym = 'query: ' / 'passage: ' (dung cho engine chinh o luong text)
    keyword (BM25 cua retrieval.py) duoc do kem cho luong text de doi chieu.

QUY TAC CHON -- CHOT TRUOC KHI DO (2026-09-12)
    issue: bat rerank chi khi nDCG@10(rerank - norerank) tren TRAIN > 0 va CI
           khong chua 0. Nguoc lai giu tat.
    text:  tien to mac dinh asym; doi sang sym chi khi R@10(sym - asym) tren DEV
           > 0 va CI khong chua 0. Voi tien to da chon, bat rerank chi khi
           R@10(+rerank - khong) tren DEV > 0 va CI khong chua 0.

DON VI BOOTSTRAP
    text:  moi truy van la mot issue rieng -> bootstrap thuong.
    issue: truy van a->b va b->a khong doc lap -> resample theo CUM LIEN THONG
           cua cac cap gold (evaluate.clusters).

Dau ra: data/processed/dup_flows_<flow>_<split>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from pipeline.dup import DuplicatePipeline, load_config           # noqa: E402
from pipeline.dup.evaluate import build_queries, clusters          # noqa: E402
from pipeline.dup.pipeline import flow_plan                        # noqa: E402
from eval_paraphrase import ci, paired                             # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked         # noqa: E402
from retrieval import SearchConfig, SearchEngine, load_jsonl       # noqa: E402
from run_ablation import CONFIGS                                   # noqa: E402

METRICS = ("recall@10", "mrr@10", "ndcg@10")
VARIANTS = {"issue": ["norerank", "rerank"],
            "text": ["sym", "sym+rerank", "asym", "asym+rerank"]}
SPLITS = {"issue": ("train", "eval"), "text": ("dev", "test")}
REPORT_SPLIT = {"issue": "eval", "text": "test"}


# ---------------------------------------------------------------- bootstrap

def cluster_means(values: np.ndarray, groups: np.ndarray, n_boot: int,
                  seed: int = 17) -> np.ndarray:
    """Trung binh bootstrap, resample theo nhom (vector hoa bang bincount)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[pick].sum(1) / cnts[pick].sum(1)


def summarize(values: np.ndarray, groups, n_boot: int) -> dict:
    if groups is None:
        return ci(values, n_boot)
    bs = cluster_means(values, groups, n_boot)
    return {"mean": round(float(values.mean()), 4),
            "lo": round(float(np.percentile(bs, 2.5)), 4),
            "hi": round(float(np.percentile(bs, 97.5)), 4)}


def paired_diff(a: np.ndarray, b: np.ndarray, groups, n_boot: int) -> dict:
    if groups is None:
        return paired(a, b, n_boot)
    d = a - b
    bs = cluster_means(d, groups, n_boot)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"delta": round(float(d.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "p": round(float(min(p, 1.0)), 4),
            "significant": bool(lo > 0 or hi < 0)}


def choose(flow: str, paired_block: dict) -> dict:
    """Ap quy tac chon da chot o docstring len ket qua cua split chon."""
    def gain(name, metric):
        d = paired_block.get(name, {}).get(metric)
        return bool(d and d["delta"] > 0 and d["significant"])

    if flow == "issue":
        return {"rerank": gain("rerank - norerank", "ndcg@10")}
    prefix = "sym" if gain("sym - asym", "recall@10") else "asym"
    return {"prefix": prefix,
            "rerank": gain(f"{prefix}+rerank - {prefix}", "recall@10")}


# ---------------------------------------------------------------- run

def _pipeline(cfg_args, docs, device: str):
    cfg = load_config(cfg_args.config, list(cfg_args.overrides))
    cfg.rerank.enabled = False
    cfg.rerank.device = device
    cfg.embedding.device = device
    return DuplicatePipeline(cfg, docs=docs).build(show_progress=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["issue", "text"], required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--config", default="configs/dup_v2_frozen.json")
    ap.add_argument("--set", dest="overrides", action="extend", nargs="+", default=[])
    ap.add_argument("--corpus-text", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.split not in SPLITS[args.flow]:
        sys.exit(f"--flow {args.flow} chi nhan split {SPLITS[args.flow]}")
    variants = args.variants or VARIANTS[args.flow]
    bad = set(variants) - set(VARIANTS[args.flow])
    if bad:
        sys.exit(f"bien the khong hop le cho {args.flow}: {sorted(bad)}")
    out = args.out or f"data/processed/dup_flows_{args.flow}_{args.split}.json"
    final = args.split == REPORT_SPLIT[args.flow] and args.flow == "text"
    if final and os.path.exists(out) and not args.force and not args.limit:
        sys.exit(f"{out} da ton tai: test chi chay mot lan. Dung --force neu that su can.")

    groups = None
    if args.flow == "issue":
        base_cfg = load_config(args.config, list(args.overrides))
        docs = load_jsonl(base_cfg.corpus_path)
        gold = load_jsonl(f"data/processed/gold_{args.split}_pairs.jsonl")
        keys = {d["key"] for d in docs}
        queries = build_queries(gold, keys)
        cl = clusters(gold)
        if args.limit:
            queries = queries[: args.limit]
        groups = np.asarray([cl[q["key"]] for q in queries])
        print(f"luong issue, split {args.split}: {len(queries)} truy van, "
              f"{len(np.unique(groups))} cum | corpus {len(docs)}")
    else:
        with open(args.clean_keys, encoding="utf-8") as f:
            clean = json.load(f)["paraphrase_clean_keys"]
        with open(args.dev_keys, encoding="utf-8") as f:
            dev = json.load(f)["paraphrase_clean_keys"]
        wanted = set(make_split(clean, dev)[args.split])
        docs = load_jsonl(args.corpus_text)
        queries = [q for q in load_jsonl(args.queries) if q["key"] in wanted]
        if args.limit:
            queries = random.Random(13).sample(queries, min(args.limit, len(queries)))
        keys = {d["key"] for d in docs}
        queries = [q for q in queries if any(k in keys for k in q["relevant"])]
        print(f"luong text, split {args.split}: {len(queries)} truy van | corpus {len(docs)}")

    names = (["keyword"] if args.flow == "text" else []) + variants
    scores = {n: {m: [] for m in METRICS} for n in names}
    latency = {}

    if args.flow == "text":
        eng = SearchEngine(docs)
        eng.build_bm25()
        cfg_kw = SearchConfig(name="keyword", top_k=10, **CONFIGS["keyword"])
        for q in queries:
            ranked = [eng.keys[i] for i in eng.search(q["query"], cfg_kw)]
            for m, v in metrics_from_ranked(ranked, set(q["relevant"])).items():
                scores["keyword"][m].append(v)

    # Mot pipeline cho moi bien the: bien the chi doi co THEO LUONG
    # (embedding.text_symmetric, rerank.text / rerank.issue), nen phep do di
    # qua dung duong dinh tuyen ma run_dup_e2e dung.
    p = _pipeline(args, docs, args.device)
    for v in variants:
        on = v.endswith("+rerank") or v == "rerank"
        kind = "cross_encoder" if on else "none"
        if args.flow == "issue":
            p.cfg.rerank.issue = kind
        else:
            p.cfg.embedding.text_symmetric = v.split("+")[0] == "sym"
            p.cfg.rerank.text = kind
        print(f"\n== {v}: {flow_plan(p.cfg, args.flow)} ==", flush=True)
        t0 = time.perf_counter()
        for j, q in enumerate(queries):
            if args.flow == "issue":
                ranked = p.rank_keys(q["key"], 10)
                rel = {r for r in q["relevant"] if r in p.key_to_idx}
            else:
                ranked = [r["bugId"] for r in p.search(q["query"], top_k=10)["results"]]
                rel = set(q["relevant"])
            for m, val in metrics_from_ranked(ranked, rel).items():
                scores[v][m].append(val)
            if (j + 1) % 100 == 0:
                print(f"    {v}: {j + 1}/{len(queries)}", flush=True)
        latency[v] = round((time.perf_counter() - t0) * 1000 / max(1, len(queries)), 1)
        print(f"  xong {v}: {latency[v]} ms/truy van", flush=True)
    del p

    scores = {n: {m: np.asarray(x) for m, x in d.items()} for n, d in scores.items()}
    comps = []
    if args.flow == "issue":
        if {"rerank", "norerank"} <= set(variants):
            comps.append(("rerank", "norerank"))
    else:
        for pre in ("sym", "asym"):
            if {pre, pre + "+rerank"} <= set(variants):
                comps.append((pre + "+rerank", pre))
        if {"sym", "asym"} <= set(variants):
            comps.append(("sym", "asym"))
        comps += [(v, "keyword") for v in variants]

    def block(idx):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        g = None if groups is None else sel(groups)
        return {"n": int(len(sel(scores[names[0]][METRICS[0]]))),
                "ci": {n: {m: summarize(sel(scores[n][m]), g, args.n_boot)
                           for m in METRICS} for n in names},
                "paired": {f"{a} - {b}": {m: paired_diff(sel(scores[a][m]), sel(scores[b][m]),
                                                         g, args.n_boot)
                                          for m in METRICS} for a, b in comps}}

    result = {"flow": args.flow, "split": args.split, "n_queries": len(queries),
              "limit": args.limit, "variants": variants, "config": args.config,
              "overrides": args.overrides, "device": args.device,
              "ms_per_query": latency, "overall": block(None)}
    if args.flow == "text":
        low = np.asarray([i for i, q in enumerate(queries) if q.get("jac_bin") == "<0.2"])
        if len(low):
            result["jac_lt_0.2"] = block(low)
    if args.split == SPLITS[args.flow][0]:
        result["choice"] = choose(args.flow, result["overall"]["paired"])

    for key in ("overall", "jac_lt_0.2"):
        if key not in result:
            continue
        b = result[key]
        print(f"\n== {key} (n={b['n']}) ==")
        for n in names:
            print(f"  {n:<14}" + "".join(f"{m} {b['ci'][n][m]['mean']:.4f}  " for m in METRICS))
        for c, d in b["paired"].items():
            print(f"  {c:<22}" + "".join(
                f"{x['delta']:+.4f}[{x['lo']:+.3f},{x['hi']:+.3f}]{'*' if x['significant'] else ' '} "
                for x in d.values()))
    print("  (* = CI 95% khong chua 0; cot: " + ", ".join(METRICS) + ")")
    if "choice" in result:
        print(f"\nCHON THEO QUY TAC DA CHOT: {result['choice']}")

    if args.limit and final:
        print("\n--limit tren test: khong ghi file")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
