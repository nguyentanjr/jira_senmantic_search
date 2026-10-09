# -*- coding: utf-8 -*-
"""
DUONG CONG DU LIEU cho bi-encoder: can bao nhieu cap train? -- DEV + gold, KHONG test.

    python pipeline/learning_curve_eval.py --device cuda --ref f100-s13 \
        --models f000=intfloat/multilingual-e5-base f010-s13=models/lc/e5-f010-s13 ... \
                 f100-s13=models/e5-base-task2
    python pipeline/learning_curve_eval.py ... --limit 20      # chay thu, khong ghi file

VI SAO
problem.txt: "danh gia tren tap held-out de xac dinh tinh kha thi va uoc luong nguong
du lieu can thiet". Loi ich cua ca hai luong den tu bi-encoder train tren cap
"mo ta / tieu de -> issue" (task_pairs_v2.jsonl, 6.239 issue).

THIET KE -- CHOT TRUOC KHI CHAY (2026-09-14, HANDOFF.md §7 "Follow-up analyses", muc B)
  Du lieu : 10% / 25% / 50% / 100% so ISSUE (moi issue giu du dong), tap long nhau
            (train_task_models.py --fraction, --frac-seed 13); 0% = e5 goc.
  Seed    : moi muc train 2 seed (13, 14). 100% / seed 13 = models/e5-base-task2.
  Sieu tham so giu nguyen -> it du lieu = it buoc.
  Do      : luong text -- dev paraphrase 500 + tieu de 495, co dau / khong dau (4 lat):
            semantic, hybrid, hybrid + ce-task blend 0.3 (reranker CO DINH);
            luong issue -- gold 134: semantic, hybrid ('query: ' hai phia).
  Nguong  : muc nho nhat f sao cho MOI muc >= f, CA HAI seed, khong thap hon model tham
            chieu (--ref) co y nghia:
              text : R@10 cua hybrid+ce o ca 4 lat (CI hi >= 0)
              issue: nDCG@10 cua hybrid tren gold (CI hi >= 0)
            Dev ~ +-0,03 -> nguong nay long; duong cong moi la ket qua chinh.

Dau ra: data/processed/learning_curve_dev.json
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
import eval_similar as es                                                  # noqa: E402
import resolution                                                          # noqa: E402
from eval_paraphrase import ci, paired                                     # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked                 # noqa: E402
from final_eval2 import QVERS, RERANK_K, SETS                              # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,     # noqa: E402
                       SearchEngine, fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS, E5_QUERY                                 # noqa: E402
from train_task_models import load_rows, subset_by_issue                   # noqa: E402

TEXT_CFGS = ("semantic", "hybrid", "hybrid+ce")
TEXT_METRICS = ("recall@10", "mrr@10", "ndcg@10")
GOLD_CFGS = ("semantic", "hybrid")
GOLD_METRICS = ("ndcg@10", "recall@10")


def model_info(name: str, path: str) -> dict:
    """Muc du lieu + seed doc tu task_meta.json; model khong co meta (model goc) = 0%."""
    meta_p = os.path.join(path, "task_meta.json")
    if not os.path.exists(meta_p):
        return {"name": name, "path": path, "fraction": 0.0, "seed": None}
    with open(meta_p, encoding="utf-8") as f:
        m = json.load(f)
    return {"name": name, "path": path, "fraction": float(m.get("fraction", 1.0)),
            "seed": m.get("seed"), "frac_seed": m.get("frac_seed", 13),
            "n_train_rows": m.get("n_train_rows")}


def threshold(passes: dict[float, list[bool]]) -> float | None:
    """Muc nho nhat f sao cho moi muc >= f deu dat o moi seed; khong co -> None."""
    best = None
    for f in sorted(passes, reverse=True):
        if not all(passes[f]):
            break
        best = f
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="ten=duong_dan")
    ap.add_argument("--ref", required=True, help="ten model tham chieu (100%% / seed 13)")
    ap.add_argument("--reranker", default="models/bge-reranker-base-task")
    ap.add_argument("--pairs", default="data/processed/task_pairs_v2.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chay thu: khong ghi file")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="data/processed/learning_curve_dev.json")
    args = ap.parse_args()

    models = [model_info(*m.split("=", 1)) for m in args.models]
    names = [m["name"] for m in models]
    if args.ref not in names or len(set(names)) != len(names):
        sys.exit("--ref phai nam trong --models va ten model khong duoc trung")
    pair_rows = load_rows(args.pairs)
    for m in models:
        sub = subset_by_issue(pair_rows, m["fraction"], m.get("frac_seed", 13)) if m["fraction"] > 0 else []
        m["n_issues"] = len({r["key"] for r in sub})
        m["n_pair_rows"] = len(sub)

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        dev = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])["dev"])

    # ------------------------------------------------ luong text, dev
    text = {}
    for s, (cpath, qpath) in SETS.items():
        corpus = load_jsonl(cpath)
        qs = [q for q in load_jsonl(qpath) if q["key"] in dev]
        if args.limit:
            qs = qs[: args.limit]
        base = SearchEngine(corpus, bm25_fold="auto")
        base.build_bm25()
        items = []
        for q in qs:
            rel = {base.key_to_idx[k] for k in q["relevant"] if k in base.key_to_idx}
            if rel:
                for qv, t in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                    items.append({"qv": qv, "text": t, "rel": rel})
        text[s] = {"corpus": corpus, "base": base, "items": items, "ranked": {}}

    cfg = {c: SearchConfig(name=c, top_k=100, **CONFIGS[c]) for c in ("semantic", "hybrid")}
    for m in models:
        t0 = time.perf_counter()
        for s, T in text.items():
            eng = SearchEngine(T["corpus"], bm25_fold="auto",
                               dense=DenseIndex(m["path"], device=args.device, batch_size=64,
                                                query_prefix="query: ", passage_prefix="passage: "))
            eng._bm25, eng._bm25_add = T["base"]._bm25, T["base"]._bm25_add
            eng.build_dense(show_progress=False)
            for c in cfg:
                T["ranked"][(m["name"], c)] = [eng.search(it["text"], cfg[c]) for it in T["items"]]
            del eng
            gc.collect()
        print(f"  text {m['name']}: {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    rr = CrossEncoderReranker(args.reranker, device=args.device, batch_size=args.ce_batch, fp16=True)
    for s, T in text.items():
        need = sorted({(i, int(d)) for m in models
                       for i, h in enumerate(T["ranked"][(m["name"], "hybrid")]) for d in h[:RERANK_K]})
        sc = rr.score_pairs([(T["items"][i]["text"], T["corpus"][d].get("text_dense", ""))
                             for i, d in need], show_progress=True)
        cache = dict(zip(need, sc.tolist()))
        for m in models:
            T["ranked"][(m["name"], "hybrid+ce")] = [
                np.concatenate([fuse_rerank(h[:RERANK_K],
                                            np.asarray([cache[(i, int(d))] for d in h[:RERANK_K]]),
                                            "blend", 0.3), h[RERANK_K:]])
                for i, h in enumerate(T["ranked"][(m["name"], "hybrid")])]
    print(f"  reranker: {time.perf_counter() - t0:.0f}s", flush=True)
    del rr
    gc.collect()

    result = {"design": "HANDOFF.md §7 Follow-up analyses (B)", "ref": args.ref,
              "reranker": args.reranker, "models": models, "n_boot": args.n_boot,
              "text": {}, "gold": {}}
    text_pass = {n: True for n in names}
    for s, T in text.items():
        for qv in QVERS:
            idx = [i for i, it in enumerate(T["items"]) if it["qv"] == qv]
            arr = {}
            for n in names:
                for c in TEXT_CFGS:
                    ms = [metrics_from_ranked(T["ranked"][(n, c)][i], T["items"][i]["rel"]) for i in idx]
                    arr[(n, c)] = {k: np.asarray([x[k] for x in ms]) for k in TEXT_METRICS}
            block = {"n": len(idx), "ci": {}, "vs_ref": {}}
            for n in names:
                block["ci"][n] = {c: {k: ci(arr[(n, c)][k], args.n_boot) for k in TEXT_METRICS}
                                  for c in TEXT_CFGS}
                if n == args.ref:
                    continue
                block["vs_ref"][n] = {c: {k: paired(arr[(n, c)][k], arr[(args.ref, c)][k], args.n_boot)
                                          for k in TEXT_METRICS} for c in TEXT_CFGS}
                if block["vs_ref"][n]["hybrid+ce"]["recall@10"]["hi"] < 0:
                    text_pass[n] = False
            result["text"][f"{s}:{qv}"] = block

    # ------------------------------------------------ luong issue, gold
    docs = load_jsonl(args.corpus)
    has_res = {d["key"]: resolution.has(d) for d in docs}
    gq = es.queries_from_gold(args.gold)
    if args.limit:
        gq = gq[: args.limit]
    bm25 = SearchEngine(docs)
    bm25.build_bm25()
    gold_arr, keys = {}, None
    for m in models:
        eng = SearchEngine(docs, dense=DenseIndex(m["path"], device=args.device, batch_size=64,
                                                  query_prefix=E5_QUERY, passage_prefix=E5_QUERY))
        eng._bm25 = bm25._bm25
        eng.build_dense(show_progress=False)
        for c in GOLD_CFGS:
            kept: list[str] = []
            sc = es.per_query(eng, gq, SearchConfig(name=c, top_k=10, **CONFIGS[c]), has_res, kept=kept)
            if keys is None:
                keys = kept
            elif kept != keys:
                sys.exit("gold: tap truy van khac nhau giua cac model")
            gold_arr[(m["name"], c)] = {k: sc[k] for k in GOLD_METRICS}
        del eng
        gc.collect()
    result["gold"] = {"n": len(keys or []), "ci": {}, "vs_ref": {}}
    gold_pass = {}
    for n in names:
        result["gold"]["ci"][n] = {c: {k: ci(gold_arr[(n, c)][k], args.n_boot) for k in GOLD_METRICS}
                                   for c in GOLD_CFGS}
        if n != args.ref:
            result["gold"]["vs_ref"][n] = {c: {k: paired(gold_arr[(n, c)][k], gold_arr[(args.ref, c)][k],
                                                         args.n_boot) for k in GOLD_METRICS}
                                           for c in GOLD_CFGS}
        gold_pass[n] = n == args.ref or result["gold"]["vs_ref"][n]["hybrid"]["ndcg@10"]["hi"] >= 0

    # ------------------------------------------------ duong cong + nguong
    def mean_over_slices(n, c, k):
        return round(float(np.mean([b["ci"][n][c][k]["mean"] for b in result["text"].values()])), 4)

    curve = []
    for m in models:
        n = m["name"]
        curve.append({"name": n, "fraction": m["fraction"], "seed": m["seed"],
                      "n_issues": m["n_issues"], "n_pair_rows": m["n_pair_rows"],
                      "text_r10_semantic": mean_over_slices(n, "semantic", "recall@10"),
                      "text_r10_hybrid_ce": mean_over_slices(n, "hybrid+ce", "recall@10"),
                      "text_mrr_hybrid_ce": mean_over_slices(n, "hybrid+ce", "mrr@10"),
                      "gold_ndcg_hybrid": result["gold"]["ci"][n]["hybrid"]["ndcg@10"]["mean"],
                      "text_pass": text_pass[n], "gold_pass": gold_pass[n]})
    curve.sort(key=lambda r: (r["fraction"], r["seed"] or 0))
    by_f_text, by_f_gold = {}, {}
    for r in curve:
        by_f_text.setdefault(r["fraction"], []).append(r["text_pass"])
        by_f_gold.setdefault(r["fraction"], []).append(r["gold_pass"])
    result["curve"] = curve
    result["threshold"] = {"text": threshold(by_f_text), "issue_gold": threshold(by_f_gold)}

    print(f"\n{'model':<12}{'muc':>6}{'seed':>6}{'issue':>7} | text R@10 sem  R@10 hy+ce  MRR hy+ce"
          f" | gold nDCG hy | dat (text, gold)")
    for r in curve:
        print(f"{r['name']:<12}{r['fraction']:>6.2f}{str(r['seed']):>6}{r['n_issues']:>7} | "
              f"{r['text_r10_semantic']:>13.3f}{r['text_r10_hybrid_ce']:>12.3f}{r['text_mrr_hybrid_ce']:>11.3f}"
              f" | {r['gold_ndcg_hybrid']:>12.3f} | {r['text_pass']}, {r['gold_pass']}")
    print(f"NGUONG (theo quy tac chot truoc): {result['threshold']}")
    if args.limit:
        print("--limit: khong ghi file")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
