# -*- coding: utf-8 -*-
"""
Thu cach sua truy van KHONG DAU: them ban bo dau vao BM25 (ca doc lan truy van).

    # buoc 1: chon tren hybrid (nhanh, khong cross-encoder)
    python pipeline/eval_bm25_fold.py --stage 1
    # buoc 2: xac nhan bien the da chon tren hybrid+rerank (CPU ~2,5 gio)
    python pipeline/eval_bm25_fold.py --stage 2 --variants none add

BOI CANH
`eval_undiacritized.py` (dev 500): bo dau truy van lam R@10 tut ~15 diem o moi
cau hinh; keyword cung tut vi "dang nhap" khong khop "đăng nhập". Cach sua re
nhat, khong can model moi: `retrieval.tokenize(fold=...)`.
    none  hanh vi cu
    add   token goc + ban bo dau
    only  bo dau toan bo
Dense (e5) KHONG doi -- cach sua nay chi cham BM25.

QUY TAC CHON -- CHOT TRUOC KHI DO (2026-09-14). Chi DEV; test 2.675 khong cham.
Moi truy van do HAI ban: goc (co dau) va khong dau. Mot bien the DAT khi:
    (a) khong dau: R@10(bien the - none) > 0 va CI khong chua 0
    (b) goc      : R@10(bien the - none) KHONG giam co y nghia (hi >= 0)
Buoc 1 xet cau hinh `hybrid`; nhieu bien the dat -> lay bien the co (a) lon
nhat; khong bien the nao dat -> dung, khong sua.
Buoc 2 ap dung (a)+(b) cho `hybrid+rerank` (cau hinh cua luong text); khong dat
-> khong dua vao search.py.

Dau ra: data/processed/bm25_fold_stage<1|2>.json
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

from eval_paraphrase import ci, paired                                  # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked               # noqa: E402
from retrieval import (FOLD_MODES, CrossEncoderReranker, DenseIndex,     # noqa: E402
                       SearchConfig, SearchEngine, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS, E5_QUERY                               # noqa: E402

METRICS = ("recall@10", "mrr@10", "ndcg@10", "recall@30")
QVERS = ("goc", "khong_dau")


def answer_rank(ranked, rel) -> int | None:
    """Hang (1-based) cua dap an dau tien trong danh sach; None neu khong co."""
    return next((r + 1 for r, i in enumerate(ranked) if i in rel), None)


def per_query_row(eng, q, qv, fold, text, rel, hybrid, reranked, ce_scores, k) -> dict:
    keys = eng.keys
    orig = q["query"]
    toks = orig.split()
    vn = sum(1 for t in toks if strip_diacritics(t) != t)
    return {
        "key": q["key"], "qv": qv, "fold": fold,
        "rank_hybrid": answer_rank(hybrid, rel),
        "rank_rerank": answer_rank(reranked, rel),
        "rerank_k": k,
        "hybrid_top10": [keys[i] for i in hybrid[:10]],
        "rerank_top10": [keys[i] for i in reranked[:10]],
        "ce_top10": [round(float(s), 4) for s in ce_scores[:10]],
        "ce_margin": round(float(ce_scores[0] - ce_scores[1]), 4) if len(ce_scores) > 1 else None,
        "overlap10": len(set(hybrid[:10].tolist()) & set(reranked[:10].tolist())),
        "n_tok": len(toks), "vn_ratio": round(vn / max(1, len(toks)), 3),
        "lexical_group": q.get("lexical_group"), "jac_bin": q.get("jac_bin"),
        "query": text,
    }


def metrics(ranked, rel) -> dict:
    m = metrics_from_ranked(ranked, rel)
    m["recall@30"] = sum(1.0 for i in ranked[:30] if i in rel) / len(rel)
    return m


def qualifies(block: dict, variant: str, cfg: str) -> dict:
    """(a) + (b) cua quy tac tren. block = result[...]['paired']."""
    nd = block["khong_dau"][cfg][f"{variant} - none"]["recall@10"]
    go = block["goc"][cfg][f"{variant} - none"]["recall@10"]
    a = nd["delta"] > 0 and nd["significant"]
    b = go["hi"] >= 0
    return {"a_undiacritized_gain": a, "b_no_diacritic_loss": b, "ok": a and b,
            "gain": nd["delta"]}


def choose(block: dict, variants, cfg: str) -> dict:
    checks = {v: qualifies(block, v, cfg) for v in variants if v != "none"}
    ok = [v for v, c in checks.items() if c["ok"]]
    best = max(ok, key=lambda v: checks[v]["gain"]) if ok else "none"
    return {"config": cfg, "checks": checks, "chosen": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rerank-k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-per-query", default="",
                    help="stage 2: ghi JSONL tung truy van (hang dap an hybrid/rerank, "
                         "top-10, diem cross-encoder) cho analyze_rerank_damage.py")
    args = ap.parse_args()

    variants = args.variants or (list(FOLD_MODES) if args.stage == 1 else None)
    if not variants or variants[0] != "none" or set(variants) - set(FOLD_MODES):
        sys.exit(f"--variants phai bat dau bang 'none', chi gom {FOLD_MODES}")
    cfgs = ("keyword", "hybrid") + (("hybrid+rerank",) if args.stage == 2 else ())
    out = args.out or f"data/processed/bm25_fold_stage{args.stage}.json"

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        dev = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])["dev"])
    corpus = load_jsonl(args.corpus)
    queries = [q for q in load_jsonl(args.queries) if q["key"] in dev]
    if args.limit:
        queries = random.Random(13).sample(queries, min(args.limit, len(queries)))

    dense = DenseIndex(args.model, device=args.device, batch_size=64,
                       query_prefix=E5_QUERY, passage_prefix="passage: ")
    rr = CrossEncoderReranker(args.reranker, device=args.device) if args.stage == 2 else None
    cfg_kw = SearchConfig(name="keyword", top_k=100, **CONFIGS["keyword"])
    cfg_hy = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])

    scores = {qv: {c: {v: {m: [] for m in METRICS} for v in variants} for c in cfgs}
              for qv in QVERS}
    items = None
    dump = open(args.dump_per_query, "w", encoding="utf-8") if (args.dump_per_query and rr) else None
    for v in variants:
        eng = SearchEngine(corpus, dense=dense, bm25_fold=v)
        t0 = time.perf_counter()
        eng.build_bm25()
        eng.build_dense(show_progress=False)
        if items is None:
            items = [(q, {eng.key_to_idx[k] for k in q["relevant"] if k in eng.key_to_idx})
                     for q in queries]
            items = [(q, rel) for q, rel in items if rel]
            print(f"dev: {len(items)} truy van | bien the {variants} | cau hinh {cfgs}")
        print(f"\n== fold={v} (BM25 {time.perf_counter() - t0:.0f}s) ==", flush=True)
        for j, (q, rel) in enumerate(items):
            for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                runs = {"keyword": eng.search(text, cfg_kw)}
                runs["hybrid"] = eng.search(text, cfg_hy)
                if rr is not None:
                    hy = runs["hybrid"]
                    head = hy[: args.rerank_k]
                    texts = [eng.docs[i].get(eng.rerank_field, "") for i in head]
                    head, ce = rr.rerank_with_scores(text, texts, head)
                    runs["hybrid+rerank"] = np.concatenate([head, hy[args.rerank_k:]])
                    if dump is not None:
                        dump.write(json.dumps(per_query_row(
                            eng, q, qv, v, text, rel, hy, runs["hybrid+rerank"], ce,
                            args.rerank_k), ensure_ascii=False) + "\n")
                for c, ranked in runs.items():
                    for m, val in metrics(ranked, rel).items():
                        scores[qv][c][v][m].append(val)
            if (j + 1) % 100 == 0:
                print(f"    {j + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del eng
    if dump is not None:
        dump.close()
        print(f"-> {args.dump_per_query}")

    arr ={qv: {c: {v: {m: np.asarray(x) for m, x in d.items()} for v, d in cc.items()}
                for c, cc in qq.items()} for qv, qq in scores.items()}
    low = np.asarray([i for i, (q, _) in enumerate(items) if q.get("jac_bin") == "<0.2"])

    def block(idx):
        sel = (lambda a: a) if idx is None else (lambda a: a[idx])
        return {
            "n": len(items) if idx is None else int(len(idx)),
            "ci": {qv: {c: {v: {m: ci(sel(arr[qv][c][v][m]), args.n_boot) for m in METRICS}
                            for v in variants} for c in cfgs} for qv in QVERS},
            "paired": {qv: {c: {f"{v} - none": {m: paired(sel(arr[qv][c][v][m]),
                                                          sel(arr[qv][c]["none"][m]),
                                                          args.n_boot) for m in METRICS}
                                for v in variants if v != "none"} for c in cfgs}
                       for qv in QVERS},
        }

    result = {"stage": args.stage, "split": "dev", "n_queries": len(items),
              "limit": args.limit, "variants": variants, "configs": list(cfgs),
              "model": args.model, "reranker": args.reranker if rr else None,
              "overall": block(None)}
    if len(low):
        result["jac_lt_0.2"] = block(low)
    decide_cfg = "hybrid" if args.stage == 1 else "hybrid+rerank"
    result["choice"] = choose(result["overall"]["paired"], variants, decide_cfg)

    for key in ("overall", "jac_lt_0.2"):
        if key not in result:
            continue
        b = result[key]
        print(f"\n== {key} (n={b['n']}) ==  cot: " + ", ".join(METRICS))
        for qv in QVERS:
            for c in cfgs:
                for v in variants:
                    x = b["ci"][qv][c][v]
                    print(f"  {qv:<10}{c:<15}{v:<5}" + "  ".join(f"{x[m]['mean']:.3f}" for m in METRICS))
                for name, d in b["paired"][qv][c].items():
                    print(f"  {'':<25}{name:<11}" + " ".join(
                        f"{y['delta']:+.4f}[{y['lo']:+.3f},{y['hi']:+.3f}]{'*' if y['significant'] else ' '}"
                        for y in d.values()))
    print(f"\nCHON THEO QUY TAC ({decide_cfg}): {json.dumps(result['choice'], ensure_ascii=False)}")

    if args.limit:
        print("\n--limit: khong ghi file")
        return
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
