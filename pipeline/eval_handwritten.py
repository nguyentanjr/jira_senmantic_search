# -*- coding: utf-8 -*-
"""
Kiem tra PHU tren 50 truy van VIET TAY (khong phai cat tu issue).

    python pipeline/eval_handwritten.py

VI SAO
Bo paraphrase va cap train deu dung "actual -> issue" nen model co the chi quen
van phong do. O day truy van duoc viet lai bang loi khac, kieu ky su hay go.

GIOI HAN -- GHI RO KHI BAO CAO
- Nguoi viet la tro ly AI, DA DOC issue dap an (tom tat + actual) truoc khi viet
  -> co the vo tinh dung lai tu. Do tu vung trung (jaccard) duoc bao cao kem.
- Khong phai van phong ky su that. n = 50 -> CI rong.
- KHONG dung de chon cau hinh; chi kiem tra chieu huong ket qua test.

QUY TAC VIET (chot truoc khi do, 2026-09-14): 50 issue ngau nhien tu TEST
(seed 20260914, `handwritten_sample_keys.json`); <= ~15 tu; tron Viet/Anh;
mo ta trieu chung, khong mo ta cach sua; khong chep cum >= 3 tu; giu thuat ngu
nghiep vu; ~1/5 truy van giu ma loi.

DO: corpus.jsonl DAY DU (issue giu actual -- dung nhu tim that), hai ban co dau
/ khong dau, BM25 fold="add", rerank top-30, blend 0.3.
    keyword | cu = e5 + bge-reranker | moi = e5-task + ce-task (+ hybrid tung dense)

Dau ra: data/processed/handwritten_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paraphrase_eval import jaccard, toks                            # noqa: E402
from eval_bm25_fold import answer_rank                                     # noqa: E402
from eval_paraphrase import ci, paired                                     # noqa: E402
from eval_rerankers import metrics_from_ranked                             # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,     # noqa: E402
                       SearchEngine, fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS                                           # noqa: E402

SYSTEMS = {"cu": ("intfloat/multilingual-e5-base", "BAAI/bge-reranker-base"),
           "moi": ("models/e5-base-task", "models/bge-reranker-base-task")}
QVERS = ("goc", "khong_dau")
METRICS = ("recall@10", "mrr@10", "ndcg@10")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/processed/handwritten_queries.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--system", nargs=3, action="append", metavar=("TEN", "DENSE", "RERANKER"),
                    help="thay he mac dinh (cu/moi), lap lai duoc")
    ap.add_argument("--goc-fold", default="none", choices=["none", "add", "only"],
                    help="che do BM25 cho ban CO DAU (ban khong dau luon dung add). "
                         "add dem tu tieng Viet hai lan -> hai truy van ngan co dau")
    ap.add_argument("--out", default="data/processed/handwritten_eval.json")
    args = ap.parse_args()

    corpus = load_jsonl(args.corpus)
    qs = load_jsonl(args.queries)
    base = SearchEngine(corpus, bm25_fold="add")
    base.build_bm25()
    bm25_by_qv = {"khong_dau": base._bm25}
    if args.goc_fold == "add":
        bm25_by_qv["goc"] = base._bm25
    else:
        g = SearchEngine(corpus, bm25_fold=args.goc_fold)
        bm25_by_qv["goc"] = g.build_bm25()
    items = []
    for q in qs:
        d = corpus[base.key_to_idx[q["key"]]]
        j = jaccard(toks(q["query"]), toks(d.get("text_bm25", "")))
        for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
            items.append({"key": q["key"], "qv": qv, "text": text,
                          "rel": {base.key_to_idx[q["key"]]}, "jaccard": round(j, 4)})

    def run(eng, it, cfg):
        """Chon chi muc BM25 theo ban truy van (fold cua chi muc phai khop tokenizer)."""
        eng._bm25 = bm25_by_qv[it["qv"]]
        eng.bm25_fold = args.goc_fold if it["qv"] == "goc" else "add"
        return eng.search(it["text"], cfg)

    ranked = {"keyword": [run(base, it, SearchConfig(name="keyword", top_k=100,
                                                     **CONFIGS["keyword"])) for it in items]}
    systems = {n: (dp, rp) for n, dp, rp in args.system} if args.system else SYSTEMS
    for sname, (dense_path, rr_path) in systems.items():
        eng = SearchEngine(corpus, bm25_fold="add",
                           dense=DenseIndex(dense_path, device=args.device, batch_size=64,
                                            query_prefix="query: ", passage_prefix="passage: "))
        eng._bm25 = base._bm25
        eng.build_dense(show_progress=True)
        cfg = SearchConfig(name="hybrid", top_k=100, **CONFIGS["hybrid"])
        hy = [run(eng, it, cfg) for it in items]
        rr = CrossEncoderReranker(rr_path, device=args.device, batch_size=64, fp16=True)
        pairs = [(it["text"], corpus[d].get("text_dense", "")) for it, h in zip(items, hy)
                 for d in h[:30]]
        s = rr.score_pairs(pairs, show_progress=True)
        out, k = [], 0
        for h in hy:
            n = len(h[:30])
            out.append(np.concatenate([fuse_rerank(h[:30], s[k:k + n], "blend", 0.3), h[30:]]))
            k += n
        ranked[f"{sname}_hybrid"] = hy
        ranked[sname] = out
        print(f"xong he {sname}", flush=True)
        del eng, rr

    names = list(ranked)
    arr = {qv: {n: {m: [] for m in METRICS} for n in names} for qv in QVERS}
    rows = []
    for i, it in enumerate(items):
        row = {"key": it["key"], "qv": it["qv"], "jaccard": it["jaccard"]}
        for n in names:
            for m, v in metrics_from_ranked(ranked[n][i], it["rel"]).items():
                arr[it["qv"]][n][m].append(v)
            row[f"rank_{n}"] = answer_rank(ranked[n][i][:100], it["rel"])
        rows.append(row)
    arr = {qv: {n: {m: np.asarray(v) for m, v in d.items()} for n, d in dd.items()}
           for qv, dd in arr.items()}

    result = {"n_queries": len(qs), "goc_fold": args.goc_fold,
              "note": "truy van do tro ly AI viet sau khi doc issue; xem docstring",
              "mean_jaccard": round(float(np.mean([r["jaccard"] for r in rows[::2]])), 4),
              "ci": {}, "vs_keyword": {}, "moi_vs_cu": {}, "per_query": rows}
    for qv in QVERS:
        result["ci"][qv] = {n: {m: ci(arr[qv][n][m], args.n_boot) for m in METRICS} for n in names}
        result["vs_keyword"][qv] = {n: {m: paired(arr[qv][n][m], arr[qv]["keyword"][m], args.n_boot)
                                        for m in METRICS} for n in names if n != "keyword"}
        s_names = list(systems)
        if len(s_names) >= 2:
            result["moi_vs_cu"][qv] = {m: paired(arr[qv][s_names[-1]][m], arr[qv][s_names[0]][m], args.n_boot)
                                       for m in METRICS}

    print(f"\n{len(qs)} truy van viet tay | jaccard trung binh voi issue {result['mean_jaccard']}")
    for qv in QVERS:
        print(f"\n== {qv} ==  R@10 MRR@10 nDCG@10 | vs keyword")
        for n in names:
            x = result["ci"][qv][n]
            line = f"  {n:<12}" + " ".join(f"{x[m]['mean']:.3f}" for m in METRICS)
            if n != "keyword":
                d = result["vs_keyword"][qv][n]
                line += " | " + " ".join(
                    f"{d[m]['delta']:+.3f}[{d[m]['lo']:+.3f},{d[m]['hi']:+.3f}]{'*' if d[m]['significant'] else ' '}"
                    for m in METRICS)
            print(line)
        d = result["moi_vs_cu"][qv]
        print("  moi - cu    " + " ".join(
            f"{d[m]['delta']:+.3f}[{d[m]['lo']:+.3f},{d[m]['hi']:+.3f}]{'*' if d[m]['significant'] else ' '}"
            for m in METRICS))
    print("\nhang dap an (co dau): key | jaccard | keyword cu moi")
    for r in rows[::2]:
        print(f"  {r['key']:<9} {r['jaccard']:.2f} | {r['rank_keyword']} {r['rank_cu']} {r['rank_moi']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
