# -*- coding: utf-8 -*-
"""
Do keyword / semantic / hybrid tren BO PARAPHRASE (bo danh gia 1).

    python pipeline/eval_paraphrase.py
    python pipeline/eval_paraphrase.py --limit 300      # chay thu cho nhanh

Chay `build_paraphrase_eval.py` truoc de sinh corpus_holdout + truy van.

Dung dung engine (`retrieval.SearchEngine`) va dung bo cau hinh (`run_ablation.
CONFIGS`) nhu `significance.py`, de so moi va so cu (significance_summary.json)
doc duoc canh nhau.

KHAC BIET VE DON VI DOC LAP
`significance.py` bootstrap theo CAP vi hai truy van a->b va b->a cua cung mot
cap khong doc lap. O day moi truy van sinh tu mot issue rieng nen ban than truy
van da la don vi doc lap -> bootstrap thuong (xem `_boot_means`).

CHE DO e5
Truy van la mo ta trieu chung (ngan), tai lieu la ca issue (dai) -> BAT DOI
XUNG: query='query: ', passage='passage: '. Dung sai khong bao loi, chi lam tut
7-19%.

LAT CAT LA PHAN QUAN TRONG NHAT
Bang tong the it y nghia. Cai can doc la lat cat theo `jac_bin`:

    jac_bin '<0.2'  truy van va tai lieu gan nhu khong chung tu vung.
                    Day la vung keyword search bat luc. Neu retrieval ngu
                    nghia co gia tri thi phai the hien o day.
    jac_bin '>=0.4' tu vung trung nhieu -> BM25 von da manh.

Neu semantic thang o lat cat thap va thua o lat cat cao, do la ket qua DUNG
nhu ly thuyet va la luan diem trung tam cua de tai. Bao cao ca hai chieu.

Dau ra: data/processed/paraphrase_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,
                       SearchEngine, load_jsonl)
from run_ablation import CONFIGS, E5_QUERY
# cluster_bootstrap khong dung o day -- xem _boot_means (moi truy van la mot cum)


def per_query(engine, queries, cfg, ks=(1, 5, 10)):
    """-> {metric: np.array theo tung truy van}. Truy van la van ban tu do."""
    maxk = max(ks)
    run = SearchConfig(**{**cfg.__dict__, "top_k": maxk})
    out = defaultdict(list)
    kept = []
    for qi, q in enumerate(queries):
        rel = {engine.key_to_idx[k] for k in q["relevant"]
               if k in engine.key_to_idx}
        if not rel:
            continue
        ranked = engine.search(q["query"], run)
        hits = [1.0 if i in rel else 0.0 for i in ranked[:maxk]]
        for k in ks:
            got = sum(hits[:k])
            out[f"recall@{k}"].append(got / len(rel))
            out[f"hit@{k}"].append(1.0 if got else 0.0)
            dcg = sum(v / np.log2(r + 2.0) for r, v in enumerate(hits[:k]))
            idcg = sum(1.0 / np.log2(r + 2.0) for r in range(min(len(rel), k)))
            out[f"ndcg@{k}"].append(dcg / idcg if idcg else 0.0)
        rr = 0.0
        for r, v in enumerate(hits):
            if v:
                rr = 1.0 / (r + 1)
                break
        out[f"mrr@{maxk}"].append(rr)
        kept.append(qi)
        if (qi + 1) % 500 == 0:
            print(f"    ... {qi + 1}/{len(queries)}", flush=True)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, kept


def _boot_means(values: np.ndarray, n_boot: int, seed: int = 17) -> np.ndarray:
    """Bootstrap tren truy van.

    Moi truy van sinh tu mot issue rieng -> moi truy van la mot cum co dung
    mot phan tu. `cluster_bootstrap` khi do quy ve bootstrap thuong, nhung cai
    dat cua no gom mang theo tung cum bang vong lap Python (3.830 cum x 5.000
    lan = qua cham). Duong nhanh nay cho ket qua tuong duong ve mat thong ke.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def ci(values: np.ndarray, n_boot: int) -> dict:
    bs = _boot_means(values, n_boot)
    return {"mean": round(float(values.mean()), 4),
            "lo": round(float(np.percentile(bs, 2.5)), 4),
            "hi": round(float(np.percentile(bs, 97.5)), 4)}


def paired(a: np.ndarray, b: np.ndarray, n_boot: int) -> dict:
    """a - b, bootstrap tren hieu theo tung truy van."""
    d = a - b
    bs = _boot_means(d, n_boot)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"delta": round(float(d.mean()), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p": round(float(min(p, 1.0)), 4),
            "significant": bool(lo > 0 or hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--queries", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--configs", nargs="*",
                    default=["keyword", "semantic", "hybrid"])
    ap.add_argument("--baseline", default="keyword")
    ap.add_argument("--metrics", nargs="*",
                    default=["recall@10", "mrr@10", "ndcg@10"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys-file", default="",
                    help="chi do tren truy van co key trong file nay: JSON co "
                         "`paraphrase_clean_keys` (build_finetune_pairs.py) hoac "
                         "mot list key. Bat buoc khi do model da fine-tune.")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base",
                    help="chi nap khi --configs co cau hinh rerank")
    ap.add_argument("--rerank-k", type=int, default=30,
                    help="so ung vien dau cua hybrid dua vao cross-encoder")
    ap.add_argument("--symmetric", action="store_true",
                    help="dung 'query: ' cho ca hai phia (mac dinh: bat doi xung)")
    ap.add_argument("--out", default="data/processed/paraphrase_eval.json")
    args = ap.parse_args()

    for p in (args.corpus, args.queries):
        if not os.path.exists(p):
            sys.exit(f"khong thay {p} -- chay build_paraphrase_eval.py truoc")

    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)
    # Model fine-tune da hoc cap duong chua `actual` cua mot so issue -- do tren
    # cac truy van do la do lai tap train. Loc TRUOC --limit de hai lan do cung
    # tap truy van.
    if args.keys_file:
        with open(args.keys_file, encoding="utf-8") as f:
            ks = json.load(f)
        ks = set(ks["paraphrase_clean_keys"] if isinstance(ks, dict) else ks)
        queries = [q for q in queries if q["key"] in ks]
        print(f"loc theo {args.keys_file}: con {len(queries)} truy van")
    if args.limit:
        queries = queries[: args.limit]

    passage_prefix = E5_QUERY if args.symmetric else "passage: "
    print(f"corpus {len(corpus)} | {len(queries)} truy van paraphrase")
    print(f"che do e5: {'DOI XUNG' if args.symmetric else 'BAT DOI XUNG'} "
          f"(query='{E5_QUERY}' passage='{passage_prefix}')")

    dense = DenseIndex(args.model, device="cpu", batch_size=32,
                       query_prefix=E5_QUERY, passage_prefix=passage_prefix)
    # Truoc day engine khong duoc truyen reranker -> `hybrid+rerank` lang le
    # chay y het `hybrid` (SearchEngine bo qua rerank khi reranker is None).
    reranker = (CrossEncoderReranker(args.reranker)
                if any(CONFIGS[c]["rerank"] for c in args.configs) else None)
    eng = SearchEngine(corpus, dense=dense, reranker=reranker)
    eng.build_bm25()
    if any(CONFIGS[c]["use_dense"] for c in args.configs):
        print("  dang nhung corpus (lan dau se lau, sau do doc tu cache)...")
        eng.build_dense(show_progress=False)

    scores, kept_idx = {}, None
    for name in args.configs:
        cfg = SearchConfig(name=name, top_k=10, rerank_k=args.rerank_k,
                           **CONFIGS[name])
        scores[name], kept_idx = per_query(eng, queries, cfg)
        print(f"  chay xong {name}")

    kept = [queries[i] for i in kept_idx]
    result = {
        "n_corpus": len(corpus),
        "n_queries": len(kept),
        "asymmetric": not args.symmetric,
        "model": args.model,
        "reranker": args.reranker if reranker else None,
        "rerank_k": args.rerank_k if reranker else None,
        "overall": {"ci": {}, "paired": {}},
        "by_rare_bin": {},
        "by_jac_bin": {},
        "by_lexical_group": {},
    }

    # ---------------------------------------------------------- tong the
    print("\n" + "=" * 78)
    print(f"TONG THE  (bootstrap {args.n_boot} lan, moi truy van la mot cum)")
    print("=" * 78)
    hdr = f"{'cau hinh':<16}" + "".join(f"{m:>26}" for m in args.metrics)
    print(hdr)
    for name in args.configs:
        row = f"{name:<16}"
        result["overall"]["ci"][name] = {}
        for m in args.metrics:
            c = ci(scores[name][m], args.n_boot)
            result["overall"]["ci"][name][m] = c
            row += f"{c['mean']:>10.4f} [{c['lo']:.3f},{c['hi']:.3f}]"
        print(row)

    base = args.baseline
    print(f"\nCHENH LECH so voi `{base}` (cap doi tren cung truy van)")
    for name in args.configs:
        if name == base:
            continue
        row = f"{name:<16}"
        result["overall"]["paired"][name] = {}
        for m in args.metrics:
            d = paired(scores[name][m], scores[base][m], args.n_boot)
            result["overall"]["paired"][name][m] = d
            mark = "*" if d["significant"] else " "
            row += f"{d['delta']:>+10.4f} [{d['lo']:+.3f},{d['hi']:+.3f}]{mark}"
        print(row)
    print("  (* = khoang tin cay 95% khong chua 0)")

    # ---------------------------------------------------------- lat cat
    def slice_report(field: str, out_key: str, title: str) -> None:
        groups = defaultdict(list)
        for i, q in enumerate(kept):
            groups[str(q.get(field))].append(i)
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        for g in sorted(groups):
            idx = np.asarray(groups[g])
            if len(idx) < 20:
                print(f"\n{g}  (n={len(idx)}) -- bo qua, qua it truy van")
                continue
            print(f"\n{g}  (n={len(idx)})")
            result[out_key][g] = {"n": int(len(idx)), "ci": {}, "paired": {}}
            for name in args.configs:
                row = f"  {name:<14}"
                result[out_key][g]["ci"][name] = {}
                for m in args.metrics:
                    c = ci(scores[name][m][idx], args.n_boot)
                    result[out_key][g]["ci"][name][m] = c
                    row += f"{c['mean']:>10.4f} [{c['lo']:.3f},{c['hi']:.3f}]"
                print(row)
            for name in args.configs:
                if name == base:
                    continue
                row = f"  {name + ' vs ' + base:<14}"
                result[out_key][g]["paired"][name] = {}
                for m in args.metrics:
                    d = paired(scores[name][m][idx], scores[base][m][idx],
                               args.n_boot)
                    result[out_key][g]["paired"][name][m] = d
                    mark = "*" if d["significant"] else " "
                    row += f"{d['delta']:>+10.4f} [{d['lo']:+.3f},{d['hi']:+.3f}]{mark}"
                print(row)

    slice_report("rare_bin", "by_rare_bin",
                 "THEO TOKEN HIEM DUNG CHUNG  (none = BM25 khong co ma/ID de bam)")
    slice_report("jac_bin", "by_jac_bin",
                 "THEO DO TRUNG TU VUNG  (<0.2 = vung keyword bat luc)")
    slice_report("lexical_group", "by_lexical_group",
                 "THEO NHOM TIN HIEU  (True = co ma loi/API/class)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
