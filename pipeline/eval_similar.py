# -*- coding: utf-8 -*-
"""
Danh gia truy hoi issue tuong dong voi NHAN PHAN BAC + chi so nghiep vu.

    # tren nhan cue thu hoi duoc (khong can gan tay)
    python pipeline/eval_similar.py --gold data/processed/similar_gold_eval.jsonl

    # tren nhan pooling do nguoi gan (bo bao cao chinh thuc)
    python pipeline/eval_similar.py --pooling data/processed/pooling_judgments_v3.csv

NAM CHI SO, DOC THEO THU TU NAY
  ndcg@K            xep hang co ton trong thu bac khong (2 quy hon 1)
  precision@K       trong K ket qua co bao nhieu issue lien quan (grade >= 1)
  recall@K (>=1)    co keo duoc tai lieu lien quan len khong
  recall@K (==2)    rieng nhom "giup truc tiep"
  resolution_hit@K  % truy van ma top-K co IT NHAT MOT tai lieu vua lien quan
                    VUA co ghi nhan cach xu ly dung duoc

Chi so cuoi la chi so NGHIEP VU, va no la con so duy nhat noi duoc gia tri that
cua he thong. Recall@K chi noi ve xep hang; mot he thong dat Recall@10 = 0,9
nhung moi tai lieu tra ve deu khong ghi cach xu ly thi khong giup ai ca.

Tran cua no bi chan boi du lieu, khong boi model: chi 9,0% issue trong corpus co
resolution dung duoc (982/10.925). Tren pool 769 phan quyet thi ty le la 12,7%,
va 31/40 truy van co it nhat mot tai lieu kem resolution -> tran ~77%.
Bao cao ca tran, neu khong con so tuyet doi se bi doc nham la model kem.

LUU Y KHI DOC SO TREN NHAN CUE
Nhan cue THIEU (mot truy van co the co 5 issue lien quan, comment chi ghi 1),
nen moi con so tren do la CAN DUOI, va nhan thieu thien vi CO HUONG: he thong
tra ve nhieu tai lieu lien quan chua duoc gan nhan thi bi phat nang hon. Dung
de so sanh cau hinh, dung dung nhu ty le tuyet doi. Nhan pooling khong co van
de nay -- do la ly do bo pooling ton tai.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution                                                        # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,  # noqa: E402
                       SearchEngine, load_jsonl)
from run_ablation import CONFIGS, E5_QUERY                                # noqa: E402


# ------------------------------------------------------------------ input
def queries_from_gold(path: str) -> list[dict]:
    """Cap co nhan -> truy van. Moi issue mot truy van, giu bac cao nhat."""
    nb: dict[str, dict[str, int]] = defaultdict(dict)
    for p in load_jsonl(path):
        a, b, g = p.get("a"), p.get("b"), int(p.get("grade", 1))
        if not (a and b):
            continue
        nb[a][b] = max(nb[a].get(b, 0), g)
        nb[b][a] = max(nb[b].get(a, 0), g)
    return [{"key": k, "rel": v} for k, v in sorted(nb.items())]


def queries_from_pooling(path: str) -> list[dict]:
    """CSV nguoi gan -> truy van. Dong chua dien `grade` bi bo qua."""
    nb: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = (r.get("grade") or "").strip()
            if g not in ("0", "1", "2"):
                continue
            nb[r["query_key"]][r["doc_key"]] = int(g)
    # Truy van khong co tai lieu nao >= 1 thi recall/ndcg khong xac dinh. Tinh
    # la 0 se keo MOI cau hinh xuong nhu nhau va lam loang khac biet giua chung
    # (vd RO-13080: ca 24 ung vien = 0). Chuan TREC: loai, va bao so luong.
    empty = sorted(k for k, v in nb.items() if not any(g >= 1 for g in v.values()))
    if empty:
        print(f"bo {len(empty)} truy van khong co tai lieu lien quan: {', '.join(empty)}")
    return [{"key": k, "rel": v} for k, v in sorted(nb.items())
            if v and k not in set(empty)]


# ---------------------------------------------------------------- metrics
def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2.0) for i, g in enumerate(gains))


def per_query(engine, queries, cfg, has_res: dict[str, bool], ks=(1, 3, 5, 10),
              kept: list | None = None):
    """`kept`: neu truyen list, ghi key cua cac truy van duoc tinh (cung thu tu mang)."""
    maxk = max(ks)
    run = SearchConfig(**{**cfg.__dict__, "top_k": maxk})
    out = defaultdict(list)
    for q in queries:
        if q["key"] not in engine.key_to_idx:
            continue
        rel = {k: g for k, g in q["rel"].items() if k in engine.key_to_idx}
        # loc theo corpus co the lam mat het tai lieu lien quan -> cung bo
        if not any(g >= 1 for g in rel.values()):
            continue
        if kept is not None:
            kept.append(q["key"])
        doc = engine.docs[engine.key_to_idx[q["key"]]]
        qtext = doc.get("text_dense") or doc.get("summary", "")
        ranked = engine.search(qtext, run,
                               exclude={engine.key_to_idx[q["key"]]})[:maxk]
        keys = [engine.keys[i] for i in ranked]
        gains = [float(rel.get(k, 0)) for k in keys]
        ideal = sorted((float(g) for g in rel.values()), reverse=True)
        n_pos = sum(1 for g in rel.values() if g >= 1)
        n_two = sum(1 for g in rel.values() if g == 2)
        for k in ks:
            idcg = dcg(ideal[:k])
            out[f"ndcg@{k}"].append(dcg(gains[:k]) / idcg if idcg else 0.0)
            got1 = sum(1 for x in keys[:k] if rel.get(x, 0) >= 1)
            # Mau so co dinh la K (TREC-style P@K), ke ca khi corpus nho hon K.
            # Trong corpus Jira that, engine luon tra du K ket qua.
            out[f"precision@{k}"].append(got1 / k)
            out[f"recall@{k}"].append(got1 / n_pos if n_pos else 0.0)
            if n_two:
                out[f"recall2@{k}"].append(
                    sum(1 for x in keys[:k] if rel.get(x, 0) == 2) / n_two)
            # chi so nghiep vu: co it nhat mot tai lieu vua lien quan vua co
            # ghi nhan cach xu ly
            out[f"resolution_hit@{k}"].append(
                1.0 if any(rel.get(x, 0) >= 1 and has_res.get(x)
                           for x in keys[:k]) else 0.0)
        rr = 0.0
        for i, x in enumerate(keys):
            if rel.get(x, 0) >= 1:
                rr = 1.0 / (i + 1)
                break
        out[f"mrr@{maxk}"].append(rr)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def boot(v: np.ndarray, n_boot: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)


def ci(v, n_boot):
    b = boot(v, n_boot)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def paired(a, b, n_boot):
    d = a - b
    bs = boot(d, n_boot)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


def unjudged_rate(eng, queries, cfg, k: int = 10) -> float:
    """Ty le tai lieu trong top-k khong co trong tap da cham cua truy van.

    Chi co nghia voi nhan pooling: q["rel"] chua ca tai lieu cham 0.
    """
    run = SearchConfig(**{**cfg.__dict__, "top_k": k})
    miss = total = 0
    for q in queries:
        if q["key"] not in eng.key_to_idx:
            continue
        doc = eng.docs[eng.key_to_idx[q["key"]]]
        qtext = doc.get("text_dense") or doc.get("summary", "")
        ranked = eng.search(qtext, run, exclude={eng.key_to_idx[q["key"]]})[:k]
        for i in ranked:
            total += 1
            miss += eng.keys[i] not in q["rel"]
    return miss / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--gold", default="")
    ap.add_argument("--pooling", default="")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--configs", nargs="*",
                    default=["keyword", "semantic", "hybrid"])
    ap.add_argument("--baseline", default="keyword")
    ap.add_argument("--metrics", nargs="*",
                    default=["ndcg@10", "precision@3", "precision@5",
                             "recall@10", "resolution_hit@10"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-base",
                    help="chi nap khi --configs co cau hinh rerank")
    ap.add_argument("--rerank-k", type=int, default=30,
                    help="so ung vien dau cua hybrid dua vao cross-encoder")
    ap.add_argument("--out", default="data/processed/eval_similar.json")
    args = ap.parse_args()

    if not (args.gold or args.pooling):
        sys.exit("can --gold hoac --pooling")
    docs = load_jsonl(args.corpus)
    queries = (queries_from_pooling(args.pooling) if args.pooling
               else queries_from_gold(args.gold))
    src = args.pooling or args.gold
    if not queries:
        sys.exit(f"khong co truy van nao tu {src} "
                 "(pooling: da dien cot `grade` chua?)")

    has_res = {d["key"]: resolution.has(d) for d in docs}
    n_res = sum(has_res.values())
    print(f"corpus {len(docs)} | {len(queries)} truy van | nguon: {src}")
    print(f"issue co resolution dung duoc: {n_res} ({n_res/len(docs):.1%}) "
          "-> tran cua resolution_hit")

    dense = DenseIndex(args.model, device="cpu", batch_size=32,
                       query_prefix=E5_QUERY, passage_prefix=E5_QUERY)
    # Truoc day engine khong duoc truyen reranker -> `hybrid+rerank` lang le
    # chay y het `hybrid` (SearchEngine bo qua rerank khi reranker is None).
    reranker = (CrossEncoderReranker(args.reranker)
                if any(CONFIGS[c]["rerank"] for c in args.configs) else None)
    eng = SearchEngine(docs, dense=dense, reranker=reranker)
    eng.build_bm25()
    if any(CONFIGS[c]["use_dense"] for c in args.configs):
        print("  dang nhung corpus (doc tu cache neu da co)...")
        eng.build_dense(show_progress=False)

    scores, unjudged = {}, {}
    for name in args.configs:
        cfg = SearchConfig(name=name, top_k=10, rerank_k=args.rerank_k,
                           **CONFIGS[name])
        scores[name] = per_query(eng, queries, cfg, has_res)
        if args.pooling:
            unjudged[name] = round(unjudged_rate(eng, queries, cfg), 4)
            print(f"  chay xong {name}  (top-10 chua cham: {unjudged[name]:.1%})")
        else:
            print(f"  chay xong {name}")

    result = {"source": src, "n_queries": len(queries),
              "model": args.model,
              "reranker": args.reranker if reranker else None,
              "rerank_k": args.rerank_k if reranker else None,
              "corpus_resolution_rate": round(n_res / len(docs), 4),
              "ci": {}, "paired": {}}
    if args.pooling:
        # Pool chi gom top-10 cua keyword/semantic/hybrid. Cau hinh keo tai lieu
        # ngoai pool len top-10 bi tinh 0 cho tai lieu do -> thien vi chong lai no.
        result["unjudged_top10"] = unjudged
    print("\n" + "=" * 78)
    print(f"KET QUA  (bootstrap {args.n_boot} lan)")
    print("=" * 78)
    print(f"{'cau hinh':<14}" + "".join(f"{m:>26}" for m in args.metrics))
    for name in args.configs:
        row = f"{name:<14}"
        result["ci"][name] = {}
        for m in args.metrics:
            v = scores[name].get(m)
            if v is None or not len(v):
                row += f"{'-':>26}"
                continue
            mean, lo, hi = ci(v, args.n_boot)
            result["ci"][name][m] = {"mean": round(mean, 4),
                                     "lo": round(lo, 4), "hi": round(hi, 4)}
            row += f"{mean:>10.4f} [{lo:.3f},{hi:.3f}]"
        print(row)

    base = args.baseline
    print(f"\nCHENH LECH so voi `{base}`")
    for name in args.configs:
        if name == base:
            continue
        row = f"{name:<14}"
        result["paired"][name] = {}
        for m in args.metrics:
            a, b = scores[name].get(m), scores[base].get(m)
            if a is None or b is None or len(a) != len(b) or not len(a):
                row += f"{'-':>26}"
                continue
            d, lo, hi, sig = paired(a, b, args.n_boot)
            result["paired"][name][m] = {"delta": round(d, 4), "lo": round(lo, 4),
                                         "hi": round(hi, 4), "significant": sig}
            row += f"{d:>+10.4f} [{lo:+.3f},{hi:+.3f}]{'*' if sig else ' '}"
        print(row)
    print("  (* = khoang tin cay 95% khong chua 0)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")
    if args.gold:
        print("\nNHAC: nhan cue THIEU -> moi con so tren la CAN DUOI, va nhan"
              " thieu thien vi\nchong lai he thong tra ve nhieu tai lieu lien"
              " quan chua duoc gan nhan.\nDung de SO SANH cau hinh, khong dung"
              " nhu ty le tuyet doi.")


if __name__ == "__main__":
    main()
