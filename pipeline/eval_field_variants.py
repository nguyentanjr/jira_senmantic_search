# -*- coding: utf-8 -*-
"""Do cac bien the TRUONG TIM KIEM cho luong issue tren gold 134 / pooling v4.

Cau hoi: thong tin quyet dinh co dang chim trong mo ta dai khong? `text_dense`
hien tai da la summary + actual + expected; cai CHUA co la dieu kien kich hoat
(sections.steps, precondition) va ma loi (signals.error_code, nay chi nam trong
text_bm25). Script dung lai chinh cac truong da co trong corpus - khong goi LLM,
khong ton han muc.

Moi bien the phai nhung lai toan bo corpus nen CPU rat cham; chay tren GPU bang
`colab/field_variants_colab.ipynb`.

So sanh ghep cap voi bien the nen (`--baseline`) bang bootstrap, vi cung mot tap
truy van nen khong duoc dung kiem dinh doc lap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl
from eval_similar import ci, paired
from search import FLOWS


def join(*xs: str) -> str:
    return "\n".join(x for x in xs if x).strip()


def sec(d: dict, k: str) -> str:
    return (d.get("sections") or {}).get(k, "") or ""


def codes(d: dict) -> str:
    return " ".join((d.get("signals") or {}).get("error_code", []) or [])


def body(d: dict) -> str:
    return sec(d, "actual") or d.get("text_business", "") or ""


VARIANTS = {
    "base": lambda d: d.get("text_dense") or d.get("summary", ""),
    "codes": lambda d: join(d.get("text_dense") or d.get("summary", ""), codes(d)),
    "steps": lambda d: join(d.get("summary", ""), sec(d, "steps"), body(d),
                            sec(d, "expected")),
    "steps_precond_codes": lambda d: join(d.get("summary", ""), sec(d, "precondition"),
                                          sec(d, "steps"), body(d), sec(d, "expected"),
                                          codes(d)),
    "no_expected": lambda d: join(d.get("summary", ""), body(d)),
    "summary_only": lambda d: d.get("summary", ""),
    "labelled": lambda d: join(
        "Loi: " + d.get("summary", ""),
        ("Hien trang: " + sec(d, "actual")) if sec(d, "actual") else "",
        ("Mong doi: " + sec(d, "expected")) if sec(d, "expected") else "",
        ("Ma loi: " + codes(d)) if codes(d) else ""),
}


def load_gold(path: str) -> dict:
    rel = collections.defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            a, b, g = r["a"].upper(), r["b"].upper(), int(r["grade"])
            rel[a][b] = max(rel[a].get(b, 0), g)
            rel[b][a] = max(rel[b].get(a, 0), g)
    return rel


def load_pooling(path: str) -> dict:
    rel = collections.defaultdict(dict)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            g = (r.get("grade") or "").strip()
            if g in {"0", "1", "2"}:
                rel[r["query_key"].upper()][r["doc_key"].upper()] = int(g)
    return rel


def ndcg(keys: list, rel: dict, k: int) -> float:
    dcg = sum((2 ** rel.get(x, 0) - 1) / math.log2(i + 2) for i, x in enumerate(keys[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def measure(docs_list: list, rel_map: dict, fn, flow: dict, args) -> dict:
    """Tra ve diem TUNG TRUY VAN de con so sanh ghep cap."""
    dl = [{**d, "text_dense": fn(d)} for d in docs_list]
    docs = {d["key"].upper(): d for d in dl}
    dense = DenseIndex(flow["model"], device=args.device, batch_size=args.batch_size,
                       query_prefix=flow["query_prefix"],
                       passage_prefix=flow["passage_prefix"])
    eng = SearchEngine(dl, dense=dense, reranker=None, bm25_fold=flow["bm25_fold"])
    eng.build_bm25()
    eng.build_dense(show_progress=False)
    cfg = SearchConfig(use_bm25=True, use_dense=True, rerank=False,
                       candidate_k=args.candidate_k, top_k=args.candidate_k)
    out = collections.defaultdict(list)
    for q, rel in rel_map.items():
        if q not in docs:
            continue
        pos = {k for k, g in rel.items() if g >= args.min_grade}
        if not pos:
            continue
        ex = {eng.key_to_idx[q]} if q in eng.key_to_idx else set()
        qtext = docs[q]["text_dense"] or docs[q].get("summary", "")
        keys = [eng.keys[i] for i in eng.search(qtext, cfg, exclude=ex)]
        out["recall@10"].append(sum(1 for x in keys[:10] if x in pos) / len(pos))
        out["recall@100"].append(sum(1 for x in keys[:100] if x in pos) / len(pos))
        out["ndcg@10"].append(ndcg(keys, rel, 10))
        out["precision@10"].append(sum(1 for x in keys[:10] if x in pos) / 10)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="")
    ap.add_argument("--flow", default="issue", choices=["issue", "text"])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--min-grade", type=int, default=1)
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="data/processed/field_variants.json")
    args = ap.parse_args()

    docs_list = load_jsonl(args.corpus)
    rel_map = load_pooling(args.pooling) if args.pooling else load_gold(args.gold)
    flow = FLOWS[args.flow]
    print(f"corpus {len(docs_list)} | truy van co nhan: {len(rel_map)} | flow={args.flow}",
          flush=True)

    scores = {}
    for name in args.variants:
        scores[name] = measure(docs_list, rel_map, VARIANTS[name], flow, args)
        m = scores[name]
        print(f"  {name:22s} " + "  ".join(f"{k}={v.mean():.4f}" for k, v in m.items()),
              flush=True)

    base = scores[args.baseline]
    out = {"flow": args.flow, "min_grade": args.min_grade,
           "set": args.pooling or args.gold, "baseline": args.baseline,
           "n_queries": int(len(next(iter(base.values())))), "variants": {}}
    print(f"\n=== so voi {args.baseline} (ghep cap, bootstrap {args.n_boot}) ===")
    for name, m in scores.items():
        row = {}
        for metric, v in m.items():
            mean, lo, hi = ci(v, args.n_boot)
            d, dlo, dhi, sig = paired(v, base[metric], args.n_boot)
            row[metric] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                           "delta": round(d, 4), "delta_lo": round(dlo, 4),
                           "delta_hi": round(dhi, 4), "significant": sig}
        out["variants"][name] = row
        r = row["recall@10"]
        n = row["ndcg@10"]
        star = " *" if r["significant"] else ""
        starn = " *" if n["significant"] else ""
        print(f"  {name:22s} R@10 {r['mean']:.4f} ({r['delta']:+.4f} "
              f"[{r['delta_lo']:+.4f},{r['delta_hi']:+.4f}]{star})   "
              f"nDCG {n['mean']:.4f} ({n['delta']:+.4f}{starn})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
