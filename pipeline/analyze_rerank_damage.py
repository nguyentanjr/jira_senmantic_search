# -*- coding: utf-8 -*-
"""
Phan bo TAC HAI cua cross-encoder theo tung truy van: co dau vs khong dau.

    python pipeline/analyze_rerank_damage.py \
        --dump data/processed/bm25_fold_perquery.jsonl --fold add

Dau vao: JSONL tu `eval_bm25_fold.py --stage 2 --dump-per-query` (dev 500).

VI SAO KHONG DUNG DELTA R@10
Moi truy van paraphrase co MOT dap an -> R@10 tung truy van la 0/1, delta chi
co -1/0/+1. Do bang HANG cua dap an truoc (hybrid) va sau rerank.

PHAM VI RERANK
Rerank chi xep lai top-k (30) cua hybrid. Dap an ngoai top-k khong the bi rerank
anh huong -> ty le giup/hai tinh tren nhom "trong tam" (hang hybrid <= k).

KET QUA TUNG TRUY VAN (`outcome`)
    pushed_out  hybrid <= 10, rerank > 10     (hai nang: mat khoi top-10)
    pulled_in   hybrid > 10,  rerank <= 10
    down / up   doi hang nhung khong vuot moc 10
    same        giu nguyen hang
Muc do: d_rr = 1/hang_rerank - 1/hang_hybrid (lien tuc, am = hai).

DOI CHUNG
Cung mot truy van co HAI ban. Nhom quan tam la "hai rieng do dau": rerank
khong hai ban co dau (d_rr >= 0) nhung hai ban khong dau (d_rr < 0).

DAC TRUNG (chi mo ta, CHUA la quy tac): ce_margin, overlap10, n_tok, vn_ratio,
lexical_group, jac_bin. Moi quy tac chon loc rut ra tu day phai chot tren dev
va kiem MOT lan tren test.

Dau ra: data/processed/rerank_damage_<fold>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import load_jsonl  # noqa: E402

QVERS = ("goc", "khong_dau")
FEATURES = ("ce_margin", "ce_top1", "overlap10", "n_tok", "vn_ratio")


def outcome(h: int | None, r: int | None) -> str:
    h = h or 10 ** 9
    r = r or 10 ** 9
    if h == r:
        return "same"
    if h <= 10 < r:
        return "pushed_out"
    if r <= 10 < h:
        return "pulled_in"
    return "up" if r < h else "down"


def d_rr(h: int | None, r: int | None) -> float:
    inv = lambda x: 1.0 / x if x else 0.0
    return inv(r) - inv(h)


def summarize(rows: list[dict], k: int) -> dict:
    inr = [x for x in rows if x["rank_hybrid"] and x["rank_hybrid"] <= k]
    oc = Counter(x["outcome"] for x in inr)
    d = np.asarray([x["d_rr"] for x in inr]) if inr else np.zeros(0)
    n = len(inr)
    pct = lambda c: round(100.0 * c / n, 1) if n else 0.0
    return {
        "n_total": len(rows), "n_in_scope": n,
        "outcome_pct": {o: pct(oc.get(o, 0)) for o in
                        ("pulled_in", "up", "same", "down", "pushed_out")},
        "helped_pct": pct(oc["pulled_in"] + oc["up"]),
        "hurt_pct": pct(oc["down"] + oc["pushed_out"]),
        "severe_pct": pct(sum(1 for x in inr if x["d_rr"] <= -0.2)),
        "d_rr_quantiles": ({q: round(float(np.percentile(d, q)), 4)
                            for q in (5, 10, 25, 50, 75, 90, 95)} if n else {}),
        "d_rr_mean": round(float(d.mean()), 4) if n else 0.0,
        "hurt_mass": round(float(d[d < 0].sum()), 3) if n else 0.0,
        "help_mass": round(float(d[d > 0].sum()), 3) if n else 0.0,
    }


def feature_contrast(rows: list[dict]) -> dict:
    """Trung binh dac trung: bi hai vs khong bi hai (trong tam)."""
    hurt = [x for x in rows if x["d_rr"] < 0]
    ok = [x for x in rows if x["d_rr"] >= 0]
    out = {}
    for f in FEATURES:
        a = [x[f] for x in hurt if x.get(f) is not None]
        b = [x[f] for x in ok if x.get(f) is not None]
        out[f] = {"hurt_mean": round(float(np.mean(a)), 4) if a else None,
                  "ok_mean": round(float(np.mean(b)), 4) if b else None,
                  "auc_hurt_lower": _auc(a, b)}
    for f in ("lexical_group", "jac_bin"):
        c_h, c_o = Counter(str(x.get(f)) for x in hurt), Counter(str(x.get(f)) for x in ok)
        out[f] = {g: {"hurt": c_h.get(g, 0), "ok": c_o.get(g, 0)}
                  for g in sorted(set(c_h) | set(c_o))}
    return out


def _auc(hurt, ok) -> float | None:
    """P(dac trung cua truy van bi hai < cua truy van khong bi hai); 0,5 = vo ich."""
    if not hurt or not ok:
        return None
    h, o = np.asarray(hurt, float), np.asarray(ok, float)
    less = (h[:, None] < o[None, :]).mean()
    ties = (h[:, None] == o[None, :]).mean()
    return round(float(less + 0.5 * ties), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/processed/bm25_fold_perquery.jsonl")
    ap.add_argument("--fold", default="add")
    ap.add_argument("--corpus", default="data/processed/corpus_holdout.jsonl")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.dump) if r["fold"] == args.fold]
    if not rows:
        sys.exit(f"khong co dong fold={args.fold} trong {args.dump}")
    k = rows[0]["rerank_k"]
    for r in rows:
        r["outcome"] = outcome(r["rank_hybrid"], r["rank_rerank"])
        r["d_rr"] = d_rr(r["rank_hybrid"], r["rank_rerank"])
        r["ce_top1"] = r["ce_top10"][0] if r["ce_top10"] else None
    by = defaultdict(dict)
    for r in rows:
        by[r["key"]][r["qv"]] = r
    pairs = {key: d for key, d in by.items() if all(v in d for v in QVERS)}

    result = {"dump": args.dump, "fold": args.fold, "rerank_k": k,
              "n_pairs": len(pairs), "by_qv": {}, "features": {}}
    for qv in QVERS:
        rs = [d[qv] for d in pairs.values()]
        result["by_qv"][qv] = summarize(rs, k)
        result["features"][qv] = feature_contrast(
            [x for x in rs if x["rank_hybrid"] and x["rank_hybrid"] <= k])

    # bang cheo: ket qua ban co dau x ban khong dau (chi truy van trong tam o CA HAI ban)
    both = [d for d in pairs.values()
            if all(d[v]["rank_hybrid"] and d[v]["rank_hybrid"] <= k for v in QVERS)]
    sign = lambda x: "giup" if x > 0 else ("hai" if x < 0 else "khong_doi")
    cross = Counter((sign(d["goc"]["d_rr"]), sign(d["khong_dau"]["d_rr"])) for d in both)
    result["crosstab_goc_x_khong_dau"] = {f"{a} | {b}": c for (a, b), c in sorted(cross.items())}
    only_undiac = [d for d in both if d["goc"]["d_rr"] >= 0 > d["khong_dau"]["d_rr"]]
    result["hurt_only_undiacritized"] = {
        "n": len(only_undiac), "of_both_in_scope": len(both),
        "pushed_out": sum(1 for d in only_undiac if d["khong_dau"]["outcome"] == "pushed_out"),
    }

    docs = {d["key"]: d for d in load_jsonl(args.corpus)}
    summ = lambda key: (docs.get(key, {}).get("summary", "") or "")[:90]
    worst = sorted(only_undiac, key=lambda d: d["khong_dau"]["d_rr"])[: args.top]
    result["worst_hurt_only_undiacritized"] = [{
        "key": d["goc"]["key"], "answer": summ(d["goc"]["key"]),
        "query_goc": d["goc"]["query"][:200], "query_khong_dau": d["khong_dau"]["query"][:200],
        "rank_goc": [d["goc"]["rank_hybrid"], d["goc"]["rank_rerank"]],
        "rank_khong_dau": [d["khong_dau"]["rank_hybrid"], d["khong_dau"]["rank_rerank"]],
        "d_rr_khong_dau": round(d["khong_dau"]["d_rr"], 3),
        "ce_margin_khong_dau": d["khong_dau"]["ce_margin"],
        "overlap10_khong_dau": d["khong_dau"]["overlap10"],
        "rerank_top3_khong_dau": [f"{x}: {summ(x)}" for x in d["khong_dau"]["rerank_top10"][:3]],
        "hybrid_top3_khong_dau": [f"{x}: {summ(x)}" for x in d["khong_dau"]["hybrid_top10"][:3]],
    } for d in worst]

    # ------------------------------------------------------------ in
    print(f"fold={args.fold} | {len(pairs)} cap truy van | rerank top-{k}")
    for qv in QVERS:
        s = result["by_qv"][qv]
        print(f"\n== {qv}: trong tam {s['n_in_scope']}/{s['n_total']} ==")
        print(f"  giup {s['helped_pct']}% | hai {s['hurt_pct']}% | hai nang (d_rr<=-0,2) {s['severe_pct']}%")
        print(f"  {s['outcome_pct']}")
        print(f"  d_rr mean {s['d_rr_mean']} | tong giup {s['help_mass']} | tong hai {s['hurt_mass']}")
        print(f"  d_rr phan vi {s['d_rr_quantiles']}")
        for f, v in result["features"][qv].items():
            print(f"  {f:<14} {v}")
    print(f"\n== bang cheo (co dau | khong dau), {len(both)} truy van trong tam o ca hai ban ==")
    for c, n in result["crosstab_goc_x_khong_dau"].items():
        print(f"  {c:<24} {n}")
    h = result["hurt_only_undiacritized"]
    print(f"  hai RIENG do dau: {h['n']} (trong do day khoi top-10: {h['pushed_out']})")
    print(f"\n== {len(worst)} truy van hai rieng do dau nang nhat ==")
    for w in result["worst_hurt_only_undiacritized"]:
        print(f"\n  {w['key']} | dap an: {w['answer']}")
        print(f"    hang co dau {w['rank_goc']} | khong dau {w['rank_khong_dau']} "
              f"| d_rr {w['d_rr_khong_dau']} | ce_margin {w['ce_margin_khong_dau']} "
              f"| overlap10 {w['overlap10_khong_dau']}")
        print(f"    q: {w['query_khong_dau'][:140]!r}")
        print(f"    hybrid top3: {w['hybrid_top3_khong_dau']}")
        print(f"    rerank top3: {w['rerank_top3_khong_dau']}")

    out = args.out or f"data/processed/rerank_damage_{args.fold}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
