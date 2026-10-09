# -*- coding: utf-8 -*-
"""
Chon cach TRON rerank tren DEV (offline), roi kiem MOT lan tren TEST.

    python pipeline/select_rerank_fusion.py --split dev    # ghi rerank_fusion_choice.json
    python pipeline/select_rerank_fusion.py --split test   # can file lua chon; chay mot lan

Dau vao: data/processed/rerank_scores_<split>.jsonl (dump_rerank_scores.py).

LUOI -- CHOT TRUOC KHI DO (2026-09-14)
    replace                        (hanh vi hien tai)
    rrf   weight in {0.3, 0.5, 0.7}, k = 60
    blend weight in {0.3, 0.5, 0.7}
weight = phan cua cross-encoder. Them `hybrid` (khong rerank) de doi chieu,
khong duoc chon.

QUY TAC CHON -- CHOT TRUOC KHI DO. So cap doi voi `replace` tren DEV:
    (a) khong dau: MRR@10 tang, CI 95% khong chua 0
    (b) khong dau: R@10 khong giam co y nghia (hi >= 0)
    (c) co dau   : R@10 va MRR@10 khong giam co y nghia (hi >= 0)
Nhieu ung vien dat -> lay (a) lon nhat; khong ai dat -> giu replace.
Ly do chon MRR: phan tich tung truy van cho thay tac hai chinh cua replace la
day dap an hybrid hang 1 xuong -- R@10 khong thay dieu do.

TEST: in replace, lua chon va hybrid cho ca hai ban truy van, cap doi voi
replace. File ket qua test khong ghi de tru khi --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paraphrase import ci, paired         # noqa: E402
from eval_rerankers import metrics_from_ranked  # noqa: E402
from retrieval import fuse_rerank, load_jsonl   # noqa: E402

QVERS = ("goc", "khong_dau")
METRICS = ("recall@10", "mrr@10", "ndcg@10")
GRID = ([("replace", None)] + [("rrf", w) for w in (0.3, 0.5, 0.7)]
        + [("blend", w) for w in (0.3, 0.5, 0.7)])


def name(method: str, w) -> str:
    return method if w is None else f"{method}@{w}"


def ranking(row: dict, method: str, w) -> list[str]:
    hy = row["hybrid"]
    if method == "hybrid":
        return hy
    k = len(row["ce"])
    head = np.asarray(hy[:k], dtype=object)
    fused = fuse_rerank(head, np.asarray(row["ce"], dtype=np.float64), method,
                        0.5 if w is None else w)
    return list(fused) + hy[k:]


def score_rows(rows: list[dict], cands) -> dict:
    out = {qv: {name(m, w): {mm: [] for mm in METRICS} for m, w in cands} for qv in QVERS}
    for r in rows:
        rel = set(r["relevant"])
        for m, w in cands:
            for mm, v in metrics_from_ranked(ranking(r, m, w), rel).items():
                out[r["qv"]][name(m, w)][mm].append(v)
    return {qv: {c: {mm: np.asarray(v) for mm, v in d.items()} for c, d in dd.items()}
            for qv, dd in out.items()}


def choose(paired_block: dict, names) -> dict:
    checks = {}
    for c in names:
        nd, go = paired_block["khong_dau"][c], paired_block["goc"][c]
        a = nd["mrr@10"]["delta"] > 0 and nd["mrr@10"]["significant"]
        b = nd["recall@10"]["hi"] >= 0
        cc = go["recall@10"]["hi"] >= 0 and go["mrr@10"]["hi"] >= 0
        checks[c] = {"a": a, "b": b, "c": cc, "ok": a and b and cc,
                     "gain_mrr_khong_dau": nd["mrr@10"]["delta"]}
    ok = [c for c in names if checks[c]["ok"]]
    return {"checks": checks,
            "chosen": max(ok, key=lambda c: checks[c]["gain_mrr_khong_dau"]) if ok else "replace"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--scores", default="")
    ap.add_argument("--choice", default="data/processed/rerank_fusion_choice.json")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    scores_path = args.scores or f"data/processed/rerank_scores_{args.split}.jsonl"
    out = args.out or f"data/processed/rerank_fusion_{args.split}.json"

    if args.split == "dev":
        cands = GRID
    else:
        if not os.path.exists(args.choice):
            sys.exit(f"chua co {args.choice}: chay --split dev truoc")
        if os.path.exists(out) and not args.force:
            sys.exit(f"{out} da ton tai: test chi chay mot lan. Dung --force neu that su can.")
        with open(args.choice, encoding="utf-8") as f:
            chosen = json.load(f)["chosen"]
        m, _, w = chosen.partition("@")
        cands = [("replace", None)] + ([(m, float(w))] if chosen != "replace" else [])
    cands = cands + [("hybrid", None)]

    rows = load_jsonl(scores_path)
    arr = score_rows(rows, cands)
    names = [name(m, w) for m, w in cands]
    low = {qv: np.asarray([i for i, r in enumerate([x for x in rows if x["qv"] == qv])
                           if r.get("jac_bin") == "<0.2"]) for qv in QVERS}

    def block(sub: bool):
        res = {"ci": {}, "vs_replace": {}}
        for qv in QVERS:
            sel = (lambda a, qv=qv: a[low[qv]]) if sub else (lambda a: a)
            res["ci"][qv] = {c: {mm: ci(sel(arr[qv][c][mm]), args.n_boot) for mm in METRICS}
                             for c in names}
            res["vs_replace"][qv] = {c: {mm: paired(sel(arr[qv][c][mm]),
                                                    sel(arr[qv]["replace"][mm]), args.n_boot)
                                         for mm in METRICS}
                                     for c in names if c != "replace"}
        return res

    n_q = len(rows) // 2
    result = {"split": args.split, "n_queries": n_q, "scores": scores_path,
              "candidates": names, "overall": block(False), "jac_lt_0.2": block(True)}
    for key in ("overall", "jac_lt_0.2"):
        b = result[key]
        print(f"\n== {key} ==  cot: R@10  MRR@10  nDCG@10 | chenh voi replace")
        for qv in QVERS:
            for c in names:
                x = b["ci"][qv][c]
                line = f"  {qv:<10}{c:<11}" + "  ".join(f"{x[mm]['mean']:.3f}" for mm in METRICS)
                if c != "replace":
                    line += " | " + " ".join(
                        f"{d['delta']:+.4f}[{d['lo']:+.3f},{d['hi']:+.3f}]{'*' if d['significant'] else ' '}"
                        for d in b["vs_replace"][qv][c].values())
                print(line)

    if args.split == "dev":
        fusions = [c for c in names if c not in ("replace", "hybrid")]
        result["choice"] = choose(result["overall"]["vs_replace"], fusions)
        with open(args.choice, "w", encoding="utf-8") as f:
            json.dump({"chosen": result["choice"]["chosen"], "rule": "xem docstring",
                       "grid": fusions, "checks": result["choice"]["checks"]},
                      f, ensure_ascii=False, indent=1)
        print(f"\nCHON THEO QUY TAC: {result['choice']['chosen']}")
        for c, v in result["choice"]["checks"].items():
            print(f"  {c:<11} {v}")
        print(f"-> {args.choice}")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
