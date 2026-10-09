"""Score the SMS defect-type ordering probe (metric fixed in the probe README).

    python pipeline/eval_sms_type_probe.py --pred <result dir>/predictions.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONFUSER = {"thua": "sai", "sai": "thua"}
OUT_OF_POOL = 101


def pair_accuracy(ranking, pos, neg):
    rank = {k: i + 1 for i, k in enumerate(ranking[:100])}
    ok = n = 0
    for p in pos:
        for q in neg:
            rp, rq = rank.get(p, OUT_OF_POOL), rank.get(q, OUT_OF_POOL)
            if rp == rq == OUT_OF_POOL:
                continue
            n += 1
            ok += rp < rq
    return (ok / n if n else None), n


def confusers_above_first(ranking, pos, neg):
    top = ranking[:10]
    first = next((i for i, k in enumerate(top) if k in pos), len(top))
    return sum(1 for k in top[:first] if k in neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/sms_type_probe_20260923/candidates.csv")
    ap.add_argument("--pools", default="colab/sms_type_probe_bundle/data/candidate_pools.jsonl")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="data/processed/sms_type_probe_20260923/evaluation_report.json")
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    group = {r["key"]: r["confirm_group"] for r in csv.DictReader(open(args.labels, encoding="utf-8-sig"))}
    pools = {json.loads(l)["query_key"]: json.loads(l) for l in open(args.pools, encoding="utf-8")}
    preds = {json.loads(l)["query_key"]: json.loads(l) for l in open(args.pred, encoding="utf-8")}
    per = []
    for q, pool in sorted(pools.items()):
        g = group[q]
        members = lambda gg: {k for k, v in group.items() if v == gg and k != q}
        pos, neg, miss = members(g), members(CONFUSER[g]), members("khong_gui")
        row = {"query_key": q, "group": g}
        for name, ranking in (("hybrid", pool["baseline_ranking"]), ("ce", preds[q]["reranked_keys"])):
            acc, n = pair_accuracy(ranking, pos, neg)
            acc_miss, n_miss = pair_accuracy(ranking, pos, miss)
            row[name] = {"pair_acc": acc, "pairs": n, "pair_acc_vs_khong_gui": acc_miss,
                         "confusers_above_first_same_top10": confusers_above_first(ranking, pos, neg),
                         "same_group_in_top10": sum(k in pos for k in ranking[:10])}
        per.append(row)

    rng = np.random.default_rng(args.seed)
    def summary(rows):
        both = [r for r in rows if r["hybrid"]["pair_acc"] is not None and r["ce"]["pair_acc"] is not None]
        h = np.array([r["hybrid"]["pair_acc"] for r in both])
        c = np.array([r["ce"]["pair_acc"] for r in both])
        d = c - h
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)] if len(d) else [0]
        mean = lambda name, key: float(np.mean([r[name][key] for r in rows]))
        return {"anchors_with_pairs": len(both), "hybrid_pair_acc": float(h.mean()) if len(h) else None,
                "ce_pair_acc": float(c.mean()) if len(c) else None, "delta": float(d.mean()) if len(d) else None,
                "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "hybrid_confusers_above_first": mean("hybrid", "confusers_above_first_same_top10"),
                "ce_confusers_above_first": mean("ce", "confusers_above_first_same_top10"),
                "hybrid_same_group_top10": mean("hybrid", "same_group_in_top10"),
                "ce_same_group_top10": mean("ce", "same_group_in_top10")}
    report = {"all": summary(per),
              "thua": summary([r for r in per if r["group"] == "thua"]),
              "sai": summary([r for r in per if r["group"] == "sai"])}
    a, t, s = report["all"], report["thua"], report["sai"]
    report["verdict_ce_helps"] = bool(t["delta"] and t["delta"] > 0 and s["delta"] and s["delta"] > 0
                                      and a["delta_ci95"][0] > 0)
    report["per_query"] = per
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
