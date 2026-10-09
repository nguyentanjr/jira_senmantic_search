"""Score hybrid vs base CE on the SMS top-10 review (metric fixed in its README).

    python pipeline/eval_sms_top10_review.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

D = Path("data/processed/sms_top10_review_20260923")
POOLS = Path("colab/sms_type_probe_bundle/data/candidate_pools.jsonl")
PRED = Path("data/processed/sms_type_probe_result/output/predictions.jsonl")
LABELS_MD = Path("sms_type_probe_confirm_ai_pre_review.md")
PROBE = Path("data/processed/sms_type_probe_20260923/candidates.csv")


def load_labels():
    labels = {}
    for line in LABELS_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(RO-\d+)\s*\|\s*\w*\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|", line)
        if m:
            labels[m.group(1)] = (m.group(2), m.group(3))
    return labels


def metrics(top, lab):
    g = [lab.get(k, ("loai", ""))[0] for k in top[:10]]
    first = next((i for i, x in enumerate(g) if x == "thua"), 10)
    pairs = [(i, j) for i, a in enumerate(g) if a == "thua" for j, b in enumerate(g) if b == "sai"]
    return {"sai_above_first_thua": sum(x == "sai" for x in g[:first]),
            "pair_acc": (sum(i < j for i, j in pairs) / len(pairs)) if pairs else None,
            "thua": g.count("thua"), "sai": g.count("sai"), "khong_gui": g.count("khong_gui"),
            "first_thua_rank": first + 1 if first < 10 else None}


def main() -> None:
    import csv
    lab = load_labels()
    anchors = sorted(r["key"] for r in csv.DictReader(open(PROBE, encoding="utf-8-sig"))
                     if r["confirm_group"] == "thua")
    pools = {json.loads(l)["query_key"]: json.loads(l)["baseline_ranking"] for l in open(POOLS, encoding="utf-8")}
    preds = {json.loads(l)["query_key"]: json.loads(l)["reranked_keys"] for l in open(PRED, encoding="utf-8")}
    per = [{"anchor": q, "hybrid": metrics(pools[q], lab), "ce": metrics(preds[q], lab)} for q in anchors]
    rng = np.random.default_rng(20260923)
    summ = {}
    for key in ("sai_above_first_thua", "pair_acc", "thua", "sai", "khong_gui"):
        rows = [r for r in per if r["hybrid"][key] is not None and r["ce"][key] is not None]
        h = np.array([r["hybrid"][key] for r in rows], float)
        c = np.array([r["ce"][key] for r in rows], float)
        d = c - h
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        summ[key] = {"n": len(rows), "hybrid": h.mean(), "ce": c.mean(), "delta": d.mean(),
                     "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    report = {"label_source": "13 user_confirmed + 159 AI_PRE_REVIEW (not human verified)",
              "summary": summ, "per_anchor": per}
    (D / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                              encoding="utf-8")
    for k, v in summ.items():
        print(f"{k:22s} n={v['n']:2d} hybrid={v['hybrid']:.3f} ce={v['ce']:.3f} "
              f"delta={v['delta']:+.3f} ci=[{v['ci95'][0]:+.3f};{v['ci95'][1]:+.3f}]")
    for r in per:
        print(r["anchor"], "H", {k: r["hybrid"][k] for k in ("thua", "sai", "sai_above_first_thua")},
              "C", {k: r["ce"][k] for k in ("thua", "sai", "sai_above_first_thua")})


if __name__ == "__main__":
    main()
