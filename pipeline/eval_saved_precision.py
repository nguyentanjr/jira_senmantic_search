# -*- coding: utf-8 -*-
"""Tinh Precision@K tu pooling runs da luu, khong nap hay chay lai model.

Vi pooling chi cham hop cua cac top-10 cu, bao cao luon ``unjudged_rate@K``.
Mot tai lieu duoc xem la lien quan khi grade >= 1. P@K dung mau so co dinh K.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_judgments(path: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grade = (row.get("grade") or "").strip()
            if grade in {"0", "1", "2"}:
                out[row["query_key"]][row["doc_key"]] = int(grade)
    return dict(out)


def load_runs(path: str) -> dict[str, dict[str, list[str]]]:
    by_config: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_config[row["config"]][row["query_key"]].append(
                (int(row["rank"]), row["doc_key"]))
    return {
        config: {
            query: [doc for _, doc in sorted(rows)]
            for query, rows in queries.items()
        }
        for config, queries in by_config.items()
    }


def confidence_interval(values: list[float], n_boot: int, seed: int) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(np.percentile(samples, 2.5)), 4),
        "hi": round(float(np.percentile(samples, 97.5)), 4),
    }


def paired_interval(a: list[float], b: list[float], n_boot: int, seed: int) -> dict:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not len(diff):
        return {"delta": None, "lo": None, "hi": None, "significant": False}
    rng = np.random.default_rng(seed)
    samples = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta": round(float(diff.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "significant": bool(lo > 0 or hi < 0)}


def evaluate(judgments: dict[str, dict[str, int]],
             runs: dict[str, dict[str, list[str]]], ks: list[int],
             n_boot: int = 5000, seed: int = 17,
             baseline: str = "keyword") -> dict:
    all_queries = sorted(judgments)
    scored = [q for q in all_queries if any(g >= 1 for g in judgments[q].values())]
    result = {
        "relevance_rule": "grade >= 1",
        "precision_denominator": "fixed K",
        "n_queries_all_judged": len(all_queries),
        "n_queries_scored": len(scored),
        "excluded_no_relevant": sorted(set(all_queries) - set(scored)),
        "configs": {}, "baseline": baseline, "paired_vs_baseline": {},
    }
    precision_values: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for config, config_runs in sorted(runs.items()):
        metrics: dict[str, dict] = {}
        for k in ks:
            precision, unjudged = [], []
            for q in scored:
                ranked = config_runs.get(q, [])[:k]
                # Thieu ket qua van tinh nhu slot khong lien quan; day la P@K.
                precision.append(sum(judgments[q].get(d, 0) >= 1 for d in ranked) / k)
                unjudged.append((sum(d not in judgments[q] for d in ranked)
                                  + max(0, k - len(ranked))) / k)
            metrics[f"precision@{k}"] = confidence_interval(
                precision, n_boot, seed + k)
            precision_values[config][k] = precision
            metrics[f"unjudged_rate@{k}"] = {
                "mean": round(float(np.mean(unjudged)), 4) if unjudged else None
            }
        result["configs"][config] = metrics
    if baseline in precision_values:
        for config in sorted(precision_values):
            if config == baseline:
                continue
            result["paired_vs_baseline"][config] = {
                f"precision@{k}": paired_interval(
                    precision_values[config][k], precision_values[baseline][k],
                    n_boot, seed + 1000 + k)
                for k in ks
            }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments",
                    default="data/processed/pooling_judgments_v4_llm.csv")
    ap.add_argument("--runs", default="data/processed/pooling_runs_v4.jsonl")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--baseline", default="keyword")
    ap.add_argument("--out", default="data/processed/precision_pooling_v4.json")
    args = ap.parse_args()

    result = evaluate(load_judgments(args.judgments), load_runs(args.runs),
                      args.ks, args.n_boot, args.seed, args.baseline)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{result['n_queries_scored']} truy van co it nhat mot tai lieu lien quan")
    for config, metrics in result["configs"].items():
        fields = []
        for k in args.ks:
            p = metrics[f"precision@{k}"]
            u = metrics[f"unjudged_rate@{k}"]["mean"]
            fields.append(f"P@{k}={p['mean']:.4f} [{p['lo']:.4f},{p['hi']:.4f}]"
                          f"; unjudged={u:.1%}")
        print(f"{config:<16} " + " | ".join(fields))
    print(f"Chenh lech paired so voi {args.baseline}:")
    for config, metrics in result["paired_vs_baseline"].items():
        fields = [f"P@{k}={metrics[f'precision@{k}']['delta']:+.4f} "
                  f"[{metrics[f'precision@{k}']['lo']:+.4f},"
                  f"{metrics[f'precision@{k}']['hi']:+.4f}]"
                  f"{'*' if metrics[f'precision@{k}']['significant'] else ''}"
                  for k in args.ks]
        print(f"{config:<16} " + " | ".join(fields))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
