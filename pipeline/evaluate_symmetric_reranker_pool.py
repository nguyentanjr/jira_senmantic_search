"""Evaluate two rankings after the union of both top-10 lists is fully judged."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def first_existing(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names!r} exists under {root}")


def metrics(ranking: list[str], labels: dict[str, int]) -> dict:
    top = ranking[:10]
    if not all(k in labels for k in top):
        raise AssertionError("Top-10 is not fully judged")
    positives = {k for k, g in labels.items() if g > 0}
    gains = [2 ** labels[k] - 1 for k in top]
    ideal = sorted((2 ** g - 1 for g in labels.values()), reverse=True)[:10]
    dcg = lambda xs: sum(g / math.log2(i + 2) for i, g in enumerate(xs))
    return {
        "precision10": sum(labels[k] > 0 for k in top) / 10,
        "pool_recall10": sum(k in positives for k in top) / len(positives),
        "grade0_count10": sum(labels[k] == 0 for k in top),
        "ndcg10": dcg(gains) / dcg(ideal) if dcg(ideal) else None,
    }


def paired_bootstrap(per_query: list[dict], modes: tuple[str, ...], names: tuple[str, ...],
                     samples: int = 20_000, seed: int = 20260923) -> dict:
    """Cluster bootstrap over query families for paired metric deltas."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_query), size=(samples, len(per_query)))
    result = {}
    for mode in modes:
        result[mode] = {}
        for name in names:
            delta = np.asarray([
                row[mode][name] - row["baseline"][name] for row in per_query
            ], dtype=np.float64)
            draws = delta[indices].mean(axis=1)
            result[mode][name] = {
                "low95": float(np.quantile(draws, 0.025)),
                "high95": float(np.quantile(draws, 0.975)),
            }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bundle, result = Path(args.bundle), Path(args.result)
    pool_path = first_existing(bundle, "dev_candidate_pools.jsonl", "candidate_pools.jsonl")
    prediction_path = first_existing(result, "dev_predictions.jsonl", "predictions.jsonl")
    pools = {r["query_key"]: r for r in load(pool_path)}
    predictions = {r["query_key"]: r for r in load(prediction_path)}
    for pool in pools.values():
        pool["judgments"] = {k: int(v) for k, v in pool["judgments"].items()}
    for path in map(Path, args.judgments):
        for row in load(path):
            pools[row["query_key"]]["judgments"][row["candidate_key"]] = int(row["grade"])
    per_query = []
    for query_key, pool in sorted(pools.items()):
        labels = pool["judgments"]
        baseline = pool["baseline_ranking"]
        reranked = predictions[query_key]["reranked_keys"]
        ce_rank = {key: i for i, key in enumerate(reranked)}
        safe_head = baseline[:1] + sorted(baseline[1:10], key=lambda key: ce_rank[key])
        safe_ranking = safe_head + baseline[10:]
        per_query.append({
            "query_key": query_key,
            "label_counts": {str(g): sum(v == g for v in labels.values()) for g in (0, 1, 2)},
            "baseline": metrics(baseline, labels),
            "reranked": metrics(reranked, labels),
            "safe_top10_pin1": metrics(safe_ranking, labels),
        })
    names = ("precision10", "pool_recall10", "grade0_count10", "ndcg10")
    macro = {mode: {name: float(np.mean([r[mode][name] for r in per_query])) for name in names}
             for mode in ("baseline", "reranked", "safe_top10_pin1")}
    delta = {
        mode: {name: macro[mode][name] - macro["baseline"][name] for name in names}
        for mode in ("reranked", "safe_top10_pin1")
    }
    family_regressions = {
        mode: {
            name: [r["query_key"] for r in per_query
                   if ((r[mode][name] > r["baseline"][name] + 1e-12)
                       if name == "grade0_count10"
                       else (r[mode][name] < r["baseline"][name] - 1e-12))]
            for name in names
        }
        for mode in ("reranked", "safe_top10_pin1")
    }
    report = {
        "queries": len(per_query), "macro": macro, "delta_vs_baseline": delta,
        "paired_query_bootstrap_delta95": paired_bootstrap(
            per_query, ("reranked", "safe_top10_pin1"), names),
        "family_regressions": family_regressions,
        "per_query": per_query,
        "top10_union_fully_judged": True,
        "adaptive_to_compared_models": True,
        "valid_independent_evaluation": False,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
