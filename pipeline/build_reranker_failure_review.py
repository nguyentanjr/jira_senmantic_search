"""Create a blinded review pool from candidates promoted by a failed reranker."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_relation_review import context
from retrieval import load_jsonl


def review_id(query_key: str, candidate_key: str) -> str:
    return hashlib.sha256((query_key + "\0" + candidate_key).encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/processed/relation_reranker_colab_20260923")
    ap.add_argument("--result", default="data/processed/relation_reranker_colab_result_20260923/output")
    ap.add_argument("--out", default="data/processed/relation_reranker_failure_review_20260923")
    ap.add_argument("--per-query", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    bundle, result = Path(args.bundle), Path(args.result)
    pools = {r["query_key"]: r for r in load_jsonl(str(bundle / "dev_candidate_pools.jsonl"))}
    predictions = {r["query_key"]: r for r in load_jsonl(str(result / "dev_predictions.jsonl"))}
    docs = {r["key"]: r for r in load_jsonl("data/processed/corpus.jsonl")}
    reviews, provenance = [], []
    used = set()
    for query_key, pool in sorted(pools.items()):
        baseline_rank = {k: i + 1 for i, k in enumerate(pool["baseline_ranking"])}
        ce_rank = {k: i + 1 for i, k in enumerate(predictions[query_key]["reranked_keys"])}
        judged = set(pool["judgments"])
        eligible = [k for k in predictions[query_key]["reranked_keys"][:20]
                    if k not in judged and k not in used]
        picks = eligible[:args.per_query]
        if len(picks) != args.per_query:
            raise AssertionError(f"Insufficient promoted unjudged candidates for {query_key}")
        for candidate_key in picks:
            used.add(candidate_key)
            rid = review_id(query_key, candidate_key)
            reviews.append({
                "review_id": rid,
                "query_key": query_key,
                "candidate_key": candidate_key,
                "query_context": context(docs[query_key]),
                "candidate_context": context(docs[candidate_key]),
                "same_context": "",
                "grade": "",
                "reason": "",
                "reviewer": "",
                "notes": "",
            })
            provenance.append({
                "review_id": rid,
                "query_key": query_key,
                "candidate_key": candidate_key,
                "baseline_rank": baseline_rank[candidate_key],
                "trained_ce_rank": ce_rank[candidate_key],
                "rank_promotion": baseline_rank[candidate_key] - ce_rank[candidate_key],
                "selection": "first four unjudged candidates in trained CE top-20",
                "use": "failure_analysis_and_future_development_only",
            })
    out.mkdir(parents=True)
    with (out / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reviews[0]))
        writer.writeheader(); writer.writerows(reviews)
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in provenance), encoding="utf-8")
    by_query = {}
    for row in reviews:
        by_query.setdefault(row["query_key"], []).append(row)
    for i, (query_key, rows) in enumerate(sorted(by_query.items()), 1):
        parts = [f"QUERY {query_key}", context(docs[query_key]), "\nCANDIDATES"]
        for row in rows:
            parts.extend([f"\n--- {row['candidate_key']} ---", context(docs[row["candidate_key"]])])
        (out / f"dossier_{i}.txt").write_text("\n".join(parts), encoding="utf-8")
    manifest = {
        "queries": len(by_query),
        "pairs": len(reviews),
        "labels_blank": True,
        "provenance_hidden_from_dossiers": True,
        "source_result_accepted": False,
        "scope": "adaptive development review; invalid for independent evaluation of the source model",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
