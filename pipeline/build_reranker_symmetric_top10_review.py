"""Build the missing labels for the union of baseline and reranker top-10."""
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


def rid(query_key: str, candidate_key: str) -> str:
    return hashlib.sha256((query_key + "\0" + candidate_key).encode()).hexdigest()[:16]


def first_existing(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names!r} exists under {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/processed/relation_reranker_colab_20260923")
    ap.add_argument("--result", default="data/processed/relation_reranker_colab_result_20260923/output")
    ap.add_argument("--known-judgments", nargs="*", default=[
        "data/processed/relation_reranker_failure_review_20260923/judgments.jsonl"])
    ap.add_argument("--out", default="data/processed/relation_reranker_symmetric_top10_review_20260923")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    bundle = Path(args.bundle)
    result = Path(args.result)
    pool_path = first_existing(bundle, "dev_candidate_pools.jsonl", "candidate_pools.jsonl")
    prediction_path = first_existing(result, "dev_predictions.jsonl", "predictions.jsonl")
    pools = {r["query_key"]: r for r in load_jsonl(str(pool_path))}
    predictions = {r["query_key"]: r for r in load_jsonl(str(prediction_path))}
    extra = [row for path in args.known_judgments for row in load_jsonl(path)]
    known = {(q, k) for q, p in pools.items() for k in p["judgments"]}
    known |= {(r["query_key"], r["candidate_key"]) for r in extra}
    docs = {r["key"]: r for r in load_jsonl("data/processed/corpus.jsonl")}
    rows, provenance = [], []
    for query_key, pool in sorted(pools.items()):
        base = pool["baseline_ranking"]
        ce = predictions[query_key]["reranked_keys"]
        br = {k: i + 1 for i, k in enumerate(base)}
        cr = {k: i + 1 for i, k in enumerate(ce)}
        union = set(base[:10]) | set(ce[:10])
        missing = sorted((k for k in union if (query_key, k) not in known),
                         key=lambda k: (min(br[k], cr[k]), k))
        for candidate_key in missing:
            review_id = rid(query_key, candidate_key)
            rows.append({
                "review_id": review_id,
                "query_key": query_key,
                "candidate_key": candidate_key,
                "query_context": context(docs[query_key]),
                "candidate_context": context(docs[candidate_key]),
                "same_context": "", "grade": "", "reason": "", "reviewer": "", "notes": "",
            })
            provenance.append({
                "review_id": review_id,
                "baseline_rank": br[candidate_key],
                "ce_rank": cr[candidate_key],
                "in_baseline_top10": br[candidate_key] <= 10,
                "in_ce_top10": cr[candidate_key] <= 10,
                "selection": "previously unjudged union of baseline top-10 and trained CE top-10",
            })
    out.mkdir(parents=True)
    with (out / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in provenance), encoding="utf-8")
    for i, query_key in enumerate(sorted({r["query_key"] for r in rows}), 1):
        query_rows = [r for r in rows if r["query_key"] == query_key]
        parts = [f"QUERY {query_key}", context(docs[query_key]), "\nCANDIDATES"]
        for row in query_rows:
            parts.extend([f"\n--- {row['candidate_key']} ---", context(docs[row["candidate_key"]])])
        (out / f"dossier_{i}.txt").write_text("\n".join(parts), encoding="utf-8")
    manifest = {
        "queries": len({r["query_key"] for r in rows}),
        "pairs": len(rows),
        "selection": "complete missing judgments for union of both top-10 rankings",
        "provenance_hidden_from_dossiers": True,
        "scope": "adaptive symmetric pooled evaluation; not independent test",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
