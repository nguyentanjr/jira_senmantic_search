"""Build hybrid top-100 pools for a fixed fresh-anchor manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_relation_reranker_bundle import candidate_pools, dump_jsonl
from retrieval import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    source = json.loads(Path(args.anchor_manifest).read_text(encoding="utf-8"))
    anchors = [row["key"] for row in source["families"]]
    docs = load_jsonl("data/processed/corpus.jsonl")
    # candidate_pools needs rows only to identify queries and initialize labels.
    marker_rows = [{"query_key": key, "candidate_key": key, "grade": 0} for key in anchors]
    pools, cache_info = candidate_pools(marker_rows, docs)
    for pool in pools:
        pool["judgments"] = {}
    out.mkdir(parents=True)
    dump_jsonl(out / "candidate_pools.jsonl", pools)
    manifest = {
        "anchors": len(anchors),
        "anchor_keys": sorted(anchors),
        "selection_source": args.anchor_manifest,
        "labels": 0,
        "model_outputs_used_for_anchor_selection": False,
        "retrieval": "production issue-flow hybrid top-100",
        "cache": cache_info,
        "scope": "fresh pooled development test fixed before cross-encoder scoring",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
