"""Build the Colab inference bundle for the SMS defect-type ordering probe.

Anchors = issues the user confirmed as `thua` or `sai` in candidates.csv.
Pools are the production issue-flow hybrid top-100; the bundle carries no labels,
so the Colab side cannot see which candidates are confirmed.

    python pipeline/build_sms_type_probe_bundle.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_relation_reranker_bundle import candidate_pools, dump_jsonl
from retrieval import load_jsonl

SCORER = Path("colab/relation_reranker_independent_eval_bundle/evaluate_relation_reranker.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/sms_type_probe_20260923/candidates.csv")
    ap.add_argument("--out", default="colab/sms_type_probe_bundle")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    rows = list(csv.DictReader(open(args.labels, encoding="utf-8-sig")))
    anchors = sorted(r["key"] for r in rows if r["confirm_group"] in ("thua", "sai"))
    docs = load_jsonl("data/processed/corpus.jsonl")
    marker_rows = [{"query_key": k, "candidate_key": k, "grade": 0} for k in anchors]
    pools, cache_info = candidate_pools(marker_rows, docs)
    for pool in pools:
        pool["judgments"] = {}
    (out / "data").mkdir(parents=True)
    dump_jsonl(out / "data" / "candidate_pools.jsonl", pools)
    manifest = {
        "anchors": len(anchors),
        "anchor_keys": anchors,
        "selection_source": args.labels,
        "labels": 0,
        "model_outputs_used_for_anchor_selection": False,
        "retrieval": "production issue-flow hybrid top-100",
        "cache": cache_info,
        "scope": "SMS defect-type ordering probe; metric fixed in sms_type_probe_20260923/README.md",
    }
    (out / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(SCORER, out / SCORER.name)
    shutil.make_archive(str(out), "zip", out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "cache"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
