# -*- coding: utf-8 -*-
"""Do latency cua dung hai luong pilot sau warm-up."""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from pipeline.retrieval import load_jsonl
from pilot.service import PilotSearchService, ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=20,
                    help="so query cho moi luong")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="data/processed/pilot_latency.json")
    args = ap.parse_args()
    if not 1 <= args.queries <= 100:
        raise SystemExit("--queries phai trong khoang 1..100")

    os.chdir(ROOT)
    service = PilotSearchService(device=args.device)
    handwritten = load_jsonl("data/processed/handwritten_queries.jsonl")
    if len(handwritten) < args.queries:
        raise SystemExit(f"chi co {len(handwritten)} handwritten queries")

    print("Warm-up model va index (khong tinh vao latency)...", flush=True)
    service.warm_up()
    for i, q in enumerate(handwritten[:args.queries], 1):
        service.search({"flow": "text", "query": q["query"], "top_k": 10})
        print(f"text {i}/{args.queries}", flush=True)
    for i, q in enumerate(handwritten[:args.queries], 1):
        service.search({"flow": "issue", "key": q["key"], "top_k": 10})
        print(f"issue {i}/{args.queries}", flush=True)

    report = service.telemetry()
    report.update({
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "queries_per_flow": args.queries,
        "warmup_excluded": True,
        "text_queries": "first N from handwritten_queries.jsonl",
        "issue_queries": "the source issue keys paired with those handwritten queries",
        "configs": {
            "text": "e5-task2 + hybrid + ce-task blend 30%",
            "issue": "e5-task2 + hybrid, no rerank",
        },
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Da ghi {out}")


if __name__ == "__main__":
    main()
