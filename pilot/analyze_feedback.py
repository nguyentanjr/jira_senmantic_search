# -*- coding: utf-8 -*-
"""Bao cao pilot tu search log + feedback ma khong coi im lang la negative."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pipeline.feedback_state import replay


def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pct(values: list[float]) -> dict:
    return ({"p50": round(float(np.percentile(values, 50)), 1),
             "p95": round(float(np.percentile(values, 95)), 1)}
            if values else {"p50": None, "p95": None})


def analyze(searches: list[dict], feedback: list[dict]) -> dict:
    by_id = {str(s.get("search_id")): s for s in searches if s.get("search_id")}
    # Phat lai theo thu tu: doi y, bo chon, "khong co dap an" deu duoc ton trong.
    latest, no_answer, _, label_changes = replay(
        feedback, lambda e: (str(e.get("search_id", ""))
                             if str(e.get("search_id", "")) in by_id else None))
    active_no_answer = set(no_answer)
    reasons, flow_latency = Counter(), defaultdict(lambda: defaultdict(list))
    full_p3, full_p5 = [], []
    statuses = Counter()
    per_search = []
    for sid, search in by_id.items():
        flow = search.get("flow", "unknown")
        for field, value in (search.get("latency") or {}).items():
            if isinstance(value, (int, float)):
                flow_latency[flow][field].append(float(value))
        shown = {str(r["key"]).upper(): int(r["rank"])
                 for r in search.get("results", [])}
        labels = {shown[key]: event["verdict"] for (event_sid, key), event in latest.items()
                  if event_sid == sid and key in shown}
        for (event_sid, _), event in latest.items():
            if event_sid == sid and event.get("verdict") == "not_related":
                reasons[str(event.get("reason") or "missing_reason")] += 1
        is_no_answer = sid in active_no_answer
        has_related = any(v == "related" for v in labels.values()) and not is_no_answer
        complete_top5 = is_no_answer or all(k in labels for k in (1, 2, 3, 4, 5))
        if search.get("review_mode") == "disabled" and not labels and not is_no_answer:
            # Nguoi dung da tat danh gia: khong tinh la bo do.
            statuses["feedback_disabled"] += 1
        elif is_no_answer:
            statuses["no_answer"] += 1
        elif has_related:
            statuses["has_related_click"] += 1
        else:
            statuses["unknown_no_positive_click"] += 1
        if search.get("review_mode") == "full_top5":
            statuses["full_top5_assigned"] += 1
            if complete_top5:
                statuses["full_top5_completed"] += 1
                full_p3.append(sum(labels.get(k) == "related" for k in (1, 2, 3)) / 3
                               if not is_no_answer else 0.0)
                full_p5.append(sum(labels.get(k) == "related" for k in (1, 2, 3, 4, 5)) / 5
                               if not is_no_answer else 0.0)
        per_search.append({"search_id": sid, "flow": flow,
                           "review_mode": search.get("review_mode", "legacy"),
                           "complete_top5": complete_top5, "no_answer": is_no_answer,
                           "has_related_click": has_related,
                           "n_labeled": len(labels),
                           "filters": search.get("filters", {})})

    latency = {flow: {field: pct(values) for field, values in fields.items()}
               for flow, fields in flow_latency.items()}
    return {
        "n_searches": len(by_id), "status_counts": dict(statuses),
        "precision_from_complete_top5_only": {
            "n": len(full_p5),
            "p@3": round(float(np.mean(full_p3)), 4) if full_p3 else None,
            "p@5": round(float(np.mean(full_p5)), 4) if full_p5 else None,
        },
        "negative_reasons": dict(reasons), "latency_ms": latency,
        "label_changes": dict(label_changes),
        "interpretation": "unknown_no_positive_click is unknown, not negative",
        "per_search": per_search,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--searches", default="data/pilot/searches.jsonl")
    ap.add_argument("--feedback", default="data/pilot/feedback.jsonl")
    ap.add_argument("--out", default="data/pilot/derived/pilot_report.json")
    args = ap.parse_args()
    result = analyze(read_jsonl(args.searches), read_jsonl(args.feedback))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_search"},
                     ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
