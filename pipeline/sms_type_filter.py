"""Demote SMS issues of a different defect type (post-processing, no training).

User decision 2026-09-23: for an "extra SMS" issue, "wrong SMS content" issues
may appear but must rank below "extra SMS" issues; the same holds between every
pair of the three SMS types (thua / sai / khong_gui).

Rule, fixed before measuring:
- the query issue must have exactly one SMS type, else nothing changes;
- inside the first ``window`` results, candidates with a DIFFERENT SMS type are
  moved to the end of the window (kept, not deleted), preserving order;
- candidates with no SMS type are treated as neutral and keep their order.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sms_type_probe import classify, load_direction

WINDOW = 30


def sms_types(docs: list[dict], direction_path: str = "data/processed/direction_labels_llm.jsonl") -> dict[str, str]:
    direction = load_direction(direction_path)
    out = {}
    for doc in docs:
        c = classify(doc, direction)
        if c:
            out[doc["key"]] = c["group"]
    return out


def demote(query_type: str | None, ranked_keys: list[str], types: dict[str, str],
           window: int = WINDOW) -> list[str]:
    if not query_type or window <= 1:
        return list(ranked_keys)
    head = list(ranked_keys[:window])
    flagged = [k for k in head if types.get(k) not in (None, query_type)]
    if not flagged or len(flagged) == len(head):
        return list(ranked_keys)
    drop = set(flagged)
    return [k for k in head if k not in drop] + flagged + list(ranked_keys[window:])


if __name__ == "__main__":
    docs = [json.loads(l) for l in open("data/processed/corpus.jsonl", encoding="utf-8")]
    t = sms_types(docs)
    from collections import Counter
    print(len(docs), Counter(t.values()))
