# -*- coding: utf-8 -*-
"""Phat lai su kien feedback pilot theo dung thu tu -> nhan cuoi cung.

Quy uoc (dung chung cho bao cao pilot va build_feedback_pairs):
- ``related`` / ``not_related`` dat nhan cho mot ket qua; lan sau cung thang.
- ``cleared`` co ``result_key``: bo nhan cua ket qua do.
- ``cleared`` khong co ``result_key``: bo "khong co dap an" cua luot tim.
- ``no_answer``: top-10 khong co ket qua lien quan -> bo cac nhan ``related``
  cua luot (nhan ``not_related`` van dung, van giu).
- ``related`` sau ``no_answer`` -> huy ``no_answer``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Iterable


def replay(events: Iterable[dict], group_of: Callable[[dict], str | None]
           ) -> tuple[dict[tuple[str, str], dict], dict[str, dict],
                      dict[tuple[str, str], set[str]], Counter]:
    labels: dict[tuple[str, str], dict] = {}
    no_answer: dict[str, dict] = {}
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    stats: Counter = Counter()
    for event in events:
        group = group_of(event)
        if not group:
            continue
        verdict = str(event.get("verdict", "")).strip().lower()
        key = str(event.get("result_key", "") or "").strip().upper()
        if verdict == "no_answer":
            dropped = [k for k, e in labels.items()
                       if k[0] == group and e.get("verdict") == "related"]
            for k in dropped:
                del labels[k]
            stats["related_dropped_by_no_answer"] += len(dropped)
            no_answer[group] = event
        elif verdict == "cleared":
            if key:
                stats["labels_cleared"] += labels.pop((group, key), None) is not None
            else:
                stats["no_answer_cleared"] += no_answer.pop(group, None) is not None
        elif verdict in {"related", "not_related"} and key:
            labels[(group, key)] = {**event, "verdict": verdict, "result_key": key}
            seen[(group, key)].add(verdict)
            if verdict == "related" and no_answer.pop(group, None) is not None:
                stats["no_answer_cancelled_by_related"] += 1
    return labels, no_answer, seen, stats
