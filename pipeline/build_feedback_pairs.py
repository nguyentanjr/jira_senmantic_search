# -*- coding: utf-8 -*-
"""Chuyen click pilot thanh cap train co audit va tach eval theo query.

Quy tac an toan:
- chi click ro rang ``related`` / ``not_related`` moi tao nhan;
- click sau cung cho cung (query, result) thang, nhung van thong ke xung dot;
- mot query chi tao cap train khi co it nhat mot positive VA mot explicit negative;
- split theo *nhom query* bang hash on dinh, khong tach tung click;
- issue thuoc cac bo eval hien co hoac contrastive review khong vao train;
- van ban duoc scrub lai truoc khi ghi ra.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from build_corpus import scrub
from feedback_state import replay


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(text: str) -> str:
    return scrub(text or "")[0].strip()


def query_identity(row: dict) -> str:
    flow = str(row.get("flow", "text")).strip().lower()
    if flow == "issue":
        return "issue:" + str(row.get("source_key", "")).strip().upper()
    query = re.sub(r"\s+", " ", str(row.get("query", "")).strip().casefold())
    return "text:" + query


def stable_split(identity: str, eval_fraction: float) -> str:
    value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12], 16)
    return "eval" if value / float(16 ** 12) < eval_fraction else "train"


def existing_eval_keys(gold_path: str, pooling_path: str,
                       contrast_path: str, clean_split_path: str = "") -> set[str]:
    """Moi issue dang nam trong mot bo do: gold eval, pooling (ca query lan doc da
    cham), contrastive va paraphrase sach cua luong text."""
    keys: set[str] = set()
    for row in read_jsonl(gold_path):
        keys.update(str(row.get(k, "")).upper() for k in ("a", "b") if row.get(k))
    if os.path.exists(pooling_path):
        with open(pooling_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                for col in ("query_key", "doc_key"):
                    if row.get(col):
                        keys.add(row[col].strip().upper())
    if clean_split_path and os.path.exists(clean_split_path):
        with open(clean_split_path, encoding="utf-8") as f:
            keys.update(str(k).strip().upper()
                        for k in json.load(f).get("paraphrase_clean_keys", []))
    if os.path.exists(contrast_path):
        with open(contrast_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                for col in ("issue_a", "issue_b", "source_a", "source_b"):
                    if row.get(col):
                        keys.add(row[col].strip().upper())
    return keys


def build(events: list[dict], docs: dict[str, dict], excluded: set[str],
          eval_fraction: float = 0.2,
          searches: list[dict] | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    search_by_id = {str(r.get("search_id", "")): r for r in (searches or [])
                    if r.get("search_id")}
    enriched = []
    for event in events:
        search = search_by_id.get(str(event.get("search_id", "")))
        if search:
            # Search log la nguon chuan cho query/filter/ranking; click chi cung
            # cap verdict, reason va note.
            shown_rank = {str(r.get("key", "")).upper(): int(r.get("rank", 0))
                          for r in search.get("results", [])}
            result_key = str(event.get("result_key", "")).upper()
            event = {**event, "flow": search.get("flow", "text"),
                     "query": search.get("query", ""),
                     "source_key": search.get("source_key", ""),
                     "filters": search.get("filters", {}),
                     "review_mode": search.get("review_mode", "standard"),
                     "rank": shown_rank.get(result_key, int(event.get("rank", 0) or 0))}
        enriched.append(event)

    tagged = []
    for event in enriched:
        ident = query_identity(event)
        group = ("search:" + str(event.get("search_id"))
                 if event.get("search_id") else ident)
        tagged.append({**event, "identity": ident, "group_identity": group})
    # Doi y, bo chon va "khong co dap an" duoc phat lai theo dung thu tu.
    latest, active_no_answer, seen_verdicts, replay_stats = replay(
        tagged, lambda e: e["group_identity"])

    groups: dict[str, list[dict]] = defaultdict(list)
    audit, stats = [], Counter()
    for (group, result), event in latest.items():
        ident = event["identity"]
        flow = str(event.get("flow", "text")).strip().lower()
        source = str(event.get("source_key", "")).strip().upper()
        reasons = []
        if result not in docs:
            reasons.append("result_not_in_corpus")
        if flow == "issue" and source not in docs:
            reasons.append("source_not_in_corpus")
        if result in excluded or (source and source in excluded):
            reasons.append("touches_existing_eval")
        split = stable_split(ident, eval_fraction)
        status = ";".join(reasons) if reasons else split
        row = {
            "identity": ident, "group_identity": group,
            "search_id": event.get("search_id", ""), "split": split, "flow": flow,
            "source_key": source, "result_key": result,
            "verdict": str(event.get("verdict", "")).lower(),
            "reason": str(event.get("reason", "")),
            "filters": json.dumps(event.get("filters", {}), ensure_ascii=False, sort_keys=True),
            "review_mode": event.get("review_mode", "legacy"),
            "last_created_at": event.get("created_at", ""),
            "had_conflicting_clicks": len(seen_verdicts[(group, result)]) > 1,
            "status": status,
        }
        audit.append(row)
        stats[row["status"]] += 1
        if not reasons:
            groups[group].append({**event, "result_key": result, "split": split})

    # "Khong co dap an" la nhan query-level, khong bien ngam thanh hard negative.
    eval_rows = []
    for group, event in active_no_answer.items():
        ident = query_identity(event)
        eval_rows.append({
            "identity": ident, "group_identity": group,
            "search_id": event.get("search_id", ""),
            "split": stable_split(ident, eval_fraction),
            "flow": event.get("flow", "text"), "query": clean_text(event.get("query", "")),
            "source_key": str(event.get("source_key", "")).upper(),
            "filters": event.get("filters", {}), "verdict": "no_answer",
        })
        stats["no_answer"] += 1

    train_pairs = []
    for group, rows in sorted(groups.items()):
        ident = rows[0]["identity"]
        split = rows[0]["split"]
        positives = sorted({r["result_key"] for r in rows if r["verdict"] == "related"})
        negatives = sorted({r["result_key"] for r in rows if r["verdict"] == "not_related"})
        if split == "eval":
            eval_rows.extend({
                "identity": ident, "flow": r.get("flow", "text"),
                "group_identity": group, "search_id": r.get("search_id", ""),
                "split": split, "filters": r.get("filters", {}),
                "query": clean_text(str(r.get("query", ""))),
                "source_key": str(r.get("source_key", "")).upper(),
                "result_key": r["result_key"], "verdict": r["verdict"],
                "reason": r.get("reason", ""),
            } for r in rows)
            continue
        if not positives or not negatives:
            stats["train_group_missing_positive_or_negative"] += 1
            continue
        sample = rows[0]
        flow = str(sample.get("flow", "text")).lower()
        source = str(sample.get("source_key", "")).upper()
        query = (docs[source].get("text_dense") or docs[source].get("summary", "")
                 if flow == "issue" else str(sample.get("query", "")))
        query = clean_text(query)
        for positive in positives:
            train_pairs.append({
                "key": source or "FB-" + hashlib.sha256(ident.encode()).hexdigest()[:12],
                "kind": f"feedback_{flow}", "flow": flow, "query": query,
                "positive": clean_text(docs[positive].get("text_dense")
                                       or docs[positive].get("summary", "")),
                "positive_key": positive,
                "negatives": [clean_text(docs[n].get("text_dense")
                                         or docs[n].get("summary", "")) for n in negatives],
                "neg_keys": negatives, "source": "pilot_explicit_feedback",
                "negative_reasons": {n: next((r.get("reason", "") for r in rows
                                               if r["result_key"] == n), "")
                                     for n in negatives},
                "search_id": sample.get("search_id", ""),
                "filters": sample.get("filters", {}),
            })
    # Kiem tra tu search log, khong suy dien query khong click la negative.
    full = [s for s in (searches or []) if s.get("review_mode") == "full_top5"]
    completed = 0
    for search in full:
        group = "search:" + str(search.get("search_id"))
        ranks = {int(event.get("rank", 0)) for (g, _), event in latest.items()
                 if g == group}
        if group in active_no_answer or all(k in ranks for k in (1, 2, 3, 4, 5)):
            completed += 1
    stats.update(replay_stats)
    stats["searches_input"] = len(searches or [])
    stats["full_top5_assigned"] = len(full)
    stats["full_top5_completed"] = completed
    stats["events_input"] = len(events)
    stats["latest_explicit_clicks"] = len(latest)
    stats["conflicting_click_pairs"] = sum(len(v) > 1 for v in seen_verdicts.values())
    stats["train_pairs"] = len(train_pairs)
    stats["eval_clicks"] = len(eval_rows)
    return train_pairs, eval_rows, sorted(audit, key=lambda x: (x["identity"], x["result_key"])), dict(stats)


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    d = "data/processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", default="data/pilot/feedback.jsonl")
    ap.add_argument("--searches", default="data/pilot/searches.jsonl")
    ap.add_argument("--corpus", default=f"{d}/corpus.jsonl")
    ap.add_argument("--gold-eval", default=f"{d}/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default=f"{d}/pooling_judgments_v4_llm.csv")
    ap.add_argument("--contrast", default=f"{d}/contrastive_jira_review.csv")
    ap.add_argument("--clean-split", default=f"{d}/finetune_split_clean.json",
                    help="paraphrase_clean_keys = bo do luong text")
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--out-dir", default="data/pilot/derived")
    args = ap.parse_args()
    if not 0 < args.eval_fraction < 1:
        raise SystemExit("--eval-fraction phai nam trong (0, 1)")

    docs = {r["key"].upper(): r for r in read_jsonl(args.corpus)}
    for path in (args.gold_eval, args.pooling, args.contrast, args.clean_split):
        if not os.path.exists(path):
            raise SystemExit(f"thieu bo do {path}: khong the chong ro an toan")
    excluded = existing_eval_keys(args.gold_eval, args.pooling, args.contrast,
                                  args.clean_split)
    train, eval_rows, audit, stats = build(
        read_jsonl(args.feedback), docs, excluded, args.eval_fraction,
        read_jsonl(args.searches))
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "feedback_train_pairs.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "feedback_eval.jsonl"), eval_rows)
    audit_path = os.path.join(args.out_dir, "feedback_audit.csv")
    with open(audit_path, "w", encoding="utf-8-sig", newline="") as f:
        cols = ["identity", "group_identity", "search_id", "split", "flow",
                "source_key", "result_key", "verdict", "reason", "filters",
                "review_mode", "last_created_at", "had_conflicting_clicks", "status"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(audit)
    manifest = {"policy": {
        "negative": "explicit not_related only", "click_resolution": "latest wins",
        "split_unit": "normalized query/source issue; samples grouped by search_id",
        "eval_fraction": args.eval_fraction,
        "excluded_existing_eval_keys": len(excluded), "text_scrubbed_on_output": True,
    }, "stats": stats}
    with open(os.path.join(args.out_dir, "feedback_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
