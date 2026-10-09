# -*- coding: utf-8 -*-
"""Kiem tra bo contrastive Jira truoc khi dung de do hoac train."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict


YES = {"1", "y", "yes", "true", "approved", "ok"}
KEY_COLS = ("issue_a", "source_a", "issue_b", "source_b")
TRAIN_PAIRS = ("data/processed/task_pairs_v2.jsonl", "data/processed/task_pairs.jsonl")


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def training_keys(paths=TRAIN_PAIRS) -> set[str]:
    """Moi issue da xuat hien trong cap train cua model dang deploy (query hoac negative)."""
    keys: set[str] = set()
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"thieu du lieu train {path}: khong the kiem tra ro")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                keys.add(str(row["key"]).upper())
                negs = row.get("neg_keys") or []
                if isinstance(negs, str):
                    negs = ast.literal_eval(negs)
                keys.update(str(k).upper() for k in negs)
    return keys


def _norm_summary(doc: dict) -> str:
    return re.sub(r"\s+", " ", str(doc.get("summary", "")).casefold()).strip()


def validate(rows: list[dict], corpus_keys: set[str],
             train_keys: set[str] | None = None,
             docs: dict[str, dict] | None = None) -> list[str]:
    errors, family_splits = [], defaultdict(set)
    key_splits: dict[str, set[str]] = defaultdict(set)
    group_ids = set()
    for i, row in enumerate(rows, 2):
        group = (row.get("group_id") or "").strip()
        family = (row.get("family_id") or group).strip()
        split = (row.get("split") or "").strip().lower()
        if not group or group in group_ids:
            errors.append(f"dong {i}: group_id rong hoac trung: {group!r}")
        group_ids.add(group)
        if split not in {"dev", "test"}:
            errors.append(f"dong {i}: split phai la dev/test")
        family_splits[family].add(split)
        for col in ("issue_a", "issue_b"):
            key = (row.get(col) or "").strip().upper()
            if key not in corpus_keys:
                errors.append(f"dong {i}: {col}={key!r} khong co trong corpus")
        if row.get("issue_a") == row.get("issue_b"):
            errors.append(f"dong {i}: issue_a va issue_b phai khac nhau")
        for col in ("source_a", "source_b"):
            key = (row.get(col) or "").strip().upper()
            if key and key not in corpus_keys:
                errors.append(f"dong {i}: {col}={key!r} khong co trong corpus")
        present = {col: (row.get(col) or "").strip().upper() for col in KEY_COLS}
        present = {col: key for col, key in present.items() if key}
        if len(set(present.values())) < len(present):
            errors.append(f"dong {i}: mot issue lap lai trong cung nhom")
        for col, key in present.items():
            key_splits[key].add(split)
            if train_keys is not None and key in train_keys:
                errors.append(f"dong {i}: {col}={key} nam trong du lieu train cua model")
        if docs is not None:
            cols = sorted(present)
            for x in range(len(cols)):
                for y in range(x + 1, len(cols)):
                    ka, kb = present[cols[x]], present[cols[y]]
                    da, db = docs.get(ka), docs.get(kb)
                    if da is None or db is None:
                        continue
                    linked = (str(da.get("dup_of") or "").upper() == kb
                              or str(db.get("dup_of") or "").upper() == ka)
                    if linked or _norm_summary(da) == _norm_summary(db):
                        errors.append(f"dong {i}: {cols[x]}={ka} va {cols[y]}={kb} la ban trung lap")
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(f"family {family!r} bi tach qua nhieu split: {sorted(splits)}")
    for key, splits in sorted(key_splits.items()):
        if len(splits) > 1:
            errors.append(f"issue {key} xuat hien o ca {sorted(splits)}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="data/processed/contrastive_jira_review.csv")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--train-pairs", nargs="+", default=list(TRAIN_PAIRS))
    args = ap.parse_args()
    with open(args.corpus, encoding="utf-8") as f:
        docs = {d["key"].upper(): d for d in map(json.loads, filter(str.strip, f))}
    rows = load_rows(args.review)
    errors = validate(rows, set(docs), training_keys(args.train_pairs), docs)
    approved = sum((r.get("approved") or "").strip().lower() in YES for r in rows)
    print(f"{len(rows)} nhom | {approved} da duyet | {len(errors)} loi")
    if errors:
        print("\n".join("- " + e for e in errors))
        sys.exit(1)
    if not approved:
        print("Chua do model: hay dien approved=yes sau khi ky su xac nhan tung nhom.")


if __name__ == "__main__":
    main()
