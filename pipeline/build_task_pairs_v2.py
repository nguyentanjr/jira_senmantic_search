# -*- coding: utf-8 -*-
"""
Du lieu train v2: cap `actual -> issue` (nhu v1) + cap `tieu de -> issue`.

    python pipeline/build_task_pairs_v2.py

VI SAO
v1 (build_task_pairs.py, 2.487 issue) chi co mot kieu truy van. Model v1 thang
lon tren bo paraphrase nhung ~ model goc tren 50 truy van viet tay. v2 them tieu
de (cau ngan ky su tu viet) cua moi issue ngoai bo do: ~6.200 issue.

Quy tac loai tru, negative, ban bo dau: giong v1 (xem build_task_pairs.py).
Tai lieu cua cap tieu de = ban da tru summary (build_title_eval.build_title_holdout),
de model khong hoc duong tat chep tieu de. Negative co the la dang khac cua
mot issue khac, khong bao gio la chinh issue do.

Dau ra: data/processed/task_pairs_v2.jsonl, task_pairs_v2_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paraphrase_eval import build_holdout, mask_issue_refs              # noqa: E402
from build_task_pairs import expand, keys_in_file, mine_negatives, near_dup_map, select  # noqa: E402
from build_title_eval import build_title_holdout, title_query                 # noqa: E402
from retrieval import load_jsonl, strip_diacritics                           # noqa: E402


def title_items(docs, excluded, min_query: int, min_doc: int):
    out = []
    for d in docs:
        if d["key"] in excluded:
            continue
        q = mask_issue_refs(title_query(d.get("summary", ""))).strip()
        h = build_title_holdout(d)
        if len(q) >= min_query and len(h["text_dense"]) >= min_doc and len(h["text_bm25"]) >= min_doc:
            out.append((d, h, q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--paraphrase", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="data/processed/pooling_judgments_v3_llm.csv")
    ap.add_argument("--near", default="data/processed/near_duplicates.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/processed/task_pairs_v2.jsonl")
    ap.add_argument("--stats", default="data/processed/task_pairs_v2_stats.json")
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    para = {q["key"] for q in load_jsonl(args.paraphrase)}
    nb = near_dup_map(load_jsonl(args.near))
    excluded = expand(para | keys_in_file(args.gold) | keys_in_file(args.pooling), nb)

    actual = [(d, h, q, "actual") for d, h, q in
              select(docs, [build_holdout(d) for d in docs], excluded, 40, 40, 1200)]
    title = [(d, h, q, "title") for d, h, q in title_items(docs, excluded, 15, 80)]
    items = actual + title
    negs = mine_negatives([(d, h, q) for d, h, q, _ in items], nb, 3, 1, 15, args.seed)

    rows = []
    for (d, h, q, kind), ng in zip(items, negs):
        for qv, text in (("goc", q), ("khong_dau", strip_diacritics(q))):
            if qv == "khong_dau" and text == q:
                continue
            rows.append({"key": d["key"], "kind": kind, "qv": qv, "query": text,
                         "positive": h["text_dense"],
                         "negatives": [items[j][1]["text_dense"] for j in ng],
                         "neg_keys": [items[j][0]["key"] for j in ng]})

    self_neg = sum(1 for r in rows if r["key"] in r["neg_keys"])
    used = {r["key"] for r in rows} | {k for r in rows for k in r["neg_keys"]}
    if used & excluded or self_neg:
        sys.exit(f"RO RI: {len(used & excluded)} key thuoc tap loai tru, {self_neg} dong co negative la chinh no")

    stats = {"n_items": len(items), "items_by_kind": dict(Counter(k for *_, k in items)),
             "n_rows": len(rows), "rows_by_kind_qv": dict(Counter(f"{r['kind']}:{r['qv']}" for r in rows)),
             "n_issues": len({d['key'] for d, *_ in items}), "excluded": len(excluded),
             "leak_check": {"output_keys_in_excluded": 0, "self_negatives": 0}}
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
