# -*- coding: utf-8 -*-
"""
Dung cap train DUNG DANG BAI TOAN luong mo ta tu do: mo ta trieu chung -> issue.

    python pipeline/build_task_pairs.py

VI SAO
Lan fine-tune e5 truoc that bai (HANDOFF §5.4b): cap train la issue <-> issue
nhan yeu, lech voi bai toan "mo ta ngan -> issue". O day cap duoc dung GIONG
cach build_paraphrase_eval.py dung bo do:
    query    = sections.actual (che ma issue, cat 1.200 ky tu)
    positive = text_dense cua ban holdout (da tru actual khoi moi truong)
Moi dong co hai ban truy van: co dau va bo dau (strip_diacritics) de model hoc
ca kieu go khong dau.

KHONG RO SANG BO DO
Loai: moi issue cua paraphrase_queries (3.829, ca phan khong sach), moi ma
issue xuat hien trong similar_gold_eval va pooling_judgments_v3_llm, VA cac
issue gan trung (near_duplicates, jaccard >= 0,72) voi nhung issue do.
Negative cung chi lay trong tap issue train. Script tu kiem lai tren dau ra.

KHAC BIET CAN GHI TRONG BAO CAO
Bo do lay moi issue co actual >= 80 ky tu -> issue con lai de train co actual
NGAN hon (mac dinh 40-79 ky tu, hoac tai lieu qua ngan sau khi tru).

NEGATIVE
3 hard negative tu BM25 (fold="add") tren tap issue train, bo chinh no, bo
issue gan trung va issue cung summary; 1 negative ngau nhien. Van co the lan
issue lien quan that (QA hay tao ticket song song) -- nhieu nhan dang biet.

Dau ra: data/processed/task_pairs.jsonl, task_pairs_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paraphrase_eval import build_holdout, mask_issue_refs       # noqa: E402
from retrieval import BM25, load_jsonl, strip_diacritics, tokenize, top_k_idx  # noqa: E402

KEY_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")


def keys_in_file(path: str) -> set[str]:
    """Moi ma issue xuat hien o bat ky dau trong file (du thua = an toan)."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        return set(KEY_RE.findall(f.read()))


def near_dup_map(rows) -> dict[str, set[str]]:
    nb: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        nb[r["a"]].add(r["b"])
        nb[r["b"]].add(r["a"])
    return nb


def expand(keys: set[str], nb: dict[str, set[str]]) -> set[str]:
    out = set(keys)
    for k in keys:
        out |= nb.get(k, set())
    return out


def select(docs, holdouts, excluded: set[str], min_query: int, min_doc: int,
           max_query: int) -> list[tuple[dict, dict, str]]:
    out = []
    for d, h in zip(docs, holdouts):
        if d["key"] in excluded:
            continue
        actual = (d.get("sections") or {}).get("actual", "")
        if len(actual) < min_query:
            continue
        if len(h.get("text_dense", "")) < min_doc or len(h.get("text_bm25", "")) < min_doc:
            continue
        q = mask_issue_refs(actual)[:max_query].strip()
        if q:
            out.append((d, h, q))
    return out


def mine_negatives(items, nb, n_hard: int, n_random: int, pool: int, seed: int):
    """-> list[list[int]] chi so negative (trong `items`) cho moi truy van."""
    rng = random.Random(seed)
    bm = BM25([tokenize(h.get("text_bm25", ""), fold="add") for _, h, _ in items])
    keys = [d["key"] for d, _, _ in items]
    summ = [" ".join((d.get("summary") or "").lower().split()) for d, _, _ in items]
    n = len(items)
    out = []
    for i, (d, _, q) in enumerate(items):
        bad = nb.get(keys[i], set())

        def ok(j):
            return (j != i and keys[j] != keys[i] and keys[j] not in bad
                    and (not summ[i] or summ[j] != summ[i]))

        cand = [int(j) for j in top_k_idx(bm.score(tokenize(q, fold="add")), pool + 1)]
        hard = [j for j in cand if ok(j)][:n_hard]
        rand = []
        tries = 0
        while len(rand) < n_random and tries < 100 and n > 1:
            j = rng.randrange(n)
            tries += 1
            if ok(j) and j not in hard and j not in rand:
                rand.append(j)
        out.append(hard + rand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--paraphrase", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="data/processed/pooling_judgments_v3_llm.csv")
    ap.add_argument("--near", default="data/processed/near_duplicates.jsonl")
    ap.add_argument("--min-query-chars", type=int, default=40)
    ap.add_argument("--min-doc-chars", type=int, default=40)
    ap.add_argument("--max-query-chars", type=int, default=1200)
    ap.add_argument("--n-hard", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=1)
    ap.add_argument("--hard-pool", type=int, default=15)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/processed/task_pairs.jsonl")
    ap.add_argument("--stats", default="data/processed/task_pairs_stats.json")
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    para = {q["key"] for q in load_jsonl(args.paraphrase)}
    gold, pool = keys_in_file(args.gold), keys_in_file(args.pooling)
    nb = near_dup_map(load_jsonl(args.near)) if os.path.exists(args.near) else {}
    excluded = expand(para | gold | pool, nb)

    holdouts = [build_holdout(d) for d in docs]
    items = select(docs, holdouts, excluded, args.min_query_chars,
                   args.min_doc_chars, args.max_query_chars)
    negs = mine_negatives(items, nb, args.n_hard, args.n_random, args.hard_pool, args.seed)

    rows = []
    for (d, h, q), ng in zip(items, negs):
        for qv, text in (("goc", q), ("khong_dau", strip_diacritics(q))):
            if qv == "khong_dau" and text == q:
                continue
            rows.append({"key": d["key"], "qv": qv, "query": text,
                         "positive": h["text_dense"],
                         "negatives": [items[j][1]["text_dense"] for j in ng],
                         "neg_keys": [items[j][0]["key"] for j in ng]})

    # ---------------------------------------------------------- tu kiem ro ri
    used = {r["key"] for r in rows} | {k for r in rows for k in r["neg_keys"]}
    leak = used & excluded
    if leak:
        sys.exit(f"RO RI: {len(leak)} ma issue dau ra thuoc tap loai tru, vd {sorted(leak)[:5]}")

    lens = np.asarray([len(q) for _, _, q in items]) if items else np.zeros(1)
    stats = {
        "n_corpus": len(docs), "n_issues": len(items), "n_rows": len(rows),
        "rows_by_qv": dict(Counter(r["qv"] for r in rows)),
        "excluded": {"paraphrase": len(para), "gold_keys": len(gold), "pooling_keys": len(pool),
                     "after_near_dup_expansion": len(excluded)},
        "query_chars_p10_p50_p90": [int(np.percentile(lens, p)) for p in (10, 50, 90)],
        "negatives_per_row": args.n_hard + args.n_random,
        "leak_check": {"output_keys_in_excluded": 0},
        "params": vars(args),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "params"}, ensure_ascii=False, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
