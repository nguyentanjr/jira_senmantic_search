# -*- coding: utf-8 -*-
"""
Bo do thu hai cho luong mo ta tu do: TIEU DE -> issue.

    python pipeline/build_title_eval.py

VI SAO
Bo paraphrase lay `actual` lam truy van. Model fine-tune tren cung kieu do thang
lon tren test nhung gan nhu khong hon model goc tren 50 truy van viet tay
(eval_handwritten.py). Tieu de la cau NGAN do chinh ky su viet (p50 ~50 ky tu,
bo tag) -> gan cach nguoi dung go hon.

CACH DUNG (giong build_paraphrase_eval.py)
    truy van : summary, bo cac [tag], che ma issue
    tai lieu : issue da TRU summary khoi text_dense / text_bm25 (tag cung mat
               vi tag nam trong summary); signals chi con gia tri nhin thay
               trong phan con lai. `actual` GIU nguyen.
Cung tap key voi paraphrase_queries -> dung lai tach dev/test cua
eval_rerankers.make_split.

Dau ra: data/processed/corpus_title_holdout.jsonl, title_queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_paraphrase_eval import bin_of, jaccard, mask_issue_refs, toks  # noqa: E402
from retrieval import load_jsonl                                          # noqa: E402

TAG_RE = re.compile(r"\[[^\]]*\]")


def title_query(summary: str) -> str:
    return " ".join(TAG_RE.sub(" ", summary or "").split())


def _subtract_ci(text: str, part: str) -> str:
    """Xoa moi lan xuat hien cua `part` (khong phan biet hoa thuong)."""
    if not text or not part:
        return text or ""
    return re.sub(r"\n{3,}", "\n\n", re.sub(re.escape(part), " ", text, flags=re.IGNORECASE)).strip()


def build_title_holdout(d: dict) -> dict:
    """Ban sao cua doc da tru summary khoi moi truong van ban."""
    r = dict(d)
    # Mo ta hay lap lai nguyen van tieu de (4,3% truy van truoc khi tru) -> tru ca
    # summary goc lan tieu de da bo tag, giong cach tru `actual` o bo paraphrase.
    biz, tech = d.get("text_business", ""), d.get("text_technical", "")
    for part in {d.get("summary", ""), title_query(d.get("summary", ""))}:
        if len(part) >= 12:
            biz, tech = _subtract_ci(biz, part), _subtract_ci(tech, part)
    sections = d.get("sections") or {}
    labels = " ".join(d.get("labels") or [])
    visible = " ".join([biz, tech, labels]).lower()
    signals = {k: [v for v in vs if str(v).lower() in visible]
               for k, vs in (d.get("signals") or {}).items()}
    signals = {k: v for k, v in signals.items() if v}
    r["summary"] = ""
    r["signals"] = signals
    r["text_dense"] = mask_issue_refs("\n".join(
        x for x in [biz, sections.get("expected", "")] if x).strip())
    r["text_bm25"] = mask_issue_refs("\n".join(x for x in [
        biz, tech, " ".join(v for vs in signals.values() for v in vs), labels] if x).strip())
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--paraphrase", default="data/processed/paraphrase_queries.jsonl")
    ap.add_argument("--min-query-chars", type=int, default=15)
    ap.add_argument("--min-doc-chars", type=int, default=80)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    holdout = [build_title_holdout(d) for d in docs]
    eval_keys = {q["key"] for q in load_jsonl(args.paraphrase)}
    queries = []
    for d, h in zip(docs, holdout):
        if d["key"] not in eval_keys:
            continue
        q = mask_issue_refs(title_query(d.get("summary", ""))).strip()
        if len(q) < args.min_query_chars or len(h["text_bm25"]) < args.min_doc_chars:
            continue
        j = jaccard(toks(q), toks(h["text_bm25"]))
        queries.append({"key": d["key"], "query": q, "relevant": [d["key"]],
                        "jaccard": round(j, 4), "jac_bin": bin_of(j),
                        "lexical_group": bool(d.get("lexical_group"))})

    with open(os.path.join(args.out_dir, "corpus_title_holdout.jsonl"), "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "title_queries.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    # ro ri: tieu de (>= 12 ky tu) con nguyen van trong tai lieu cua chinh no?
    by = {h["key"]: h for h in holdout}
    leak = sum(1 for q in queries if len(q["query"]) >= 12 and q["query"].lower() in by[q["key"]]["text_bm25"].lower())
    print(f"{len(queries)} truy van tieu de / {len(eval_keys)} key eval | "
          f"tieu de con nguyen van trong tai lieu: {leak} ({100 * leak / max(1, len(queries)):.1f}%)")


if __name__ == "__main__":
    main()
