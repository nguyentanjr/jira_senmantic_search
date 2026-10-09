"""Blind review file: union of hybrid and base-CE top-10 for the confirmed
`thua` anchors of the SMS probe. Provenance (which ranking, rank) is kept in a
separate file so the reviewer does not see it.

    python pipeline/build_sms_top10_review.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sms_type_probe import ACTUAL, EXPECTED, KHONG, NOT_THUA, SAI, SMS, THUA, clip, flow_of

LABELS = Path("data/processed/sms_type_probe_20260923/candidates.csv")
POOLS = Path("colab/sms_type_probe_bundle/data/candidate_pools.jsonl")
PRED = Path("data/processed/sms_type_probe_result/output/predictions.jsonl")
OUT = Path("data/processed/sms_top10_review_20260923")


def suggest(doc):
    summary, body = doc.get("summary", ""), doc.get("text_business", "")
    m = ACTUAL.search(body)
    focus = summary + "\n" + (m.group(1) if m else "")
    if not SMS.search(summary + "\n" + body):
        return "khac"
    hits = [g for g, ok in (("thua", THUA.search(focus) and not NOT_THUA.search(summary)),
                            ("sai", SAI.search(focus)), ("khong_gui", KHONG.search(focus))) if ok]
    return hits[0] if len(hits) == 1 else ("tron" if hits else "khac")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Output exists: {OUT}")
    confirmed = {r["key"]: r["confirm_group"] for r in csv.DictReader(open(LABELS, encoding="utf-8-sig"))}
    docs = {json.loads(l)["key"]: json.loads(l) for l in open("data/processed/corpus.jsonl", encoding="utf-8")}
    pools = {json.loads(l)["query_key"]: json.loads(l) for l in open(POOLS, encoding="utf-8")}
    preds = {json.loads(l)["query_key"]: json.loads(l)["reranked_keys"] for l in open(PRED, encoding="utf-8")}
    anchors = sorted(k for k, g in confirmed.items() if g == "thua")
    rows, prov = [], []
    for q in anchors:
        hyb, ce = pools[q]["baseline_ranking"], preds[q]
        union = list(dict.fromkeys(hyb[:10] + ce[:10]))
        union.sort(key=lambda k: hashlib.sha256(f"{q}:{k}".encode()).hexdigest())  # hide order
        for k in union:
            d = docs[k]
            m = EXPECTED.search(d.get("text_business", ""))
            a = ACTUAL.search(d.get("text_business", ""))
            pre = confirmed.get(k, "")
            rows.append({"anchor": q, "anchor_summary": docs[q]["summary"], "candidate": k,
                         "suggested": pre if pre else suggest(d),
                         "suggested_source": "user_confirmed" if pre else "regex",
                         "confirm": pre if pre in ("thua", "sai", "khong_gui", "tron") else "",
                         "note": "", "summary": d["summary"],
                         "actual": clip(a.group(1) if a else d.get("text_business", ""), 400),
                         "expected": clip(m.group(1) if m else "", 300), "url": d.get("url", "")})
            prov.append({"anchor": q, "candidate": k,
                         "hybrid_rank": hyb.index(k) + 1, "ce_rank": ce.index(k) + 1})
    OUT.mkdir(parents=True)
    with open(OUT / "review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "provenance.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prov), encoding="utf-8")
    uniq = {r["candidate"] for r in rows}
    todo = sum(1 for r in rows if not r["confirm"])
    print(f"anchors={len(anchors)} rows={len(rows)} unique_candidates={len(uniq)} need_label={todo}")


if __name__ == "__main__":
    main()
