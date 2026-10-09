# -*- coding: utf-8 -*-
"""
Lay mau cap `Cloners` de soat tay: clone co thuc su la duplicate khong?

    python pipeline/audit_clone_links.py --n 50

Ly do ton tai:
Tap gold hien tai (138 cap) duoc dao tu comment ky su tu ghi "trung/tuong tu
RO-xxx". Ky su chi ghi duoc cau do khi HO DA TIM RA cap trung -- bang chinh
keyword search cua Jira. Nen tap gold gan nhu chi chua nhung cap ma keyword
search da giai duoc, va khong dung lam thuoc do cho retrieval ngu nghia.

Quan he `Cloners` do Jira ghi, khong phu thuoc ky su co nho go comment hay
khong -> khong mang thien lech do. Trong corpus co ~4.178 canh clone hai dau
deu nam trong corpus, gap ~30 lan tap gold.

NHUNG clone != duplicate. Clone thuong la cung bug test lai o moi truong /
version khac, hoac clone template test case. Script nay lay mau ngau nhien de
uoc luong ty le thuc su la duplicate truoc khi quyet dinh mo rong tap gold.

Dau ra:
  data/processed/clone_audit_sample.csv   mo bang Excel, dien cot `label`
  data/processed/clone_audit_sample.txt   doc bang mat tren terminal
  data/processed/clone_link_stats.json    thong ke tren TOAN BO cap clone

Cot `label` dien tay, bon gia tri:
  dup     cung mot defect -- dung lam positive
  related lien quan nhung khac defect -- KHONG dung lam positive
  diff    khac han
  ?       khong du thong tin de ket luan
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter

TOKEN_RE = re.compile(r"[0-9a-zà-ỹ_]{2,}")


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall(unicodedata.normalize("NFKC", (s or "").lower())))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_corpus(path: str) -> dict[str, dict]:
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs[d["key"]] = d
    return docs


def clone_pairs(docs: dict[str, dict]) -> list[tuple[str, str]]:
    """Canh clone vo huong, hai dau deu nam trong corpus, da khu trung."""
    seen = set()
    for key, d in docs.items():
        for lk in d.get("links", []):
            if lk.get("type") != "Cloners":
                continue
            tgt = lk.get("target")
            if not tgt or tgt == key or tgt not in docs:
                continue
            seen.add((key, tgt) if key < tgt else (tgt, key))
    return sorted(seen)


def days_between(a: str, b: str) -> str:
    from datetime import date
    try:
        ya, ma, da = (int(x) for x in a.split("-"))
        yb, mb, db = (int(x) for x in b.split("-"))
        return str(abs((date(ya, ma, da) - date(yb, mb, db)).days))
    except Exception:
        return ""


def row_for(a: str, b: str, docs: dict[str, dict]) -> dict:
    da, db = docs[a], docs[b]
    fa, fb = da.get("facets", {}), db.get("facets", {})
    sa, sb = da.get("summary", ""), db.get("summary", "")
    return {
        "a": a,
        "b": b,
        "sum_jaccard": round(jaccard(toks(sa), toks(sb)), 3),
        "same_root_cause": int(bool(fa.get("root_cause"))
                               and fa.get("root_cause") == fb.get("root_cause")),
        "same_env": int(bool(fa.get("test_env"))
                        and fa.get("test_env") == fb.get("test_env")),
        "day_gap": days_between(da.get("created", ""), db.get("created", "")),
        "summary_a": sa,
        "summary_b": sb,
        "root_cause_a": fa.get("root_cause", ""),
        "root_cause_b": fb.get("root_cause", ""),
        "env_a": fa.get("test_env", ""),
        "env_b": fb.get("test_env", ""),
        "url_a": da.get("url", ""),
        "url_b": db.get("url", ""),
        "label": "",
        "note": "",
    }


FIELDS = ["a", "b", "label", "note", "sum_jaccard", "same_root_cause",
          "same_env", "day_gap", "summary_a", "summary_b",
          "root_cause_a", "root_cause_b", "env_a", "env_b", "url_a", "url_b"]


def write_csv(path: str, rows: list[dict]) -> None:
    # utf-8-sig de Excel doc dung tieng Viet
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_txt(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("SOAT CAP CLONE -- dien label: dup / related / diff / ?\n")
        f.write("=" * 78 + "\n\n")
        for i, r in enumerate(rows, 1):
            f.write(f"[{i:>3}] {r['a']}  <->  {r['b']}"
                    f"   (jaccard tieu de = {r['sum_jaccard']}, "
                    f"cach nhau {r['day_gap']} ngay)\n")
            f.write(f"  A: {r['summary_a']}\n")
            f.write(f"  B: {r['summary_b']}\n")
            if r["root_cause_a"] or r["root_cause_b"]:
                f.write(f"  root cause : {r['root_cause_a']!r}"
                        f" | {r['root_cause_b']!r}\n")
            if r["env_a"] or r["env_b"]:
                f.write(f"  moi truong : {r['env_a']!r} | {r['env_b']!r}\n")
            f.write(f"  {r['url_a']}\n  {r['url_b']}\n")
            f.write("  label: ____________\n\n")


def bucket(x: float) -> str:
    for lo in (0.9, 0.8, 0.6, 0.4, 0.2):
        if x >= lo:
            return f">={lo}"
    return "<0.2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--n", type=int, default=50, help="so cap lay mau")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        sys.exit(f"khong thay corpus: {args.corpus}")

    docs = load_corpus(args.corpus)
    pairs = clone_pairs(docs)
    if not pairs:
        sys.exit("khong tim thay canh Cloners nao hai dau deu trong corpus")

    all_rows = [row_for(a, b, docs) for a, b in pairs]

    # -- thong ke tren TOAN BO cap clone -------------------------------------
    # Cau hoi quan trong: clone co bi dinh dung cai bias cua tap gold khong?
    # Neu phan lon cap clone co jaccard tieu de rat cao thi chung cung chi la
    # cap "de", va nguon nhan nay khong cuu duoc benchmark ngu nghia.
    js = sorted(r["sum_jaccard"] for r in all_rows)
    hist = Counter(bucket(x) for x in js)
    n = len(js)

    def pct(i: float) -> float:
        return round(js[min(int(i * n), n - 1)], 3)

    # doi chieu voi tap gold hien tai, neu co
    gold_js: list[float] = []
    gold_path = os.path.join(args.out_dir, "dup_candidates_from_text.jsonl")
    if os.path.exists(gold_path):
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                a, b = g.get("a"), g.get("b")
                if a in docs and b in docs:
                    gold_js.append(jaccard(toks(docs[a]["summary"]),
                                           toks(docs[b]["summary"])))
    gold_js.sort()

    stats = {
        "n_docs": len(docs),
        "n_clone_pairs": n,
        "sum_jaccard_percentiles": {"p10": pct(.10), "p25": pct(.25),
                                    "p50": pct(.50), "p75": pct(.75),
                                    "p90": pct(.90)},
        "sum_jaccard_hist": dict(hist),
        "same_root_cause_rate": round(
            sum(r["same_root_cause"] for r in all_rows) / n, 4),
        "same_env_rate": round(sum(r["same_env"] for r in all_rows) / n, 4),
    }
    if gold_js:
        m = len(gold_js)
        stats["gold_comment_pairs"] = m
        stats["gold_sum_jaccard_p50"] = round(gold_js[m // 2], 3)

    # -- lay mau -------------------------------------------------------------
    rng = random.Random(args.seed)
    sample = rng.sample(all_rows, min(args.n, n))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "clone_audit_sample.csv")
    txt_path = os.path.join(args.out_dir, "clone_audit_sample.txt")
    json_path = os.path.join(args.out_dir, "clone_link_stats.json")

    write_csv(csv_path, sample)
    write_txt(txt_path, sample)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"corpus            : {len(docs)} bug")
    print(f"cap clone (2 dau) : {n}")
    print(f"jaccard tieu de   : p50={stats['sum_jaccard_percentiles']['p50']} "
          f"p90={stats['sum_jaccard_percentiles']['p90']}")
    if gold_js:
        print(f"  (tap gold comment: {stats['gold_comment_pairs']} cap, "
              f"p50={stats['gold_sum_jaccard_p50']})")
    print(f"cung root_cause   : {stats['same_root_cause_rate']:.1%}")
    print(f"cung moi truong   : {stats['same_env_rate']:.1%}")
    print(f"\nlay mau {len(sample)} cap -> soat tay:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
