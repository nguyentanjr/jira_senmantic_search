# -*- coding: utf-8 -*-
"""
Dung BO DANH GIA PARAPHRASE (bo danh gia 1 trong de cuong).

    python pipeline/build_paraphrase_eval.py

VAN DE CAN GIAI
Tap gold hien tai (212 cap) dao tu comment ky su tu ghi "trung/tuong tu RO-xxx".
Ky su chi ghi duoc cau do khi HO DA TIM RA cap trung -- bang chinh keyword
search cua Jira. Nen tap gold gan nhu chi chua nhung cap ma keyword search DA
giai duoc. Do la ly do semantic do THUA keyword tren tap do (0.5072 vs 0.5435):
khong phai model kem, ma tap do da loai san cac ca ngu nghia.

Them data khong cuu duoc: corpus 4.388 -> 10.925 nhung mat do nhan tut tu
31,5 xuong 19,4 cap/1000 issue. Nguon nhan nay khong scale.

CACH LAM
Truy van la phan `Actual Result` cua chinh issue -- van ban do NGUOI VIET, mo
ta trieu chung. Tai lieu duoc index sau khi da BO phan do di. Dap an la chinh
issue goc.

    truy van  :  sections["actual"]
    tai lieu  :  summary + steps/precondition + expected   (da tru actual)
    dap an    :  chinh issue do

Quan he giua hai phia dung la quan he paraphrase can do: cung mot defect, hai
cach dien dat khac nhau, deu la van ban that. Vi du RO-10003:

    truy van : "Khong hien thi thong tin giao dich vua Reject len list danh
                sach Approve"
    summary  : "[EU APP][Transaction Reversal- Reject] Reject thanh cong, quay
                lai man hinh hien thi list danh sach giao dich, khong hien thi
                giao dich vua Reject"

Uu diem so voi sinh paraphrase bang luat thay tu dong nghia: van ban that,
khong nhan tao, va sinh duoc hang nghin truy van ma khong can gan nhan tay.
Bo nay SCALE tuyen tinh theo corpus -- khac han nguon comment.

GIOI HAN PHAI GHI RO KHI BAO CAO
Truy van va tai lieu den tu cung mot issue, nen bo nay do KHA NANG BAT KHE
LECH TU VUNG cua retriever, khong phai do truc tiep bai toan phat hien trung
lap. No BO SUNG cho tap gold, khong thay the. Bao cao ca hai.

CHONG RO RI (bat buoc, neu khong bo nay vo nghia)
  1. `actual` bi tru khoi CA text_dense LAN text_bm25 (biz chua actual).
  2. `sections["actual"]` bi xoa -- StructuredExtractor doc truong nay.
  3. `signals` bi loc: gia tri nao chi xuat hien trong actual thi bo di,
     neu khong ma loi se lam cau noi tat.
  4. Ma issue (RO-xxxx) bi che o ca hai phia.

PHAN LAT CAT
Moi truy van duoc gan `jac_bin` = do trung tu vung voi tai lieu dap an.
Lat cat `<0.2` la noi keyword search bat luc -- do la cho phai chung minh
retrieval ngu nghia thang. Xem eval_paraphrase.py.

Dau ra:
  data/processed/corpus_holdout.jsonl      corpus da tru actual (haystack)
  data/processed/paraphrase_queries.jsonl  truy van + dap an + lat cat
  data/processed/paraphrase_build_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

ISSUE_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")
TOKEN_RE = re.compile(r"[0-9a-zà-ỹ_]{2,}")


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall(unicodedata.normalize("NFKC", (s or "").lower())))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mask_issue_refs(s: str) -> str:
    return ISSUE_RE.sub(" <ISSUE> ", s or "")


def subtract(text: str, part: str) -> str:
    """Tru chuoi `part` khoi `text`. Tru theo dong de chiu duoc lech khoang trang."""
    if not part or not text:
        return text or ""
    out = text.replace(part, " ")
    if out != text:
        return re.sub(r"\n{3,}", "\n\n", out).strip()
    # khong khop nguyen khoi -> tru theo tung dong khong rong
    drop = {ln.strip() for ln in part.splitlines() if len(ln.strip()) >= 12}
    if not drop:
        return text
    keep = [ln for ln in text.splitlines() if ln.strip() not in drop]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def build_holdout(d: dict) -> dict:
    """Ban sao cua doc, da tru phan `actual` khoi moi truong van ban."""
    actual = (d.get("sections") or {}).get("actual", "")
    r = dict(d)
    sections = {k: v for k, v in (d.get("sections") or {}).items() if k != "actual"}
    r["sections"] = sections

    # Phai tru khoi CA HAI truong. `split_business_technical` cat description
    # theo tieu chi ky thuat, nen phan `actual` bi xe doi: cau van xuoi roi vao
    # text_business, con SQL / JSON / trace-id / log roi vao text_technical.
    # Chi tru text_business thi 44,5% truy van van con token song sot ben
    # text_technical (p90: 71% token truy van) -> BM25 di duong tat va moi so
    # do duoc deu vo nghia.
    biz_ho = subtract(d.get("text_business", ""), actual)
    tech_ho = subtract(d.get("text_technical", ""), actual)
    r["text_business"] = biz_ho
    r["text_technical"] = tech_ho

    summ = d.get("summary", "")
    tech = tech_ho
    labels = " ".join(d.get("labels") or [])

    # signals: bo gia tri chi ton tai trong phan da tru
    visible = " ".join([summ, biz_ho, tech, labels]).lower()
    signals = {}
    for k, vs in (d.get("signals") or {}).items():
        kept = [v for v in vs if str(v).lower() in visible]
        if kept:
            signals[k] = kept
    r["signals"] = signals

    r["text_dense"] = mask_issue_refs("\n".join(
        x for x in [summ, biz_ho, sections.get("expected", "")] if x).strip() or summ)
    r["text_bm25"] = mask_issue_refs("\n".join(x for x in [
        summ, biz_ho, tech,
        " ".join(v for vs in signals.values() for v in vs),
        labels,
    ] if x).strip())
    return r


def bin_of(x: float) -> str:
    for lo in (0.6, 0.4, 0.2):
        if x >= lo:
            return f">={lo}"
    return "<0.2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-query-chars", type=int, default=80,
                    help="do dai toi thieu cua phan actual de lam truy van")
    ap.add_argument("--min-doc-chars", type=int, default=80,
                    help="do dai toi thieu cua tai lieu SAU khi tru actual")
    ap.add_argument("--max-query-chars", type=int, default=1200)
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        sys.exit(f"khong thay corpus: {args.corpus}")

    docs = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    holdout = [build_holdout(d) for d in docs]

    queries = []
    skip = Counter()
    for d, h in zip(docs, holdout):
        actual = (d.get("sections") or {}).get("actual", "")
        if len(actual) < args.min_query_chars:
            skip["actual qua ngan / khong co"] += 1
            continue
        doc_text = h.get("text_bm25", "")
        if len(doc_text) < args.min_doc_chars:
            skip["tai lieu rong sau khi tru"] += 1
            continue
        q = mask_issue_refs(actual)[: args.max_query_chars].strip()
        qt, dt = toks(q), toks(doc_text)
        j = jaccard(qt, dt)
        queries.append({
            "key": d["key"],
            "query": q,
            "relevant": [d["key"]],
            "jaccard": round(j, 4),
            "jac_bin": bin_of(j),
            "lexical_group": bool(d.get("lexical_group")),
            "n_query_tok": len(qt),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    ho_path = os.path.join(args.out_dir, "corpus_holdout.jsonl")
    q_path = os.path.join(args.out_dir, "paraphrase_queries.jsonl")
    s_path = os.path.join(args.out_dir, "paraphrase_build_stats.json")

    with open(ho_path, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -- KIEM RO RI TON DU ------------------------------------------------
    # Token cua truy van con song sot trong tai lieu la binh thuong (tu chung,
    # tu nghiep vu). Cai phai gan bang 0 la token HIEM: neu mot trace-id hay
    # ma giao dich cua truy van con nam trong tai lieu thi BM25 tim ra ngay
    # ma khong can hieu gi ca. Do bang idf tho tren chinh corpus.
    df = Counter()
    for h in holdout:
        for t in set(toks(h.get("text_bm25", ""))):
            df[t] += 1
    rare_cut = max(2, len(holdout) // 500)      # xuat hien o <= 0,2% tai lieu
    ho_by_key = {h["key"]: h for h in holdout}
    #
    # Phan con lai KHONG tru duoc: `summary` la mot phan hop le cua tai lieu,
    # va neu mot ma loi xuat hien o ca summary lan actual thi do la thuoc tinh
    # that cua issue, khong phai loi dung bo du lieu. Nen thay vi duoi cho ve 0,
    # bien no thanh LAT CAT KIEM SOAT: nhom `none` la nhung truy van khong chia
    # se BAT KY token hiem nao voi tai lieu -- o do BM25 buoc phai dua vao tu
    # vung thuong, va do moi la phep thu sach cho retrieval ngu nghia.
    n_leak, leak_frac = 0, []
    for q in queries:
        qt = toks(q["query"])
        rare = {t for t in qt if df.get(t, 0) <= rare_cut}
        doc = toks(ho_by_key[q["key"]].get("text_bm25", ""))
        hit = rare & doc
        f = len(hit) / len(rare) if rare else 0.0
        if hit:
            n_leak += 1
        leak_frac.append(f)
        q["rare_overlap"] = round(f, 4)
        q["rare_bin"] = "none" if not hit else "some"

    # ghi truy van SAU khi da gan rare_bin / rare_overlap
    with open(q_path, "w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    js = sorted(q["jaccard"] for q in queries)
    n = len(js) or 1
    stats = {
        "n_corpus": len(docs),
        "n_queries": len(queries),
        "skipped": dict(skip),
        "jaccard_percentiles": {
            f"p{p}": round(js[min(int(p / 100 * n), n - 1)], 3)
            for p in (10, 25, 50, 75, 90)} if js else {},
        "jac_bin": dict(Counter(q["jac_bin"] for q in queries)),
        "lexical_group_rate": round(
            sum(q["lexical_group"] for q in queries) / n, 4),
        "residual_rare_token_leak": {
            "rare_df_cutoff": rare_cut,
            "queries_with_leak": n_leak,
            "queries_with_leak_rate": round(n_leak / n, 4),
            "mean_rare_token_leak": round(sum(leak_frac) / n, 4),
        },
    }
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"corpus            : {len(docs)}")
    print(f"truy van sinh duoc: {len(queries)}")
    for k, v in skip.most_common():
        print(f"  bo qua: {k}: {v}")
    if js:
        print(f"jaccard truy van/tai lieu: "
              f"p25={stats['jaccard_percentiles']['p25']} "
              f"p50={stats['jaccard_percentiles']['p50']} "
              f"p75={stats['jaccard_percentiles']['p75']}")
    rl = stats["residual_rare_token_leak"]
    print(f"ro ri ton du (token hiem, df<={rare_cut}): "
          f"{rl['queries_with_leak']} truy van ({rl['queries_with_leak_rate']:.1%}), "
          f"trung binh {rl['mean_rare_token_leak']:.1%} token hiem")
    print("phan lat cat:")
    for k in ("<0.2", ">=0.2", ">=0.4", ">=0.6"):
        v = stats["jac_bin"].get(k, 0)
        print(f"  {k:>6}: {v:>5}  ({v / n:.1%})")
    print(f"\n-> {ho_path}\n-> {q_path}\n-> {s_path}")


if __name__ == "__main__":
    main()
