# -*- coding: utf-8 -*-
"""Phase do luong + kiem tra ro ri nhan.

Bo danh gia: 199 cap ky su TU GHI trong comment ("trung bug RO-10049",
"tuong tu RO-10025"). Day la nhan that, khong phai nhan sinh ra tu do do
tuong dong -- rat quan trong: neu lay chinh Jaccard lam nhan thi bai do chi
do lai chinh no.

Moi issue trong tap gold sinh MOT truy van; dap an la cac issue duoc noi
truc tiep voi no. Khong lay bao dong bac cau (A~B, B~C khong suy ra A~C):
bac cau la gia dinh, khong phai nhan.

Chi so: Recall@K, Precision@K, Hit@K, MRR@K, nDCG@K.

KIEM TRA RO RI (chay truoc moi benchmark, xem audit_leakage):
  L1  truy van co bi tra ve chinh no khong
  L2  van ban duoc index co chua ma issue cua doi phuong khong -- nhan gold
      duoc dao tu chinh cau nhac do, de nguyen la dua mot phan nhan vao input
  L3  tap train va tap eval co chung issue / chung cum lien thong khong
  L4  co dac trung nao duoc tinh tu nhan khong
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from ._compat import load_jsonl
from .features import FEATURE_NAMES

_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")


# --------------------------------------------------------------- gold set
#
# CUE KHONG PHAI DEU LA "TRUNG LAP". Do tren 326 cap dao duoc:
#
#     tuong tu   181 (55,5%)      trung        109 (33,4%)
#     giong bug   19              duplicate     12
#     giong loi    1              da log bug     3 / da co bug 1
#
# Tren tap eval, cue trung lap chi chiem 29/84 = 34,5%. Gop tat ca lai roi
# goi la "duplicate" la sai ten: hai phan ba tap do la "tuong tu". Mot he
# thong tra ve issue tuong tu bi cham la miss neu nhan la trung lap, va
# nguoc lai -- nen phai bao cao tach ra.
#
# `unverified` (da log bug / da co bug) KHONG duoc coi la positive: cau do
# co the tro toi mot bug khac han, chi noi rang van de da duoc ghi nhan o
# dau do. Chi 5 cap, nhung de nguyen trong positive la sai nguyen tac.
CUE_CLASS = {
    "trùng": "duplicate", "duplicate": "duplicate",
    "tương tự": "similar", "giống bug": "similar", "giống lỗi": "similar",
    "đã log bug": "unverified", "đã có bug": "unverified",
}


def cue_class(cue: str) -> str:
    return CUE_CLASS.get((cue or "").strip().lower(), "unknown")


def build_queries_by_cue(pairs: list[dict], valid_keys: set[str]
                         ) -> dict[str, list[dict]]:
    """-> {lop cue: truy van}. `all` = moi cap tru `unverified`.

    Truy van cua lop X chi giu lang gieng noi bang cue thuoc lop X. Issue nao
    khong co lang gieng nao trong lop do thi khong sinh truy van cho lop do --
    neu khong se dem la miss mot thu khong ton tai trong lop.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_class[cue_class(p.get("cue", ""))].append(p)

    out = {}
    for cls in ("duplicate", "similar", "unverified", "unknown"):
        if by_class.get(cls):
            out[cls] = build_queries(by_class[cls], valid_keys)
    out["all"] = build_queries(
        [p for p in pairs if cue_class(p.get("cue", "")) != "unverified"],
        valid_keys)
    return out


def build_queries(pairs: list[dict], valid_keys: set[str]) -> list[dict]:
    """Cap -> truy van. Moi issue mot truy van, dap an la lang gieng truc tiep."""
    nb: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        a, b = p.get("a"), p.get("b")
        if a in valid_keys and b in valid_keys and a != b:
            nb[a].add(b)
            nb[b].add(a)
    return [{"key": k, "relevant": sorted(v)} for k, v in sorted(nb.items())]


def clusters(pairs: list[dict]) -> dict[str, int]:
    """Union-find -> cum lien thong (dung cho kiem tra ro ri theo cum)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, b = p.get("a"), p.get("b")
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = {}
    out = {}
    for k in parent:
        r = find(k)
        out[k] = roots.setdefault(r, len(roots))
    return out


# ----------------------------------------------------------------- metrics
def evaluate(pipeline, queries: list[dict], ks=(1, 5, 10),
             progress_every: int = 0) -> dict:
    maxk = max(ks)
    agg = defaultdict(float)
    n = 0
    for qi, q in enumerate(queries):
        if q["key"] not in pipeline.key_to_idx:
            continue
        rel = {r for r in q["relevant"] if r in pipeline.key_to_idx}
        if not rel:
            continue
        ranked = pipeline.rank_keys(q["key"], maxk)
        n += 1
        hits = [1.0 if k in rel else 0.0 for k in ranked[:maxk]]
        for k in ks:
            h = hits[:k]
            got = sum(h)
            agg[f"recall@{k}"] += got / len(rel)
            agg[f"hit@{k}"] += 1.0 if got else 0.0
            agg[f"precision@{k}"] += got / k
            dcg = sum(v / math.log2(r + 2.0) for r, v in enumerate(h))
            idcg = sum(1.0 / math.log2(r + 2.0) for r in range(min(len(rel), k)))
            agg[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0
        rr = 0.0
        for r, v in enumerate(hits):
            if v:
                rr = 1.0 / (r + 1)
                break
        agg[f"mrr@{maxk}"] += rr
        if progress_every and (qi + 1) % progress_every == 0:
            print(f"    ... {qi+1}/{len(queries)}", flush=True)
    out = {k: round(v / n, 4) for k, v in agg.items()} if n else {}
    out["queries"] = n
    return out


def evaluate_by_group(pipeline, queries, docs_by_key: dict, ks=(1, 5, 10)) -> dict:
    """Tach nhom lexical-advantage / semantic-advantage (cam ket cua de tai)."""
    g = {"lexical (co ma loi/API/class)": [], "semantic (thuan nghiep vu)": []}
    for q in queries:
        d = docs_by_key.get(q["key"], {})
        key = ("lexical (co ma loi/API/class)" if d.get("lexical_group")
               else "semantic (thuan nghiep vu)")
        g[key].append(q)
    return {k: evaluate(pipeline, v, ks=ks) for k, v in g.items() if v}


# --------------------------------------------------- chat luong QUYET DINH
def decision_metrics(pipeline, queries: list[dict], top_k: int = 10) -> dict:
    """Do chat luong NHAN, khong phai chat luong xep hang.

    Vi sao can mot ham rieng: Recall@K / MRR / nDCG duoc tinh tu THU TU, ma
    nguong khong he doi thu tu -- doi t_duplicate tu 0.5 len 0.95 thi ca bon
    chi so do khong nhuc nhich mot chu so nao. Khong co ham nay thi khong co
    cach nao biet nguong dat dung hay sai.

    Hai con so tra ve, va can doc dung:

      dup_recall      trong so cap gold DA duoc lay ve trong top-K, bao nhieu
                      phan tram duoc gan 'duplicate'. Day la recall THAT.

      dup_precision   trong so cap duoc gan 'duplicate', bao nhieu phan tram
                      la gold. Day la CAN DUOI, khong phai precision that:
                      tap gold chi co 138 cap dao tu comment nen chac chan
                      con nhieu cap trung lap chua ai ghi. Mot cap bi tinh
                      la sai o day co the thuc su dung.
                      -> dung de SO SANH hai cau hinh, khong dung de bao cao
                         nhu mot ty le tuyet doi.
    """
    from collections import Counter
    lab = Counter()
    n_gold_retrieved = n_gold_dup = 0
    n_dup = n_dup_gold = 0
    n_gold_shown_as = Counter()

    for q in queries:
        if q["key"] not in pipeline.key_to_idx:
            continue
        rel = {r for r in q["relevant"] if r in pipeline.key_to_idx}
        if not rel:
            continue
        res = pipeline.search_by_key(q["key"], top_k=top_k)
        for r in res["results"]:
            lab[r["label"]] += 1
            is_gold = r["bugId"] in rel
            if r["label"] == "duplicate":
                n_dup += 1
                n_dup_gold += is_gold
            if is_gold:
                n_gold_retrieved += 1
                n_gold_shown_as[r["label"]] += 1
                n_gold_dup += r["label"] == "duplicate"

    rec = n_gold_dup / n_gold_retrieved if n_gold_retrieved else 0.0
    pre = n_dup_gold / n_dup if n_dup else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    return {
        "dup_recall": round(rec, 4),
        "dup_precision_lower_bound": round(pre, 4),
        "dup_f1_lower_bound": round(f1, 4),
        "n_labeled_duplicate": n_dup,
        "n_gold_retrieved": n_gold_retrieved,
        "gold_label_mix": dict(n_gold_shown_as),
        "label_mix": dict(lab),
    }


# ------------------------------------------------------------- ro ri nhan
def audit_leakage(pipeline, queries: list[dict], gold_pairs: list[dict],
                  train_path: str = "data/processed/gold_train_pairs.jsonl",
                  eval_path: str = "data/processed/gold_eval_pairs.jsonl",
                  sample: int = 40) -> dict:
    import os

    out: dict = {}

    # L1 -- truy van tra ve chinh no?
    self_hits = 0
    for q in queries[:sample]:
        if q["key"] not in pipeline.key_to_idx:
            continue
        if q["key"] in pipeline.rank_keys(q["key"], 10):
            self_hits += 1
    out["L1_self_in_results"] = self_hits

    # L2 -- van ban index co nhac ma issue cua doi phuong?
    masked = pipeline.cfg.preprocess.mask_issue_refs
    leak = 0
    by = {d["key"]: d for d in pipeline.docs}
    for p in gold_pairs:
        a, b = p.get("a"), p.get("b")
        if a in by and b in by:
            for x, y in ((a, b), (b, a)):
                if y in _KEY_RE.findall(by[x].get("text_bm25", "")):
                    leak += 1
    out["L2_cross_reference_in_indexed_text"] = leak
    out["L2_masking_enabled"] = masked
    out["L2_status"] = "OK (da che)" if masked or leak == 0 else "RUI RO"

    # L3 -- train / eval chung issue hoac chung cum?
    if os.path.exists(train_path) and os.path.exists(eval_path):
        tr, ev = load_jsonl(train_path), load_jsonl(eval_path)
        ktr = {k for p in tr for k in (p["a"], p["b"])}
        kev = {k for p in ev for k in (p["a"], p["b"])}
        cl = clusters(gold_pairs)
        ctr = {cl.get(k) for k in ktr}
        cev = {cl.get(k) for k in kev}
        out["L3_shared_issue_keys"] = len(ktr & kev)
        out["L3_shared_clusters"] = len((ctr & cev) - {None})
        out["L3_status"] = ("OK" if not (ktr & kev) and not ((ctr & cev) - {None})
                            else "RUI RO")
    else:
        out["L3_status"] = "khong co file train/eval de doi chieu"

    # L4 -- dac trung nao dung den nhan?
    suspicious = [f for f in FEATURE_NAMES
                  if any(w in f for w in ("label", "gold", "dup_of", "mention",
                                          "cue", "link"))]
    out["L4_label_derived_features"] = suspicious
    out["L4_status"] = "OK" if not suspicious else "RUI RO"
    return out
