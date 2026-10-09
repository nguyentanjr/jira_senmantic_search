# -*- coding: utf-8 -*-
"""
Dung tap nhan de fine-tune, tu chinh du lieu Jira (khong can gan tay them).

    python pipeline/build_training_pairs.py

Bon nguon nhan, xep theo do tin cay giam dan:

  1. GOLD      -- cap ky su tu ghi "trung/tuong tu <RO-xxx>" trong comment.
                  Nhan that, do nguoi xac nhan. Chi dung lam eval + train.
  2. HARD NEG  -- cap gan trung tu vung nhung KHAC ban chat ky thuat
                  ("Bat miniapp" vs "Tat miniapp"). Day la nhom mau co gia tri
                  huan luyen cao nhat theo de tai, va sinh ra mien phi.
  3. WEAK POS  -- cap cung (team, root_cause, defect_type) + van ban gan nhau.
                  Dinh nghia "bug tuong tu trong domain" da nam san mot phan
                  trong cac field nay. Nhan yeu -> tach rieng, co the tat.
  4. EASY NEG  -- cap ngau nhien khac team/root_cause. De, dung de can bang.

QUAN TRONG: cap GOLD dung lam eval KHONG duoc xuat hien trong tap train.
Script chia gold theo ty le --gold-eval-ratio va ghi ra hai file rieng.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import load_jsonl, tokenize
from domain import (classify_pair, context_label, parse_tags,
                    _canon_function)

TOKEN_RE = re.compile(r"[0-9a-zà-ỹ_]{2,}")


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall(unicodedata.normalize("NFKC", (s or "").lower())))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def facet_key(d: dict) -> tuple:
    f = d.get("facets", {})
    return (f.get("team", ""), f.get("root_cause", ""), f.get("defect_type", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--gold", default="data/processed/dup_candidates_from_text.jsonl")
    ap.add_argument("--near", default="data/processed/near_duplicates.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--gold-eval-ratio", type=float, default=0.5)
    # Tren nguong nay, hai summary gan nhu trung khop sau khi bo tien to [Module]
    # -> nhieu kha nang la CUNG mot bug (vd bao tren 2 moi truong), khong phai
    # negative. Tach rieng de nguoi soat, khong dua vao tap train.
    ap.add_argument("--hard-neg-max-jaccard", type=float, default=0.90)
    ap.add_argument("--weak-min-jaccard", type=float, default=0.45)
    ap.add_argument("--weak-max-jaccard", type=float, default=0.75)
    ap.add_argument("--weak-per-issue", type=int, default=3)
    ap.add_argument("--easy-neg-per-pos", type=int, default=2)
    ap.add_argument("--max-gold-degree", type=int, default=6,
                    help="bac toi da cua mot issue trong do thi gold "
                         "(chan hub kieu RO-11923: 36 canh)")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    random.seed(a.seed)

    corpus = load_jsonl(a.corpus)
    by = {d["key"]: d for d in corpus}
    print(f"corpus: {len(corpus)} issue")

    text = {d["key"]: (d.get("summary", "") + "\n" + d.get("text_business", ""))
            for d in corpus}
    tk = {k: toks(v) for k, v in text.items()}

    # ---------------- 1. GOLD -------------------------------------------
    gold, all_pairs = [], []
    if os.path.exists(a.gold):
        for p in load_jsonl(a.gold):
            if not (p.get("a") and p.get("b")):
                continue
            all_pairs.append((p["a"], p["b"]))
            if p["a"] in by and p["b"] in by:
                gold.append((p["a"], p["b"], p.get("cue", "")))
    # -- CHAN HUB --------------------------------------------------------
    # Bac p50 cua do thi gold la 1, nhung RO-11923 ("[Perf][Cashout][init]
    # status_code=504") co 36 canh tren tong 326 cap -- gan chac la mot comment
    # liet ke hang loat bug perf lien quan, khong phai 36 cap trung lap that.
    # De nguyen thi 28% tap train xoay quanh mot issue duy nhat va model hoc
    # cai hub do chu khong hoc quan he trung lap.
    #
    # Giu canh co cue trung lap truoc, roi den cue con lai; phan thua ghi ra
    # file rieng de soat chu khong vut im lang.
    deg = Counter()
    for x, y, _ in gold:
        deg[x] += 1
        deg[y] += 1
    over = {k for k, v in deg.items() if v > a.max_gold_degree}
    dropped = []
    if over:
        kept, seen = [], Counter()
        prio = {"trùng": 0, "duplicate": 0}          # cue chac chan hon -> giu truoc
        for x, y, c in sorted(gold, key=lambda p: prio.get(p[2], 1)):
            hot = [k for k in (x, y) if k in over]
            if any(seen[k] >= a.max_gold_degree for k in hot):
                dropped.append({"a": x, "b": y, "cue": c,
                                "reason": "hub_degree_cap"})
                continue
            for k in hot:
                seen[k] += 1
            kept.append((x, y, c))
        print(f"CHAN HUB: {len(over)} issue vuot bac {a.max_gold_degree} "
              f"(cao nhat {deg.most_common(1)[0]}) -> bo {len(dropped)} canh, "
              f"con {len(kept)} cap")
        gold = kept

    # Tach eval/train theo CUM LIEN THONG, tranh ro ri qua cap bac cau.
    #
    # Cum phai dung TOAN BO cap dao duoc, khong chi cap nam trong corpus.
    # Ban dau union-find chi chay tren `gold` (da loc theo corpus) -- va do
    # la ro ri that: RO-7437 va RO-8218 khong noi truc tiep voi nhau nhung
    # cung noi voi RO-200 (issue khong lot vao corpus vi mo ta qua ngan).
    # Loc truoc roi moi union thi hai ben roi vao hai cum "rieng", va bi
    # chia mot ben train mot ben eval -- trong khi ky su coi chung la MOT
    # nhom trung lap. Do lai tren toan do thi: 2 cum bi tach nham nhu vay.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for x, y in all_pairs:
        union(x, y)
    comps = defaultdict(list)
    for x, y, c in gold:
        comps[find(x)].append((x, y, c))
    comp_ids = sorted(comps)
    random.shuffle(comp_ids)
    n_eval = int(len(comp_ids) * a.gold_eval_ratio)
    eval_comps = set(comp_ids[:n_eval])
    gold_eval = [p for c in eval_comps for p in comps[c]]
    gold_train = [p for c in comp_ids[n_eval:] for p in comps[c]]
    print(f"GOLD: {len(gold)} cap, {len(comp_ids)} cum lien thong")
    print(f"  -> eval {len(gold_eval)} cap / train {len(gold_train)} cap"
          f"  (chia theo cum, khong ro ri bac cau)")

    train_keys = {k for x, y, _ in gold_train for k in (x, y)}
    eval_keys = {k for x, y, _ in gold_eval for k in (x, y)}

    # ---------------- 2. HARD NEGATIVE -----------------------------------
    # Phan loai cap gan-trung bang KIEN THUC DOMAIN, khong bang nguong so hoc.
    # [Tag] trong summary mang kenh + tac nhan: USSD/APP, EU/Agent/Super Agent.
    # Cung trieu chung tren hai tac nhan khac nhau = HAI bug, khong phai trung.
    gold_set = {tuple(sorted((x, y))) for x, y, _ in gold}
    hard, suspect = [], []
    ctx_stat = Counter()
    if os.path.exists(a.near):
        for p in load_jsonl(a.near):
            x, y = p.get("a"), p.get("b")
            if not (x in by and y in by) or tuple(sorted((x, y))) in gold_set:
                continue
            j = p.get("jaccard", 0.0)
            sa, sb = by[x].get("summary", ""), by[y].get("summary", "")
            kind, sig = classify_pair(sa, sb, text_sim=j)
            ctx_stat[kind] += 1
            fired = [k for k in ("operator_diff", "number_diff", "negation_diff",
                                 "function_diff", "entity_diff", "order_diff", "actor_diff",
                                 "channel_diff") if sig[k]]
            if kind in ("duplicate", "review"):
                suspect.append({"a": x, "b": y, "jaccard": j,
                                "verdict": kind,
                                "context": context_label(parse_tags(sa)),
                                "signals": fired,
                                "signal_detail": {k: sig[k] for k in fired},
                                "pattern": " | ".join(fired) or "(khong tin hieu)",
                                "summary_a": sa[:130], "summary_b": sb[:130]})
            elif kind == "different":
                hard.append({"anchor": x, "negative": y, "jaccard": j,
                             "ctx_a": context_label(parse_tags(sa)),
                             "ctx_b": context_label(parse_tags(sb)),
                             "signals": fired,
                             "source": "strong_context_diff"})
    n_dup = sum(1 for s_ in suspect if s_["verdict"] == "duplicate")
    n_rev = sum(1 for s_ in suspect if s_["verdict"] == "review")
    print(f"HARD NEG: {len(hard)} cap -- tin hieu MANH (khac tac nhan/kenh)")
    print(f"CAN SOAT: {len(suspect)} cap")
    print(f"   - {n_dup} khong co tin hieu phan biet nao (kha nang trung cao)")
    print(f"   - {n_rev} co tin hieu yeu -> KHONG tu gan nhan, cho nguoi quyet")

    # ---------------- 3. WEAK POSITIVE (metadata) ------------------------
    groups = defaultdict(list)
    for d in corpus:
        fk = facet_key(d)
        if all(fk):                       # chi lay issue day du ca 3 facet
            groups[fk].append(d["key"])
    weak = []
    n_grp = 0
    for fk, keys in groups.items():
        if len(keys) < 2 or len(keys) > 400:
            continue
        n_grp += 1
        for i, x in enumerate(keys):
            cand = []
            for y in keys:
                if y == x:
                    continue
                j = jaccard(tk[x], tk[y])
                if a.weak_min_jaccard <= j <= a.weak_max_jaccard:
                    cand.append((j, y))
            cand.sort(reverse=True)
            for j, y in cand[: a.weak_per_issue]:
                pair = tuple(sorted((x, y)))
                if pair in gold_set:
                    continue
                weak.append({"a": pair[0], "b": pair[1], "jaccard": round(j, 3),
                             "team": fk[0], "root_cause": fk[1],
                             "defect_type": fk[2]})
    seen = set()
    weak = [w for w in weak
            if not ((w["a"], w["b"]) in seen or seen.add((w["a"], w["b"])))]
    print(f"WEAK POS: {len(weak)} cap tu {n_grp} nhom (team+root_cause+defect_type)"
          f", jaccard {a.weak_min_jaccard}-{a.weak_max_jaccard}")

    # ---------------- 4. EASY NEGATIVE -----------------------------------
    all_keys = list(by)
    easy = []
    n_easy = a.easy_neg_per_pos * (len(gold_train) + len(weak))
    guard = 0
    while len(easy) < n_easy and guard < n_easy * 20:
        guard += 1
        x, y = random.sample(all_keys, 2)
        if facet_key(by[x]) == facet_key(by[y]):
            continue
        if jaccard(tk[x], tk[y]) > 0.25:
            continue
        if tuple(sorted((x, y))) in gold_set:
            continue
        easy.append({"anchor": x, "negative": y, "source": "random"})
    print(f"EASY NEG: {len(easy)} cap")

    # ---------------- ghi file -------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)

    def dump(name, rows):
        p = os.path.join(a.out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {p}  ({len(rows)})")

    print("\nghi file:")
    dump("gold_eval_pairs.jsonl",
         [{"a": x, "b": y, "cue": c, "split": "eval"} for x, y, c in gold_eval])
    dump("gold_train_pairs.jsonl",
         [{"a": x, "b": y, "cue": c, "split": "train"} for x, y, c in gold_train])
    if dropped:
        dump("gold_dropped_hub_pairs.jsonl", dropped)
    dump("hard_negatives.jsonl", hard)
    suspect.sort(key=lambda r: (r["verdict"] != "duplicate",
                                r["pattern"], r["a"]))
    dump("suspected_duplicates_for_review.jsonl", suspect)
    dump("weak_positives.jsonl", weak)
    dump("easy_negatives.jsonl", easy)

    # tap train hop nhat cho bi-encoder (MultipleNegativesRankingLoss)
    mnrl = []
    for x, y, _ in gold_train:
        mnrl.append({"anchor": text[x], "positive": text[y],
                     "weight": 1.0, "source": "gold"})
    for w in weak:
        mnrl.append({"anchor": text[w["a"]], "positive": text[w["b"]],
                     "weight": 0.3, "source": "weak"})
    random.shuffle(mnrl)
    dump("train_bi_encoder.jsonl", mnrl)

    # tap train cho cross-encoder (nhan nhi phan)
    ce = []
    for x, y, _ in gold_train:
        ce.append({"query": text[x], "doc": text[y], "label": 1.0, "source": "gold"})
    for w in weak:
        ce.append({"query": text[w["a"]], "doc": text[w["b"]], "label": 0.7,
                   "source": "weak"})
    for h in hard:
        if h["anchor"] in train_keys or h["negative"] in train_keys \
                or (h["anchor"] not in eval_keys and h["negative"] not in eval_keys):
            ce.append({"query": text[h["anchor"]], "doc": text[h["negative"]],
                       "label": 0.0, "source": "hard_neg"})
    for e in easy:
        ce.append({"query": text[e["anchor"]], "doc": text[e["negative"]],
                   "label": 0.0, "source": "easy_neg"})
    random.shuffle(ce)
    dump("train_cross_encoder.jsonl", ce)

    print("\n-- TOM TAT --")
    print(f"  bi-encoder : {len(mnrl)} cap duong "
          f"({sum(1 for m in mnrl if m['source']=='gold')} gold / "
          f"{sum(1 for m in mnrl if m['source']=='weak')} weak)")
    c = Counter(x["source"] for x in ce)
    print(f"  cross-enc  : {len(ce)} mau  {dict(c)}")
    print(f"  eval giu rieng: {len(gold_eval)} cap gold, "
          f"{len(eval_keys)} issue KHONG xuat hien trong train")


if __name__ == "__main__":
    main()
