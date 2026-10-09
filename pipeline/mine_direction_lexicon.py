# -*- coding: utf-8 -*-
"""Khai thac tu dien chi chieu tu nhan LLM, de dung cho PHIA TRUY VAN.

Bo loc nguoc chieu chi chay khi biet chieu cua truy van. Tai lieu thi da co
nhan LLM (31,5% corpus), nhung truy van la van ban tu do go tai cho - khong
the goi LLM moi lan tim kiem (them do tre va ton han muc). Vi vay: hoc cum tu
dac trung cua tung chieu TU chinh 3.443 nhan LLM, roi so khop cum tu do voi
truy van. Khong ton gi luc chay.

Cach chon cum tu: n-gram 2-5 tu xuat hien nhieu o mot ben va rat it o ben kia
cua CUNG mot truc. Diem = so lan xuat hien ben nay / (so lan ben kia + 1).
Luat viet tay ban dau van duoc giu lam nen; file nay chi BO SUNG.

Bai hoc da tra gia: cum "co quyen" nam trong "khong co quyen" nen tung gan
nguoc chieu hang loat. Vi vay cum khai thac ra deu di qua ``contains()`` cua
build_direction_pairs, ham nay bo qua lan xuat hien nam sau tu phu dinh.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_direction_pairs import AXES, judging_text
from direction_filter import llm_labels

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Cum tu chi chieu BAT BUOC chua mot trong cac tu nay. Khong co rang buoc nay,
# n-gram hoc ra cum BOI CANH chu khong phai cum chi chieu: "sai pin" tung duoc
# hoc cho chieu "thua OTP" (vi cac issue ve sai PIN hay thuoc chieu do), roi
# gan nham cho truy van "sai PIN nhung KHONG bi hoi OTP" - nguoc han.
POLARITY = {"không", "khong", "chưa", "chua", "vẫn", "van", "thừa", "thua",
            "thiếu", "thieu", "double", "lặp", "lap", "trùng", "trung",
            "luôn", "luon", "ko", "mất", "mat"}


def ngrams(text: str, lo: int, hi: int) -> Counter:
    words = [m.group().casefold() for m in WORD.finditer(text)]
    out = Counter()
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            out[" ".join(words[i:i + n])] += 1
    return out


def mine(docs: dict[str, dict], labels: dict[str, set], args) -> dict:
    per_side: dict[tuple[str, str], Counter] = defaultdict(Counter)
    docs_per_side: Counter = Counter()
    background: Counter = Counter()
    for key, doc in docs.items():
        grams = set(ngrams(judging_text(doc)[:600], args.min_n, args.max_n))
        background.update(grams)
        label = labels.get(key)
        if not label:
            continue
        axis, side = next(iter(label))
        # Chi lay TIEU DE + actual, dung nguon ma nguoi doc dung de ket luan.
        per_side[(axis, side)].update(grams)
        docs_per_side[(axis, side)] += 1
    n_bg = max(1, len(docs))

    lexicon: dict[str, dict[str, list[str]]] = {}
    for axis, sides in AXES.items():
        names = list(sides)
        for side in names:
            mine_c = per_side.get((axis, side), Counter())
            others = Counter()
            for other in names:
                if other != side:
                    others.update(per_side.get((axis, other), Counter()))
            n_side = max(1, docs_per_side.get((axis, side), 0))
            picked = []
            for phrase, count in mine_c.most_common():
                if count < args.min_count:
                    break
                if not (set(phrase.split()) & POLARITY):
                    continue          # cum boi canh, khong phai cum chi chieu
                # So voi NEN toan corpus, khong chi voi ben doi dien: ben doi
                # dien thuong it tai lieu hon nhieu nen cum chung chung
                # ("thanh cong", "giao dich") van lot qua phep so ty le do.
                lift = (count / n_side) / max(1e-9, background[phrase] / n_bg)
                if lift < args.min_ratio:
                    continue
                if others[phrase] / max(1, sum(
                        docs_per_side.get((axis, o), 0)
                        for o in names if o != side)) > args.max_other_rate:
                    continue          # xuat hien nhieu o chinh chieu nguoc lai
                if any(p in phrase for p in sides[side]):
                    continue          # da co trong tu dien viet tay
                picked.append((phrase, count, round(lift, 1)))
            picked.sort(key=lambda x: -x[2])
            lexicon.setdefault(axis, {})[side] = [p for p, _, _ in picked[:args.top]]
    return {"docs_per_side": {f"{a}/{s}": n for (a, s), n in docs_per_side.items()},
            "lexicon": lexicon}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--labels", default="data/processed/direction_labels_llm.jsonl")
    ap.add_argument("--out", default="data/processed/direction_lexicon_mined.json")
    ap.add_argument("--min-n", type=int, default=2)
    ap.add_argument("--max-n", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5,
                    help="so tai lieu toi thieu chua cum tu")
    ap.add_argument("--min-ratio", type=float, default=4.0,
                    help="ty le xuat hien o chieu nay so voi nen toan corpus")
    ap.add_argument("--max-other-rate", type=float, default=0.05,
                    help="ty le toi da duoc phep xuat hien o chieu nguoc lai")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        docs = {}
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs[str(d["key"]).upper()] = d
    result = mine(docs, llm_labels(args.labels), args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    for axis, sides in result["lexicon"].items():
        for side, phrases in sides.items():
            n = result["docs_per_side"].get(f"{axis}/{side}", 0)
            print(f"{axis}/{side} ({n} issue): {len(phrases)} cum moi")
            for p in phrases[:6]:
                print(f"    {p}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
