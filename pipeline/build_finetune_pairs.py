# -*- coding: utf-8 -*-
"""
Dung tap cap FINE-TUNE SACH -- khong issue nao cua bo do lot vao cap train.

    python pipeline/build_finetune_pairs.py                                   # mac dinh
    python pipeline/build_finetune_pairs.py --exclude-paraphrase --suffix _noparaphrase

VI SAO KHONG DUNG train_bi_encoder.jsonl / train_cross_encoder.jsonl CU
Soat ngay 2026-09-11 bang so khop van ban voi corpus:
  1. 46 issue cua similar_gold_eval nam trong gold_train_pairs.jsonl --
     build_training_pairs.py va build_similar_gold.py chia train/eval DOC LAP
     nhau, nen cum lien thong cua bo nay roi vao train cua bo kia.
  2. 2/40 truy van pooling nam trong cap duong (RO-11708, RO-20156), va 3 cap
     pooling nam nguyen trong tap train.
  3. 661 dap an paraphrase nam trong cap duong, ma text_business chua phan
     `actual` -- chinh la truy van cua bo paraphrase.
File cu chi luu van ban, khong luu ma issue, nen khong soat duoc. File moi luu
a_key / b_key cho moi dong.

CHINH SACH MAC DINH
  gold   = similar_gold_train.jsonl: chia CUNG dot voi similar_gold_eval nen
           khong chung cum (do: 0 issue chung).
  loai   = moi issue cua similar_gold_eval + moi truy van pooling. Khong cap
           pooling nao duoc hoc. Tai lieu pooling xuat hien trong cap KHAC thi
           giu: do la tiep xuc van ban corpus, khong phai cap dang duoc do.
  paraphrase: GIU issue, nhung ghi ra danh sach truy van SACH (dap an khong nam
           trong cap duong nao) -> bao cao paraphrase sau fine-tune tren tap do.
           Do tren du lieu hien tai: 3.175/3.829 sach. Loai han paraphrase thi
           tap train con mot nua (gold 73, weak 775) -- de o --exclude-paraphrase
           de kiem tra do vung.

KHONG ghi de file cua build_training_pairs.py: split gold cua pipeline dup phai
giu nguyen, neu khong cac ket qua dup cu khong con so duoc.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

#: grade cua similar gold -> nhan/trong so. Grade 1 = "dang doc" nen thap hon.
GOLD_LABEL = {2: 1.0, 1: 0.7}
#: giu nhu build_training_pairs.py de so sanh duoc voi tap cu
WEAK_BI_WEIGHT = 0.3
WEAK_CE_LABEL = 0.7


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(gold: list[dict], weak: list[dict], hard: list[dict], easy: list[dict],
          text: dict[str, str], exclude: set[str], seed: int = 13):
    """-> (bi_rows, ce_rows, stat). Cap co mot dau bi loai hoac khong co van ban
    thi bo. Cap duong trung nhau (gold va weak) chi giu mot lan, uu tien gold."""
    stat = Counter()
    seen: set[frozenset] = set()
    bi, ce = [], []

    def usable(x: str, y: str, src: str) -> bool:
        if x not in text or y not in text:
            stat[f"{src}_bo_khong_co_van_ban"] += 1
            return False
        if x in exclude or y in exclude:
            stat[f"{src}_bo_do_loai"] += 1
            return False
        return True

    for r in gold:
        x, y = r["a"], r["b"]
        label = GOLD_LABEL.get(int(r.get("grade", 0)))
        if label is None:
            stat["gold_bo_grade_la"] += 1
            continue
        if not usable(x, y, "gold") or frozenset((x, y)) in seen:
            continue
        seen.add(frozenset((x, y)))
        src = f"gold{int(r['grade'])}"
        bi.append({"anchor": text[x], "positive": text[y], "a_key": x, "b_key": y,
                   "weight": label, "source": src})
        ce.append({"query": text[x], "doc": text[y], "a_key": x, "b_key": y,
                   "label": label, "source": src})

    for r in weak:
        x, y = r["a"], r["b"]
        if not usable(x, y, "weak") or frozenset((x, y)) in seen:
            continue
        seen.add(frozenset((x, y)))
        bi.append({"anchor": text[x], "positive": text[y], "a_key": x, "b_key": y,
                   "weight": WEAK_BI_WEIGHT, "source": "weak"})
        ce.append({"query": text[x], "doc": text[y], "a_key": x, "b_key": y,
                   "label": WEAK_CE_LABEL, "source": "weak"})

    for src, rows in (("hard_neg", hard), ("easy_neg", easy)):
        for r in rows:
            x, y = r["anchor"], r["negative"]
            if not usable(x, y, src) or frozenset((x, y)) in seen:
                continue
            ce.append({"query": text[x], "doc": text[y], "a_key": x, "b_key": y,
                       "label": 0.0, "source": src})

    rng = random.Random(seed)
    rng.shuffle(bi)
    rng.shuffle(ce)
    return bi, ce, stat


def main() -> None:
    D = "data/processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=f"{D}/corpus.jsonl")
    ap.add_argument("--gold-train", default=f"{D}/similar_gold_train.jsonl")
    ap.add_argument("--gold-eval", default=f"{D}/similar_gold_eval.jsonl")
    ap.add_argument("--weak", default=f"{D}/weak_positives.jsonl")
    ap.add_argument("--hard", default=f"{D}/hard_negatives.jsonl")
    ap.add_argument("--easy", default=f"{D}/easy_negatives.jsonl")
    ap.add_argument("--pooling", default=f"{D}/pooling_judgments_v3_llm.csv")
    ap.add_argument("--paraphrase", default=f"{D}/paraphrase_queries.jsonl")
    ap.add_argument("--exclude-paraphrase", action="store_true")
    ap.add_argument("--out-dir", default=D)
    ap.add_argument("--suffix", default="_clean")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if not args.suffix:
        sys.exit("--suffix rong se ghi de train_*.jsonl cu cua build_training_pairs.py")

    text = {}
    for d in load_jsonl(args.corpus):
        text[d["key"]] = d.get("summary", "") + "\n" + d.get("text_business", "")

    gold_eval_keys = {k for r in load_jsonl(args.gold_eval) for k in (r["a"], r["b"])}
    with open(args.pooling, encoding="utf-8-sig", newline="") as f:
        pool_rows = list(csv.DictReader(f))
    pool_q = {r["query_key"] for r in pool_rows}
    pool_pairs = {frozenset((r["query_key"], r["doc_key"])) for r in pool_rows}
    par_keys = {r["key"] for r in load_jsonl(args.paraphrase)}

    exclude = gold_eval_keys | pool_q
    if args.exclude_paraphrase:
        exclude |= par_keys

    bi, ce, stat = build(load_jsonl(args.gold_train), load_jsonl(args.weak),
                         load_jsonl(args.hard), load_jsonl(args.easy),
                         text, exclude, args.seed)

    # Soat lai chinh dau ra -- khong tin vao logic loc o tren.
    rows_keys = [(r["a_key"], r["b_key"]) for r in bi + ce]
    touched = {k for pair in rows_keys for k in pair}
    leak_excluded = touched & exclude
    leak_pool_pairs = sum(1 for x, y in rows_keys if frozenset((x, y)) in pool_pairs)
    if leak_excluded or leak_pool_pairs:
        sys.exit(f"RO RI: {len(leak_excluded)} issue bi loai van con, "
                 f"{leak_pool_pairs} cap pooling nam trong train")

    pos_keys = {k for r in bi for k in (r["a_key"], r["b_key"])}
    clean_par = sorted(par_keys - pos_keys)

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}
    for name, rows in (("train_bi_encoder", bi), ("train_cross_encoder", ce)):
        p = os.path.join(args.out_dir, f"{name}{args.suffix}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[name] = p
    split = {
        "policy": {"gold": os.path.basename(args.gold_train),
                   "exclude": ["similar_gold_eval issues", "pooling query_key"]
                   + (["paraphrase answer issues"] if args.exclude_paraphrase else []),
                   "seed": args.seed},
        "n_excluded_issues": len(exclude),
        "bi_encoder": dict(Counter(r["source"] for r in bi)),
        "cross_encoder": dict(Counter(r["source"] for r in ce)),
        "dropped": dict(stat),
        "leak_check": {"excluded_issues_in_output": 0, "pooling_pairs_in_output": 0},
        "paraphrase_clean_queries": len(clean_par),
        "paraphrase_total_queries": len(par_keys),
        "paraphrase_clean_keys": clean_par,
    }
    sp = os.path.join(args.out_dir, f"finetune_split{args.suffix}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)

    print(f"loai {len(exclude)} issue (gold eval {len(gold_eval_keys)}, pooling query "
          f"{len(pool_q)}{', paraphrase ' + str(len(par_keys)) if args.exclude_paraphrase else ''})")
    print(f"bi-encoder   : {len(bi)} cap duong  {split['bi_encoder']}")
    print(f"cross-encoder: {len(ce)} mau       {split['cross_encoder']}")
    print(f"bo            : {dict(stat)}")
    print("soat ro ri    : 0 issue bi loai, 0 cap pooling trong dau ra")
    print(f"paraphrase sach: {len(clean_par)}/{len(par_keys)} truy van "
          "(bao cao sau fine-tune tren tap nay)")
    for p in list(paths.values()) + [sp]:
        print(f"-> {p}")


if __name__ == "__main__":
    main()
