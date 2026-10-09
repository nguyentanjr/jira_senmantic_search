# -*- coding: utf-8 -*-
"""
Thu hoi cue trong comment thanh NHAN LIEN QUAN PHAN BAC.

    python pipeline/build_similar_gold.py

DUOI TRUC CU, 60% TAP GOLD BI COI LA NHAN BAN
Khi de tai con dat la "phat hien trung lap", cue "tuong tu" gay roi: no khong
phai duplicate nen bi tron vao nhan duong roi lam hong dinh nghia, hoac bi loai.
Do tren 326 cap dao duoc: `tuong tu` 181 (55,5%), `trung` 109, `giong bug` 19,
`duplicate` 12.

DUOI TRUC MOI, NO CHINH LA NHAN CAN TIM
De tai da chuyen sang "truy hoi issue tuong dong de tai su dung tri thuc". Khi
ky su tu tay viet "van de nay tuong tu RO-xxx" trong comment, ho da XAC NHAN
rang doc RO-xxx co ich khi dieu tra issue nay. Do dung la quan he can do.

    cue trung / duplicate           -> grade 2  (giup truc tiep)
    cue tuong tu / giong bug / loi  -> grade 1  (dang doc)
    cue da log bug / da co bug      -> LOAI     (chi noi "van de da duoc ghi
                                                 nhan o dau do", co the tro toi
                                                 mot bug khac han)

Tap dung duoc tang tu ~121 cap len ~300, khong ton them cong gan nhan nao.

VE HUB
`build_training_pairs.py` chan hub o bac 6 vi RO-11923 co 36 canh -- duoi truc
trung lap thi 36 cap trung lap quanh mot issue la vo ly. Duoi truc nay thi
NGUOC LAI: mot issue duoc 36 issue khac tro toi la tai lieu co gia tri NHAT,
dung thu ma he thong nen dua len dau. Nen mac dinh o day KHONG chan hub
(--max-degree 0). Van de ve suy giam da dang cho tap TRAIN thi giai bang cach
khac, khong phai bang cach vut bo nhan dung.

Van giu: tach train/eval theo CUM LIEN THONG, tinh tren TOAN BO cap dao duoc
(khong chi cap nam trong corpus) -- neu khong se ro ri qua cap bac cau.

Dau ra:
  data/processed/similar_gold_eval.jsonl
  data/processed/similar_gold_train.jsonl
  data/processed/similar_gold_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution                                                   # noqa: E402
from retrieval import load_jsonl                                    # noqa: E402

GRADE = {
    "trùng": 2, "duplicate": 2,
    "tương tự": 1, "giống bug": 1, "giống lỗi": 1,
}
DROP = {"đã log bug", "đã có bug"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--pairs", default="data/processed/dup_candidates_from_text.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--eval-ratio", type=float, default=0.5)
    ap.add_argument("--max-degree", type=int, default=0,
                    help="0 = khong chan hub (mac dinh duoi truc nay)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    docs = {d["key"]: d for d in load_jsonl(args.corpus)}
    graded, all_pairs, skipped = [], [], Counter()
    for p in load_jsonl(args.pairs):
        a, b = p.get("a"), p.get("b")
        if not (a and b) or a == b:
            continue
        all_pairs.append((a, b))
        cue = (p.get("cue") or "").strip().lower()
        if cue in DROP:
            skipped["cue khong xac nhan"] += 1
            continue
        g = GRADE.get(cue)
        if g is None:
            skipped[f"cue la {cue!r}"] += 1
            continue
        if a not in docs or b not in docs:
            skipped["khong nam trong corpus"] += 1
            continue
        graded.append({"a": a, "b": b, "cue": cue, "grade": g})

    if args.max_degree:
        deg = Counter()
        for x in graded:
            deg[x["a"]] += 1
            deg[x["b"]] += 1
        hot = {k for k, v in deg.items() if v > args.max_degree}
        seen, kept = Counter(), []
        for x in sorted(graded, key=lambda z: -z["grade"]):
            h = [k for k in (x["a"], x["b"]) if k in hot]
            if any(seen[k] >= args.max_degree for k in h):
                continue
            for k in h:
                seen[k] += 1
            kept.append(x)
        print(f"chan hub bac {args.max_degree}: {len(graded)} -> {len(kept)}")
        graded = kept

    # -- tach theo cum lien thong, tren TOAN BO cap (chong ro ri bac cau) ----
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in all_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps = defaultdict(list)
    for x in graded:
        comps[find(x["a"])].append(x)
    ids = sorted(comps)
    random.shuffle(ids)
    n_eval = int(len(ids) * args.eval_ratio)
    ev = [x for c in ids[:n_eval] for x in comps[c]]
    tr = [x for c in ids[n_eval:] for x in comps[c]]

    os.makedirs(args.out_dir, exist_ok=True)

    def dump(name, rows):
        p = os.path.join(args.out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {p}  ({len(rows)})")

    dump("similar_gold_eval.jsonl", ev)
    dump("similar_gold_train.jsonl", tr)

    # bao nhieu tai lieu dap an co resolution dung duoc -> tran cua
    # resolution-hit@K tren tap nay
    tgt = {x["b"] for x in ev} | {x["a"] for x in ev}
    with_res = sum(1 for k in tgt if k in docs and resolution.has(docs[k]))
    stats = {
        "n_pairs_graded": len(graded),
        "by_grade": dict(Counter(x["grade"] for x in graded)),
        "by_cue": dict(Counter(x["cue"] for x in graded)),
        "skipped": dict(skipped),
        "n_components": len(ids),
        "n_eval": len(ev), "n_train": len(tr),
        "eval_issues": len(tgt),
        "eval_issues_with_resolution": with_res,
    }
    with open(os.path.join(args.out_dir, "similar_gold_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"\ncap co nhan phan bac : {len(graded)}")
    for g, v in sorted(Counter(x['grade'] for x in graded).items(), reverse=True):
        print(f"  grade {g}: {v}")
    for k, v in skipped.most_common():
        print(f"  bo qua: {k}: {v}")
    print(f"cum lien thong       : {len(ids)}  -> eval {len(ev)} / train {len(tr)}")
    print(f"issue trong tap eval : {len(tgt)}, "
          f"co resolution dung duoc: {with_res} ({with_res/max(1,len(tgt)):.1%})")


if __name__ == "__main__":
    main()
