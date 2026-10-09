# -*- coding: utf-8 -*-
"""Huan luyen reranker hoc duoc tren dung vector dac trung cua phase 4.

    python pipeline/train_dup_reranker.py
    python pipeline/train_dup_reranker.py --model gbdt --out data/models/reranker.joblib

Sau do dung ma khong sua mot dong pipeline nao:

    python pipeline/run_dup_e2e.py --mode eval --split eval \\
        --set rerank.enabled=true --set rerank.kind=sklearn

Chinh sach nhan (day la phan de sai nhat, nen viet ro):

  DUONG   cap gold nam trong SPLIT TRAIN. Chi train split -- cum eval khong
          duoc dung o bat ky dau trong file nay.

  AM      ung vien do chinh he thong tra ve nhung KHONG phai gold, VA co it
          nhat mot trong hai bang chung:
             - co tin hieu mau thuan (khac tac nhan/kenh/trang thai/...), hoac
             - diem thap hon nguong 'related'.

  BO QUA  ung vien khong phai gold, khong co mau thuan, ma diem lai cao.
          Day la nhom "co the la trung lap chua ai ghi nhan" -- tap gold chi
          co 138 cap dao tu comment, chac chan KHONG day du. Gan chung la
          negative se day model hoc rang hai bug giong nhau la khac nhau.
          Bo qua thi mat mot it du lieu; gan sai thi hong nhan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dup import DuplicatePipeline, load_config          # noqa: E402
from pipeline.dup._compat import load_jsonl                      # noqa: E402
from pipeline.dup.evaluate import build_queries                  # noqa: E402
from pipeline.dup.features import FEATURE_NAMES                  # noqa: E402


def collect(pipe, queries, gold_map, max_neg_per_query=12):
    X, y, meta = [], [], Counter()
    t_related = pipe.cfg.decision.t_related
    for q in queries:
        i = pipe.key_to_idx.get(q["key"])
        if i is None:
            continue
        rep = pipe.reps[i]
        text = pipe.docs[i].get(pipe.cfg.embedding.text_field, "") \
            or pipe.docs[i].get("summary", "")
        cands = pipe.gen.generate(rep, text, exclude={i}, query_idx=i)
        rel = gold_map.get(q["key"], set())
        negs = 0
        for c in cands:
            f = pipe.feat.extract(rep, pipe.reps[c.idx],
                                  semantic=c.semantic_score,
                                  lexical=c.lexical_score)
            bd = pipe.scorer.score(f)
            if c.key in rel:
                X.append(f.vector()); y.append(1); meta["pos"] += 1
                continue
            if negs >= max_neg_per_query:
                continue
            if bd.negatives or bd.final < t_related:
                X.append(f.vector()); y.append(0); meta["neg"] += 1
                negs += 1
            else:
                meta["skipped_uncertain"] += 1
    return np.vstack(X) if X else np.zeros((0, len(FEATURE_NAMES))), \
        np.array(y), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="extend", nargs="+",
                    default=[])
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbdt"])
    ap.add_argument("--out", default="data/models/reranker.joblib")
    ap.add_argument("--max-neg-per-query", type=int, default=12)
    # Duoi truc "truy hoi issue tuong dong", nhan duong la MOI cap co grade >= 1
    # (ca "trung" lan "tuong tu"), khong chi cap trung lap. Xem
    # build_similar_gold.py: tap dung duoc tang tu ~121 len 208 cap.
    ap.add_argument("--train-pairs",
                    default="data/processed/similar_gold_train.jsonl")
    ap.add_argument("--eval-pairs",
                    default="data/processed/similar_gold_eval.jsonl")
    a = ap.parse_args()

    cfg = load_config(a.config, a.overrides)
    cfg.rerank.enabled = False           # khong rerank khi dang thu thap mau
    pipe = DuplicatePipeline(cfg).build()

    tr = load_jsonl(a.train_pairs)
    ev = load_jsonl(a.eval_pairs)
    print(f"nhan duong: {a.train_pairs}  ({len(tr)} cap)")
    q_tr = build_queries(tr, set(pipe.key_to_idx))
    gold_map = {q["key"]: set(q["relevant"]) for q in q_tr}

    eval_keys = {k for p in ev for k in (p["a"], p["b"])}
    q_tr = [q for q in q_tr if q["key"] not in eval_keys]
    print(f"truy van TRAIN: {len(q_tr)} (da loai moi key thuoc cum eval)")

    X, y, meta = collect(pipe, q_tr, gold_map, a.max_neg_per_query)
    print(f"mau: {dict(meta)}  X={X.shape}")
    if X.shape[0] < 20 or y.sum() < 5:
        sys.exit("khong du mau de huan luyen -- can them nhan gold")

    from sklearn.linear_model import LogisticRegression
    if a.model == "gbdt":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=cfg.seed)
    else:
        model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=cfg.seed)
    model.fit(X, y)

    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(X)[:, 1]
    print(f"AUC tren chinh tap train (chi de kiem tra fit, KHONG phai ket qua "
          f"bao cao): {roc_auc_score(y, p):.3f}")
    if hasattr(model, "coef_"):
        w = sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: -abs(x[1]))
        print("\nhe so lon nhat:")
        for n, c in w[:12]:
            print(f"  {n:<32} {c:+.3f}")

    import joblib
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(model, a.out)
    with open(a.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump({"features": list(FEATURE_NAMES), "model": a.model,
                   "n_samples": int(X.shape[0]), "n_pos": int(y.sum()),
                   "sample_policy": dict(meta)}, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {a.out}")
    print("dung: --set rerank.enabled=true --set rerank.kind=sklearn")


if __name__ == "__main__":
    main()
