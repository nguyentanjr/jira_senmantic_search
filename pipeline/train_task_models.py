# -*- coding: utf-8 -*-
"""
Fine-tune bi-encoder (e5-base) va cross-encoder (bge-reranker-base) tren
task_pairs.jsonl (build_task_pairs.py). Chay tren GPU (Colab).

    python pipeline/train_task_models.py --model bi --out models/e5-base-task
    python pipeline/train_task_models.py --model ce --out models/bge-reranker-base-task
    python pipeline/train_task_models.py --model bi --out /tmp/smoke --max-steps 5   # thu

    # duong cong du lieu (learning_curve_eval.py): 25% so issue, seed train 14
    python pipeline/train_task_models.py --model bi --pairs data/processed/task_pairs_v2.jsonl \
        --fraction 0.25 --seed 14 --out models/lc/e5-f025-s14

SIEU THAM SO CO DINH TRUOC (2026-09-14) -- khong chon theo dev
    bi : MultipleNegativesRankingLoss (anchor, positive, 1 hard negative),
         NO_DUPLICATES, 1 epoch, lr 2e-5, batch 32, warmup 10%, seq 256,
         tien to 'query: ' / 'passage: ' (khop luong text)
    ce : BinaryCrossEntropyLoss, moi truy van 1 positive + 4 negative,
         1 epoch, lr 2e-5, batch 16, warmup 10%, max_length 256
1 epoch de han che troi xa model goc (lan fine-tune truoc lam hong e5).

--fraction giu TRON issue (moi dong cua issue duoc chon), tap long nhau theo
--frac-seed; sieu tham so giu nguyen -> it du lieu thi it buoc hon. Mac dinh
(--fraction 1, --seed 13) cho dung du lieu va seed nhu cac model da train.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

BASE_BI = "intfloat/multilingual-e5-base"
BASE_CE = "BAAI/bge-reranker-base"
SEED = 13


def load_rows(path: str, limit: int = 0) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:limit] if limit else rows


def subset_by_issue(rows: list[dict], fraction: float, seed: int = SEED) -> list[dict]:
    """Giu moi dong cua ceil(fraction x so issue) issue.

    Xao danh sach key da sort bang `seed` roi lay phan dau -> cac muc long nhau
    (tap 10% nam trong tap 25% ...), cung seed thi cung tap. Thu tu dong giu nhu
    file goc; fraction >= 1 tra ve nguyen danh sach.
    """
    if fraction >= 1.0:
        return rows
    if fraction <= 0 or any("key" not in r for r in rows):
        raise ValueError("fraction phai trong (0, 1] va moi dong phai co 'key'")
    keys = sorted({r["key"] for r in rows})
    random.Random(seed).shuffle(keys)
    keep = set(keys[:math.ceil(fraction * len(keys))])
    return [r for r in rows if r["key"] in keep]


def bi_columns(rows) -> dict:
    return {"anchor": [r["query"] for r in rows],
            "positive": [r["positive"] for r in rows],
            "negative": [r["negatives"][0] for r in rows]}


def ce_columns(rows) -> dict:
    q, d, y = [], [], []
    for r in rows:
        q.append(r["query"]); d.append(r["positive"]); y.append(1.0)
        for n in r["negatives"]:
            q.append(r["query"]); d.append(n); y.append(0.0)
    return {"query": q, "doc": d, "label": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bi", "ce"], required=True)
    ap.add_argument("--pairs", default="data/processed/task_pairs.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="model goc (mac dinh e5-base / bge-reranker-base)")
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = mac dinh (bi 32, ce 16). e5-large tren T4: bi 16 (vong 2, chot truoc)")
    ap.add_argument("--cached-mini-batch", type=int, default=0,
                    help="bi: >0 dung CachedMultipleNegativesRankingLoss (GradCache) voi mini-batch nay. "
                         "e5-large batch 16 thuong bi OOM tren T4 -> batch 32 + mini-batch 8")
    ap.add_argument("--max-steps", type=int, default=-1, help="chi de chay thu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="ty le so ISSUE giu lai (duong cong du lieu); 1.0 = tat ca")
    ap.add_argument("--frac-seed", type=int, default=SEED,
                    help="seed chon issue -- giu co dinh de cac muc long nhau")
    ap.add_argument("--seed", type=int, default=SEED, help="seed train")
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = subset_by_issue(load_rows(args.pairs, args.limit), args.fraction, args.frac_seed)
    common = dict(output_dir=args.out + "-ckpt", num_train_epochs=1, learning_rate=2e-5,
                  warmup_ratio=0.1, fp16=torch.cuda.is_available(), seed=args.seed,
                  logging_steps=25, save_strategy="no", report_to="none",
                  max_steps=args.max_steps)

    if args.model == "bi":
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments, losses)
        from sentence_transformers.training_args import BatchSamplers
        base = args.base or BASE_BI
        model = SentenceTransformer(base)
        model.max_seq_length = 256
        ds = Dataset.from_dict(bi_columns(rows))
        targs = SentenceTransformerTrainingArguments(
            per_device_train_batch_size=args.batch or 32, batch_sampler=BatchSamplers.NO_DUPLICATES,
            prompts={"anchor": "query: ", "positive": "passage: ", "negative": "passage: "},
            **common)
        loss = (losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.cached_mini_batch)
                if args.cached_mini_batch else losses.MultipleNegativesRankingLoss(model))
        SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
        n = len(ds)
    else:
        from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                         CrossEncoderTrainingArguments)
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        base = args.base or BASE_CE
        model = CrossEncoder(base, num_labels=1, max_length=256)
        ds = Dataset.from_dict(ce_columns(rows))
        targs = CrossEncoderTrainingArguments(per_device_train_batch_size=args.batch or 16, **common)
        CrossEncoderTrainer(model=model, args=targs, train_dataset=ds,
                            loss=BinaryCrossEntropyLoss(model)).train()
        n = len(ds)

    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    with open(os.path.join(args.out, "task_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base, "kind": args.model, "n_train_rows": n,
                   "batch": args.batch or (32 if args.model == "bi" else 16),
                   "cached_mini_batch": args.cached_mini_batch,
                   "n_pair_rows": len(rows), "n_issues": len({r.get("key") for r in rows}),
                   "fraction": args.fraction, "frac_seed": args.frac_seed,
                   "epochs": 1, "lr": 2e-5, "max_steps": args.max_steps,
                   "pairs": args.pairs, "seed": args.seed},
                  f, ensure_ascii=False, indent=1)
    print(f"-> {args.out} ({n} dong train)")


if __name__ == "__main__":
    sys.exit(main())
