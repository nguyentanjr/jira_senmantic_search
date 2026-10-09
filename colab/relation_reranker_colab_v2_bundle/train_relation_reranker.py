"""Train and gate an issue-to-issue cross-encoder reranker.

Designed for Google Colab GPU.  It exports weights only when every declared
gate passes; metrics and predictions are always exported.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


class Preferences(Dataset):
    def __init__(self, rows: list[dict]): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def score_pairs(model, tokenizer, pairs, device, max_length, batch_size=32):
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            tokens = tokenizer([q for q, _ in batch], [d for _, d in batch], padding=True,
                               truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            values.extend(model(**tokens).logits.reshape(-1).float().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def preference_metrics(rows, pos_scores, neg_scores):
    by_query = defaultdict(list)
    for row, p, n in zip(rows, pos_scores, neg_scores):
        by_query[row["query_key"]].append(float(p > n) + 0.5 * float(p == n))
    return {
        "pairs": len(rows),
        "queries": len(by_query),
        "accuracy_micro": float(np.mean(pos_scores > neg_scores)) if rows else None,
        "accuracy_macro_query": float(np.mean([np.mean(v) for v in by_query.values()])) if by_query else None,
        "mean_score_gap": float(np.mean(pos_scores - neg_scores)) if rows else None,
    }


def dcg(grades):
    if not grades: return 0.0
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ranking_metrics(ranking, labels):
    positives = {k for k, g in labels.items() if int(g) > 0}
    known_in_candidates = positives & set(ranking)
    top10 = ranking[:10]
    judged_order = [k for k in ranking if k in labels]
    gains = [int(labels[k]) for k in judged_order]
    ideal = sorted((int(g) for g in labels.values()), reverse=True)
    denom = dcg(ideal)
    return {
        "known_recall10": sum(k in positives for k in top10) / len(positives) if positives else None,
        "known_recall100": len(known_in_candidates) / len(positives) if positives else None,
        "known_zero_count10": sum(int(labels.get(k, -1)) == 0 for k in top10),
        "judged_pool_ndcg": dcg(gains) / denom if denom else None,
    }


def mean_metric(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def evaluate_pools(model, tokenizer, pools, device, max_length, batch_size):
    records = []
    for pool in pools:
        pairs = [(pool["query_text"], c["text"]) for c in pool["candidates"]]
        scores = score_pairs(model, tokenizer, pairs, device, max_length, batch_size)
        order = np.argsort(-scores, kind="stable")
        reranked = [pool["candidates"][i]["key"] for i in order]
        labels = {k: int(v) for k, v in pool["judgments"].items()}
        records.append({
            "query_key": pool["query_key"],
            "baseline": ranking_metrics(pool["baseline_ranking"], labels),
            "reranked": ranking_metrics(reranked, labels),
            "reranked_keys": reranked,
        })
    summary = {}
    for mode in ("baseline", "reranked"):
        summary[mode] = {k: mean_metric([r[mode] for r in records], k) for k in
                         ("known_recall10", "known_recall100", "known_zero_count10", "judged_pool_ndcg")}
    return summary, records


def collate_fn(tokenizer, max_length):
    def collate(rows):
        pos = tokenizer([r["query_text"] for r in rows], [r["positive_text"] for r in rows],
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        neg = tokenizer([r["query_text"] for r in rows], [r["negative_text"] for r in rows],
                        padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return pos, neg, torch.tensor([r["margin"] for r in rows], dtype=torch.float32), \
            torch.tensor([r["query_weight"] for r in rows], dtype=torch.float32)
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--max-steps", type=int, default=0, help="0 means full run; use 1 for a smoke test")
    ap.add_argument("--lora", action="store_true", help="train a small LoRA adapter")
    ap.add_argument("--export-candidate", action="store_true",
                    help="export a quarantined LoRA adapter even when the preliminary gate fails")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    train = load_jsonl(data / "train_preferences.jsonl")
    dev = load_jsonl(data / "dev_preferences.jsonl")
    pools = load_jsonl(data / "dev_candidate_pools.jsonl")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["document_overlap_train_dev"] or manifest["unjudged_used_as_negative"]:
        raise RuntimeError("Bundle leakage/integrity assertion failed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    if args.export_candidate and not args.lora:
        raise ValueError("--export-candidate requires --lora to keep the artifact small")
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["query", "value"], modules_to_save=["classifier"], bias="none"))
        model.print_trainable_parameters()
    dev_pos = [(r["query_text"], r["positive_text"]) for r in dev]
    dev_neg = [(r["query_text"], r["negative_text"]) for r in dev]
    base_pref = preference_metrics(dev,
        score_pairs(model, tokenizer, dev_pos, device, args.max_length, args.eval_batch_size),
        score_pairs(model, tokenizer, dev_neg, device, args.max_length, args.eval_batch_size))
    base_pool, _ = evaluate_pools(model, tokenizer, pools, device, args.max_length, args.eval_batch_size)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Preferences(train), batch_size=args.batch_size, shuffle=True,
                        generator=generator, collate_fn=collate_fn(tokenizer, args.max_length))
    train_query_count = len({r["query_key"] for r in train})
    # query_weight sums to one per query.  This factor preserves ordinary
    # mean-loss scale while keeping every query's total epoch contribution equal.
    query_normalizer = len(train) / train_query_count
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(loader) * args.epochs / args.grad_accum)
    if args.max_steps: total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(total_steps * 0.1)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train(); optimizer.zero_grad(set_to_none=True)
    updates = 0; losses = []
    for epoch in range(args.epochs):
        for step, (pos, neg, margin, weight) in enumerate(loader):
            pos = {k: v.to(device) for k, v in pos.items()}; neg = {k: v.to(device) for k, v in neg.items()}
            margin, weight = margin.to(device), weight.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ps = model(**pos).logits.reshape(-1); ns = model(**neg).logits.reshape(-1)
                raw = torch.nn.functional.softplus(margin - (ps - ns))
                loss = (raw * weight * query_normalizer).mean() / args.grad_accum
            scaler.scale(loss).backward(); losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                updates += 1
                if args.max_steps and updates >= args.max_steps: break
        if args.max_steps and updates >= args.max_steps: break

    trained_pref = preference_metrics(dev,
        score_pairs(model, tokenizer, dev_pos, device, args.max_length, args.eval_batch_size),
        score_pairs(model, tokenizer, dev_neg, device, args.max_length, args.eval_batch_size))
    trained_pool, predictions = evaluate_pools(model, tokenizer, pools, device, args.max_length, args.eval_batch_size)
    b, t = trained_pool["baseline"], trained_pool["reranked"]
    per_query_safe = all(r["reranked"]["known_recall10"] >= r["baseline"]["known_recall10"]
                         for r in predictions if r["baseline"]["known_recall10"] is not None)
    gates = {
        "judged_pool_ndcg_improves": t["judged_pool_ndcg"] > b["judged_pool_ndcg"],
        "known_recall10_not_lower": t["known_recall10"] >= b["known_recall10"],
        "known_recall100_unchanged": t["known_recall100"] >= b["known_recall100"],
        "known_grade0_exposure_not_higher": t["known_zero_count10"] <= b["known_zero_count10"],
        "no_dev_query_loses_known_recall10": per_query_safe,
        "preference_accuracy_improves_over_base_ce": (
            trained_pref["accuracy_macro_query"] > base_pref["accuracy_macro_query"]),
        "full_run": args.max_steps == 0,
    }
    passed = all(gates.values())
    report = {
        "accepted": passed,
        "weights_exported": bool(passed or args.export_candidate),
        "model_format": "lora_adapter" if args.lora else "full_model",
        "device": str(device),
        "updates": updates,
        "mean_training_loss": float(np.mean(losses)) if losses else None,
        "base_cross_encoder_preferences": base_pref,
        "trained_preferences": trained_pref,
        "base_cross_encoder_pool": base_pool,
        "production_baseline_vs_trained": trained_pool,
        "gates": gates,
        "limitations": [
            "AI development labels, not an independent human-reviewed test set",
            "Only five held-out issue families; passing is a research gate, not production proof",
            "Unjudged top-100 candidates remain unknown and are never counted as negatives",
        ],
        "config": vars(args),
        "data_manifest": manifest,
    }
    dump_jsonl(out / "dev_predictions.jsonl", predictions)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if passed:
        model_dir = out / "accepted_model"
        model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
        (model_dir / "relation_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_candidate:
        model_dir = out / "candidate_adapter"
        model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
        (model_dir / "NOT_ACCEPTED.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": passed, "gates": gates,
                      "production_baseline_vs_trained": trained_pool,
                      "trained_preferences": trained_pref}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
