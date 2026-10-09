"""Score a fixed fresh issue pool with the unmodified BGE cross-encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def score_pairs(model, tokenizer, pairs, device, max_length, batch_size):
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            tokens = tokenizer([q for q, _ in batch], [d for _, d in batch],
                               padding=True, truncation=True, max_length=max_length,
                               return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(**tokens).logits.reshape(-1)
            scores.extend(logits.float().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model_outputs_used_for_anchor_selection"] or manifest["labels"] != 0:
        raise RuntimeError("Expected a fixed, unlabeled, model-independent anchor pool")
    pools = load_jsonl(data / "candidate_pools.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Use a Colab GPU runtime")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    rows = []
    for pool in pools:
        pairs = [(pool["query_text"], candidate["text"]) for candidate in pool["candidates"]]
        scores = score_pairs(model, tokenizer, pairs, device, args.max_length, args.batch_size)
        order = np.argsort(-scores, kind="stable")
        reranked = [pool["candidates"][i]["key"] for i in order]
        baseline = pool["baseline_ranking"]
        ce_rank = {key: i for i, key in enumerate(reranked)}
        safe_head = baseline[:1] + sorted(baseline[1:10], key=lambda key: ce_rank[key])
        rows.append({
            "query_key": pool["query_key"],
            "reranked_keys": reranked,
            "reranked_scores": [float(scores[i]) for i in order],
            "safe_top10_pin1": safe_head,
        })
        print(pool["query_key"], flush=True)
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {
        "completed": True,
        "queries": len(rows),
        "model": args.model,
        "device": str(device),
        "policy": "keep hybrid top-10 membership, pin hybrid rank 1, order positions 2-10 by base CE",
        "labels_used": 0,
        "manifest": manifest,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
