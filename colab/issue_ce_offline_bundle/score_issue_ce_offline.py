"""Cham BAAI/bge-reranker-base cho moi cap (issue, ung vien top-100) -- chi inference.

    python score_issue_ce_offline.py --data data --out output

Chay theo shard truy van; shard da xong (out/shards/*.npy) duoc bo qua, nen
runtime Colab bi ngat thi chay lai cung lenh voi --out tren Google Drive la
tiep tuc. Cuoi cung gop thanh out/ce_scores.npy (float32 [n_queries, 100],
NaN o o trong cua pools.npy) va out/manifest.json.

Cung cach cham voi danh gia 10 family (evaluate_relation_reranker.py):
AutoModelForSequenceClassification, logit tho, max_length 256, autocast fp16.
Trong moi shard cap duoc xep theo do dai de bot padding -- chi doi thu tu tinh,
diem moi cap giu nguyen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

ScoreFn = Callable[[Sequence[tuple[str, str]]], np.ndarray]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_shard(pools: np.ndarray, start: int, texts: Sequence[str],
                score_fn: ScoreFn) -> np.ndarray:
    """pools: [rows, k] chi so doc ung vien; truy van cua dong r la doc start + r."""
    rows, cols = np.nonzero(pools >= 0)
    out = np.full(pools.shape, np.nan, dtype=np.float32)
    if len(rows) == 0:
        return out
    q = rows + start
    c = pools[rows, cols]
    lengths = np.array([len(texts[a]) + len(texts[b]) for a, b in zip(q, c)])
    order = np.argsort(lengths, kind="stable")
    pairs = [(texts[q[i]], texts[c[i]]) for i in order]
    scores = np.asarray(score_fn(pairs), dtype=np.float32).reshape(-1)
    if len(scores) != len(pairs):
        raise RuntimeError(f"scorer tra {len(scores)} diem cho {len(pairs)} cap")
    out[rows[order], cols[order]] = scores
    return out


def run(data: Path, out: Path, score_fn: ScoreFn, scorer_info: dict,
        shard_size: int = 250, log: Callable[[str], None] = print) -> dict:
    manifest_in = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    texts = [d["text"] for d in load_jsonl(data / "docs.jsonl")]
    pools = np.load(data / "pools.npy")
    if pools.shape[0] != manifest_in["n_queries"] or len(texts) != manifest_in["n_docs"]:
        raise RuntimeError("pools/docs khong khop manifest")
    pools_sha = file_sha(data / "pools.npy")
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stamp = shard_dir / "pools_sha256.txt"
    if stamp.exists() and stamp.read_text().strip() != pools_sha:
        raise RuntimeError(f"{shard_dir} chua shard cua bundle KHAC; xoa hoac doi --out")
    stamp.write_text(pools_sha)

    n = pools.shape[0]
    t0, done_pairs = time.perf_counter(), 0
    for start in range(0, n, shard_size):
        end = min(n, start + shard_size)
        path = shard_dir / f"scores_{start:06d}_{end:06d}.npy"
        if path.exists():
            continue
        block = score_shard(pools[start:end], start, texts, score_fn)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, block)
        tmp.replace(path)
        done_pairs += int((pools[start:end] >= 0).sum())
        el = time.perf_counter() - t0
        log(f"{end}/{n} truy van | {done_pairs / el:.0f} cap/s | {el / 60:.1f} phut")

    scores = np.full(pools.shape, np.nan, dtype=np.float32)
    for start in range(0, n, shard_size):
        end = min(n, start + shard_size)
        scores[start:end] = np.load(shard_dir / f"scores_{start:06d}_{end:06d}.npy")
    valid = pools >= 0
    if np.isnan(scores[valid]).any() or not np.isnan(scores[~valid]).all():
        raise RuntimeError("NaN lech voi o trong cua pools")
    if not np.isfinite(scores[valid]).all():
        raise RuntimeError("diem khong huu han")
    np.save(out / "ce_scores.npy", scores)
    manifest = {"input": manifest_in, "pools_sha256": pools_sha,
                "docs_sha256": file_sha(data / "docs.jsonl"),
                "scores_shape": list(scores.shape), "scored_pairs": int(valid.sum()),
                "scorer": scorer_info, "labels_used": False, "training": False,
                "score_stats": {"min": float(scores[valid].min()),
                                "max": float(scores[valid].max()),
                                "mean": float(scores[valid].mean())}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return manifest


def make_hf_scorer(model_name: str, max_length: int, batch_size: int,
                   device_name: str = "cuda") -> tuple[ScoreFn, dict]:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Can GPU: Runtime > Change runtime type > T4 GPU")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    model.eval()

    def score(pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        res = []
        with torch.inference_mode():
            for s in range(0, len(pairs), batch_size):
                batch = pairs[s:s + batch_size]
                tok = tokenizer([q for q, _ in batch], [d for _, d in batch], padding=True,
                                truncation=True, max_length=max_length, return_tensors="pt")
                tok = {k: v.to(device) for k, v in tok.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(**tok).logits.reshape(-1)
                res.append(logits.float().cpu().numpy())
        return np.concatenate(res) if res else np.empty(0, dtype=np.float32)

    info = {"model": model_name, "model_commit": getattr(model.config, "_commit_hash", None),
            "max_length": max_length, "batch_size": batch_size,
            "autocast": "float16" if device.type == "cuda" else None,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "torch": torch.__version__,
            "transformers": transformers.__version__}
    return score, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--device", default="cuda", help="cpu chi de chay thu vai truy van")
    args = ap.parse_args()
    score_fn, info = make_hf_scorer(args.model, args.max_length, args.batch_size, args.device)
    m = run(Path(args.data), Path(args.out), score_fn, info, args.shard_size,
            log=lambda s: print(s, flush=True))
    print(json.dumps({k: m[k] for k in ("scores_shape", "scored_pairs", "score_stats")},
                     indent=2))


if __name__ == "__main__":
    main()
