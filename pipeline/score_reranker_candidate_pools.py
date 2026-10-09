"""Score self-contained candidate pools with a local cross-encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import CrossEncoderReranker, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    pools = load_jsonl(args.pools)
    reranker = CrossEncoderReranker(args.model, device=args.device,
                                    batch_size=args.batch_size, max_length=args.max_length)
    rows = []
    for pool in pools:
        pairs = [(pool["query_text"], candidate["text"]) for candidate in pool["candidates"]]
        scores = reranker.score_pairs(pairs)
        order = np.argsort(-scores, kind="stable")
        rows.append({
            "query_key": pool["query_key"],
            "reranked_keys": [pool["candidates"][i]["key"] for i in order],
            "reranked_scores": [float(scores[i]) for i in order],
            "model": args.model,
        })
        print(pool["query_key"], flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
