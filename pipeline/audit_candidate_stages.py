"""Fresh CPU issue-flow audit; no model selection or historical-test tuning.

Writes branch traces, known-label failure stages, and a paired comparison with
an optional saved pre-fix retrieval.py. Old embeddings are never imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np

from dump_candidates import load_gold, load_pooling
from embedding_cache import file_hash
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl
from search import FLOWS


def measure(keys, rel):
    positive = {key for key, grade in rel.items() if grade >= 1}
    ideal = sorted(rel.values(), reverse=True)[:10]
    denominator = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return {"recall@10": len(set(keys[:10]) & positive) / len(positive),
            "recall@100": len(set(keys[:100]) & positive) / len(positive),
            "ndcg@10": sum(rel.get(key, 0) / np.log2(i + 2)
                           for i, key in enumerate(keys[:10])) / denominator}


def classify(key, trace):
    if key in trace["final"][:10]:
        return "hit_top10"
    if key in trace["fused"]:
        return "ranking_below_top10"
    if key in trace["union"]:
        return "lost_in_fusion"
    return "missing_both_branches"


class QueryMemo:
    """Share exactly the same fresh vectors across the old and new engine."""
    def __init__(self, dense, queries):
        self.dense = dense
        texts = list(dict.fromkeys(queries))
        vectors = dense.encode(texts, is_query=True, show_progress=True)
        self.vectors = dict(zip(texts, vectors))

    def encode(self, texts, is_query):
        return np.asarray([self.vectors[text] for text in texts])

    def score(self, vector):
        return self.dense.score(vector)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--gold", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="data/processed/pooling_judgments_v4_llm.csv")
    ap.add_argument("--legacy-source", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    paths = [args.corpus, args.gold, args.pooling, __file__,
             str(Path(__file__).with_name("retrieval.py")),
             str(Path(__file__).with_name("embedding_cache.py"))]
    if args.legacy_source:
        paths.append(args.legacy_source)
    fingerprints = {str(p): file_hash(p) for p in paths}
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit("Output already exists; use a new directory")
    out.mkdir(parents=True)
    import torch
    torch.set_num_threads(args.threads)
    flow = FLOWS["issue"]
    docs = load_jsonl(args.corpus)
    dense = DenseIndex(flow["model"], device="cpu", query_prefix=flow["query_prefix"],
                       passage_prefix=flow["passage_prefix"], batch_size=16)
    texts = [d.get("text_dense", "") for d in docs]
    print(f"Building verified corpus cache for {len(docs)} issues", flush=True)
    dense.build(texts, show_progress=True)
    sets = {"gold": load_gold(args.gold), "pooling_v4": load_pooling(args.pooling)}
    by_key = {d["key"]: d for d in docs}
    queries = [(name, key, by_key[key].get("text_dense") or by_key[key].get("summary", ""),
                {k: g for k, g in rel.items() if k in by_key and k != key})
               for name, labels in sets.items() for key, rel in sorted(labels.items())
               if key in by_key]
    memo = QueryMemo(dense, [q[2] for q in queries])
    eng = SearchEngine(docs, dense=memo, bm25_fold=flow["bm25_fold"])
    eng.build_bm25()
    legacy = None
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location("retrieval_before_fix", args.legacy_source)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        legacy = module.SearchEngine(docs, dense=memo, bm25_fold=flow["bm25_fold"])
        legacy._bm25 = eng._bm25
    cfg = SearchConfig(top_k=100, candidate_k=100)
    rows = []
    with (out / "traces.jsonl").open("w", encoding="utf-8") as stream:
        for name, key, query, rel in queries:
            trace = {}
            exclude = {eng.key_to_idx[key]}
            ranked = eng.search(query, cfg, exclude=exclude, trace=trace)
            traces = {stage: [eng.keys[i] for i in ids] for stage, ids in trace.items()}
            positive = {k for k, g in rel.items() if g >= 1}
            row = {"set": name, "key": key, "trace": traces, "rel": rel,
                   "stages": {k: classify(k, traces) for k in sorted(positive)},
                   "metrics": measure(traces["final"], rel) if positive else None}
            if legacy is not None:
                before = [eng.keys[i] for i in legacy.search(query, cfg, exclude=exclude)]
                row.update({"before": before,
                            "before_metrics": measure(before, rel) if positive else None,
                            "same_top10": before[:10] == traces["final"][:10],
                            "candidate_count_before": len(before),
                            "candidate_count_after": len(ranked)})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            rows.append(row)
    report = {"flow": "issue", "inference": "fresh CPU query embeddings",
              "corpus_embeddings": "verified content-addressed cache" if dense.last_build_cache_hit else "fresh CPU inference",
              "scope": "development diagnostics; labels incomplete; not independent acceptance",
              "threads": args.threads, "config": cfg.__dict__,
              "cache_identity": dense.cache_identity(texts), "sets": {}}
    for name in sets:
        selected = [r for r in rows if r["set"] == name and r["metrics"] is not None]
        stages = Counter(stage for r in selected for stage in r["stages"].values())
        summary = {"queries": len(selected), "known_relevant_pairs": sum(stages.values()),
                   "stages": dict(stages),
                   "metrics": {m: float(np.mean([r["metrics"][m] for r in selected]))
                               for m in ("recall@10", "recall@100", "ndcg@10")}}
        if legacy is not None:
            summary["before_metrics"] = {m: float(np.mean([r["before_metrics"][m] for r in selected]))
                                         for m in summary["metrics"]}
            summary["top10_changed"] = sum(not r["same_top10"] for r in selected)
            summary["candidate_count_fixed"] = sum(r["candidate_count_after"] > r["candidate_count_before"]
                                                   for r in selected)
            summary["rescued_known_pairs_at10"] = sum(len(set(r["trace"]["final"][:10])
                .difference(r["before"][:10]).intersection(r["stages"])) for r in selected)
            summary["lost_known_pairs_at10"] = sum(len(set(r["before"][:10])
                .difference(r["trace"]["final"][:10]).intersection(r["stages"])) for r in selected)
        report["sets"][name] = summary
    if fingerprints != {str(p): file_hash(p) for p in paths}:
        raise RuntimeError("Audit inputs/source changed during inference; no report published")
    report["sha256"] = fingerprints
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["sets"], indent=2), flush=True)


if __name__ == "__main__":
    main()
