"""Blind pair review before contrastive training: a direction label is not relevance.

Existing direction triples are development material, never a fresh holdout.
Ranks, cosine scores and proposed positive/negative roles stay in a sidecar.
No unreviewed pair is emitted as a training example.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

from build_corpus import scrub
from build_feedback_pairs import existing_eval_keys
from embedding_cache import file_hash
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl


def components(rows):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in rows:
        keys = [row["key"], row["positive_key"], *row["neg_keys"]]
        roots = [find(key) for key in keys]
        root = min(roots)
        for key in roots:
            parent[key] = root
    return {key: find(key) for key in parent}


def context(doc):
    # Include the complete business description: sections.actual can contain
    # desired behaviour or contradict the title. The reviewer must see both.
    sections = doc.get("sections") or {}
    text = "\n".join([doc.get("summary", ""),
                      "DESCRIPTION:\n" + (doc.get("text_business") or doc.get("text_dense", "")),
                      "ACTUAL:\n" + sections.get("actual", ""),
                      "EXPECTED:\n" + sections.get("expected", "")])
    return scrub(text)[0]


def make_pack(rows, docs, scores, excluded, anchors=40, seed=20260923):
    if anchors <= 0:
        raise ValueError("anchors must be positive")
    groups = components(rows)
    blocked = {groups[key] for key in excluded if key in groups}
    kept = [r for r in rows if groups[r["key"]] not in blocked]
    for row in kept:
        keys = [row["key"], row["positive_key"], *row["neg_keys"]]
        if len(set(keys)) != len(keys) or any(key not in docs for key in keys):
            raise ValueError(f"Missing/repeated issue in triple {row['key']}")
    # Historical cosine is only a review priority, not a verified label or gain.
    indexed = {(r["key"], r["pos_key"]): r for r in scores}

    def margin(row):
        score = indexed.get((row["key"], row["positive_key"]))
        if not score or set(score["cos_neg"]) != set(row["neg_keys"]):
            raise ValueError(f"Missing/mismatched score row for {row['key']}")
        return score["cos_pos"] - max(score["cos_neg"].values())

    for row in kept:
        margin(row)
    buckets = defaultdict(list)
    for row in kept:
        buckets[(row["axis"], row["side"])].append(row)
    rng = random.Random(seed)
    queues = []
    for bucket in sorted(buckets):
        ordered = sorted(buckets[bucket], key=lambda r: (margin(r), r["key"]))
        hard = ordered[:(len(ordered) + 1) // 2]
        rest = ordered[len(hard):]
        rng.shuffle(rest)
        queue = []
        while hard or rest:
            if hard:
                queue.append(hard.pop(0))
            if rest:
                queue.append(rest.pop(0))
        queues.append(queue)
    selected = []
    while len(selected) < anchors and any(queues):
        for queue in queues:
            if queue and len(selected) < anchors:
                selected.append(queue.pop(0))
    reviews, sidecar = [], []
    seen = set()
    for row in selected:
        score = indexed[(row["key"], row["positive_key"])]
        candidates = [(row["positive_key"], "proposed_positive", score["cos_pos"])]
        candidates.extend((key, "proposed_negative", score["cos_neg"][key]) for key in row["neg_keys"])
        for key, role, cosine in candidates:
            pair = (row["key"], key)
            if pair in seen:
                continue
            seen.add(pair)
            rid = hashlib.sha256((pair[0] + "\0" + pair[1]).encode()).hexdigest()[:16]
            reviews.append({"review_id": rid, "query_key": pair[0], "candidate_key": pair[1],
                            "query_context": context(docs[pair[0]]),
                            "candidate_context": context(docs[pair[1]]),
                            "same_context": "", "grade": "", "reason": "",
                            "reviewer": "", "notes": ""})
            sidecar.append({"review_id": rid, "group_id": groups[row["key"]],
                            "use": "development_review_only", "proposed_role": role,
                            "axis": row["axis"], "side": row["side"],
                            "historical_cosine": cosine, "historical_margin": margin(row),
                            "score_provenance": "historical file; not fresh inference"})
    rng.shuffle(reviews)
    counts = Counter(groups[row["key"]] for row in kept)
    stats = {"input_triples": len(rows), "eligible_triples": len(kept),
             "excluded_components": len(blocked), "selected_anchors": len(selected),
             "review_pairs": len(reviews), "connected_components_all_eligible": len(counts),
             "component_sizes_in_triples": sorted(counts.values(), reverse=True),
             "selected_by_axis_side": dict(Counter(r["axis"] + ":" + r["side"] for r in selected)),
             "training_rows_emitted": 0}
    return reviews, sidecar, stats


def expand_pack(reviews, sidecar, source_rows, docs, engine, excluded, per_anchor=2):
    """Find unlabelled candidates without requiring a pre-existing axis label.

    This bypasses the small direction-labelled/Jaccard pool. Similarity only
    selects what a human should inspect; it never supplies a relevance label.
    """
    existing = defaultdict(set)
    for row in source_rows:
        existing[row["key"]].update([row["positive_key"], *row["neg_keys"]])
    blocked_ids = {engine.key_to_idx[k] for k in excluded if k in engine.key_to_idx}
    added_links = []
    added = 0
    for key in sorted({r["query_key"] for r in reviews}):
        query = docs[key].get("text_dense") or docs[key].get("summary", "")
        banned = blocked_ids | {engine.key_to_idx[key]}
        banned |= {engine.key_to_idx[k] for k in existing[key] if k in engine.key_to_idx}
        trace = {}
        ranked = engine.search(query, SearchConfig(top_k=per_anchor), exclude=banned, trace=trace)
        for idx in ranked:
            candidate = engine.keys[idx]
            rid = hashlib.sha256((key + "\0" + candidate).encode()).hexdigest()[:16]
            reviews.append({"review_id": rid, "query_key": key, "candidate_key": candidate,
                            "query_context": context(docs[key]), "candidate_context": context(docs[candidate]),
                            "same_context": "", "grade": "", "reason": "", "reviewer": "", "notes": ""})
            sidecar.append({"review_id": rid, "use": "development_review_only",
                            "proposed_role": "unjudged_retrieved_candidate",
                            "score_provenance": "fresh verified corpus/query inference",
                            "bm25_rank": trace["bm25"].index(int(idx)) + 1 if int(idx) in trace["bm25"] else None,
                            "dense_rank": trace["dense"].index(int(idx)) + 1 if int(idx) in trace["dense"] else None})
            added_links.append({"key": key, "positive_key": candidate, "neg_keys": []})
            added += 1
    # Names in added_links are an input schema only, not positive labels.
    # Shared new candidates can connect previously separate split groups.
    graph = components(source_rows + added_links)
    by_id = {r["review_id"]: r for r in reviews}
    for row in sidecar:
        row["group_id"] = graph[by_id[row["review_id"]]["query_key"]]
    random.Random(20260923).shuffle(reviews)
    return {"new_unjudged_candidates": added,
            "connected_components_after_expansion": len(set(graph.values()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/processed/direction_pairs_v2_clean.jsonl")
    ap.add_argument("--scores", default="data/processed/direction_pairs_v2_e5_cos.json")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--anchors", type=int, default=40)
    ap.add_argument("--expand-candidates", type=int, default=0,
                    help="Additional unjudged retrieved documents per anchor; requires local ML runtime")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit("Output already exists; use a new directory")
    protected = ["data/processed/similar_gold_eval.jsonl",
                 "data/processed/pooling_judgments_v4_llm.csv",
                 "data/processed/contrastive_jira_review.csv",
                 "data/processed/finetune_split_clean.json"]
    # Preserve the strict exclusion policy of this particular source dataset.
    for path in protected:
        if not Path(path).is_file():
            raise SystemExit(f"Missing protected-set file: {path}")
    excluded = existing_eval_keys(*protected)
    rows = load_jsonl(args.pairs)
    docs = {r["key"]: r for r in load_jsonl(args.corpus)}
    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    reviews, sidecar, stats = make_pack(rows, docs, scores, excluded, args.anchors)
    if args.expand_candidates < 0:
        raise SystemExit("--expand-candidates must be nonnegative")
    if args.expand_candidates:
        import torch
        from audit_candidate_stages import QueryMemo
        from search import FLOWS
        torch.set_num_threads(4)
        flow = FLOWS["issue"]
        dense = DenseIndex(flow["model"], query_prefix=flow["query_prefix"],
                           passage_prefix=flow["passage_prefix"], batch_size=16)
        texts = [doc.get("text_dense", "") for doc in docs.values()]
        # Do not accidentally start a second full rebuild while an audit runs.
        import embedding_cache
        metadata = dense.cache_identity(texts)
        path = embedding_cache.cache_path(dense.cache_dir, "corpus", metadata)
        dense.emb = embedding_cache.load(path, metadata)
        if dense.emb is None:
            raise SystemExit("Run audit_candidate_stages first: verified corpus cache is missing")
        queries = [docs[key].get("text_dense") or docs[key].get("summary", "")
                   for key in sorted({r["query_key"] for r in reviews})]
        engine = SearchEngine(list(docs.values()), dense=QueryMemo(dense, queries),
                              bm25_fold=flow["bm25_fold"])
        engine.build_bm25()
        stats.update(expand_pack(reviews, sidecar, rows, docs, engine, excluded, args.expand_candidates))
        stats["review_pairs"] = len(reviews)
        stats["expansion_cache_identity"] = metadata
    out.mkdir(parents=True)
    with (out / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        cols = ["review_id", "query_key", "candidate_key", "query_context", "candidate_context",
                "same_context", "grade", "reason", "reviewer", "notes"]
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(reviews)
    (out / "provenance.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                 for r in sidecar), encoding="utf-8")
    stats["policy"] = {"blind": "proposed roles and scores hidden from review.csv",
                       "labels": "blank until independently reviewed",
                       "split": "development only; group by all connected issues before later splits",
                       "sample": "balance axis/side; alternate lower and higher historical margins",
                       "holdout": "must collect new independent queries; this pack is not a holdout"}
    sources = [args.pairs, args.scores, args.corpus, *protected, __file__]
    if args.expand_candidates:
        sources += [str(Path(__file__).with_name(name)) for name in
                    ("retrieval.py", "embedding_cache.py", "audit_candidate_stages.py", "search.py")]
    stats["sha256"] = {p: file_hash(p) for p in sources}
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text('''# Chấm quan hệ issue trước khi train

Chỉ mở `review.csv` khi chấm; `provenance.jsonl` giữ giả thuyết của bộ khai thác
để đối chiếu sau. Thứ tự đã được xáo, nhãn đều để trống.

- `same_context`: yes / no / unclear — cùng hành động, đối tượng và điều kiện nghiệp vụ?
- `grade`: 2 = cùng lỗi/có thể dùng trực tiếp để xử lý; 1 = liên quan và hữu ích
  cho điều tra cùng loại lỗi; 0 = khác loại lỗi hoặc hành vi ngược chiều; ? = thiếu bằng chứng.
- `reason`: same_defect / related_defect / opposite_direction / wrong_defect_type /
  different_context / unclear. Điền `reviewer`, ghi căn cứ ngắn ở `notes`.

Đọc cả tiêu đề, mô tả, actual và expected; nếu chúng mâu thuẫn thì chọn ?.
Chung trục “SMS thiếu/thừa” hoặc “nút bật/tắt” chưa đủ để xác nhận positive.
Hai issue ngược chiều nhưng khác ngữ cảnh cũng chưa phải hard negative hữu ích.
Không biến ô trống/? thành 0. Nhãn chiều trước đây có hỗ trợ từ AI; pack này
cần kỹ sư xác nhận quan hệ, không phải chỉ xác nhận nhãn chiều lần nữa.

Đây là dữ liệu phát triển đã được xem trong các thí nghiệm cũ. Không dùng làm
holdout mới. Chưa tạo file train; sau chấm phải kiểm tra cả positive và negative,
chia theo nhóm issue liên thông và đo trên truy vấn mới độc lập trước khi triển khai.
''', encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
