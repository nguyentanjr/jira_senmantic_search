"""Dev-only candidate audit. Live BM25 + explicitly labelled historical CE replay.

No model inference, parameter selection, or test-set tuning. All variants of a
source query stay in its existing dev split. Each reported slice contains at most
one observation per source issue; variants are never pooled for bootstrap CIs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from eval_paraphrase import paired
from eval_rerank_k import SETS, load_dev_keys
from eval_rerankers import metrics_from_ranked
from eval_similar import queries_from_gold, queries_from_pooling
from feedback_state import replay
from group_stats import has_signal
from retrieval import (SearchConfig, SearchEngine, fuse_rerank, load_jsonl,
                       strip_diacritics, top_k_allowed)


def variants(text):
    def partial(parity):
        count = 0

        def replace(match):
            nonlocal count
            word = match.group()
            if strip_diacritics(word) != word:
                count += 1
                if count % 2 == parity:
                    return strip_diacritics(word)
            return word

        return re.sub(r"\S+", replace, text)

    return {"goc": text, "khong_dau": strip_diacritics(text),
            "mixed_a": partial(0), "mixed_b": partial(1),
            "uppercase": text.upper(), "unicode_nfd": unicodedata.normalize("NFD", text),
            "whitespace": " \t ".join(text.split())}


def rank_of(ranked, relevant):
    return next((i + 1 for i, k in enumerate(ranked) if k in relevant), None)


def replay_stage(row):
    hybrid = np.asarray(row["hybrid"], dtype=object)
    if len(row["ce"]) < min(20, len(hybrid)):
        raise ValueError("incomplete CE cache")
    head = hybrid[:20]
    final = np.concatenate([fuse_rerank(head, np.asarray(row["ce"][:20]),
                                       "blend", 0.3), hybrid[20:]])
    relevant = set(row["relevant"])
    h, f = rank_of(hybrid, relevant), rank_of(final, relevant)
    final_hit = f is not None and f <= 10
    hybrid_hit = h is not None and h <= 10
    stage = ("success" if final_hit else "missing_top100" if h is None
             else "outside_top20" if h > 20 else "inside_head_ranking")
    return {"hybrid_rank": h, "final_rank": f, "failure_stage": stage,
            "rerank_lost_hit10": hybrid_hit and not final_hit,
            "rerank_rescued_hit10": not hybrid_hit and final_hit,
            "final_metrics": metrics_from_ranked(final, relevant),
            "hybrid_metrics": metrics_from_ranked(hybrid, relevant)}


def slices(query):
    signal = has_signal(query["query"])
    groups = ["all", "technical_signal" if signal else "no_technical_signal",
              "short_le8" if len(query["query"].split()) <= 8 else "long_gt8"]
    if query.get("jac_bin") == "<0.2":
        groups.append("low_lexical_overlap")
    return groups


def legacy_keyword(engine, query, exclude=None):
    """Exact old keyword path: top-100 including zero padding, then exclude."""
    scores = engine.bm25_scores(query)
    ranked = top_k_allowed(scores, 100)
    if exclude:
        ranked = np.asarray([i for i in ranked if i not in exclude], dtype=np.int64)
    return scores, ranked


def lexical_row(engine, query, relevant, exclude=None):
    scores, before = legacy_keyword(engine, query, exclude)
    after = engine.search(query, SearchConfig(use_dense=False, top_k=100), exclude=exclude)
    return {"positive_docs": int(np.count_nonzero(scores > 0)),
            "zero_padded_before": int(np.count_nonzero(scores[before] == 0)),
            "zero_padded_after": int(np.count_nonzero(scores[after] == 0)),
            "candidate_list_unchanged": bool(np.array_equal(before, after)),
            "top10_unchanged": bool(np.array_equal(before[:10], after[:10])),
            "before": metrics_from_ranked(before, relevant),
            "after": metrics_from_ranked(after, relevant),
            "before_rank": rank_of(before, relevant), "after_rank": rank_of(after, relevant),
            "before_top10": [engine.keys[i] for i in before[:10]],
            "after_top10": [engine.keys[i] for i in after[:10]]}


def summarize_lexical(rows, n_boot):
    result = {}
    groups = defaultdict(list)
    for row in rows:
        for group in row["groups"]:
            groups[f"{row['set']}:{row['variant']}:{group}"].append(row)
    for name, items in sorted(groups.items()):
        if len({r["key"] for r in items}) != len(items):
            raise ValueError(f"duplicate source issue in slice {name}")
        stats = {"n": len(items), "changed_top10": sum(not r["top10_unchanged"] for r in items),
                 "changed_candidates": sum(not r["candidate_list_unchanged"] for r in items),
                 "sparse_lt100": sum(r["positive_docs"] < 100 for r in items),
                 "no_lexical_matches": sum(r["positive_docs"] == 0 for r in items),
                 "zero_padding_before": sum(r["zero_padded_before"] for r in items),
                 "zero_padding_after": sum(r["zero_padded_after"] for r in items),
                 "metrics": {}}
        for metric in ("recall@10", "mrr@10"):
            before = np.asarray([r["before"][metric] for r in items])
            after = np.asarray([r["after"][metric] for r in items])
            stats["metrics"][metric] = {"before": float(before.mean()), "after": float(after.mean()),
                                        "paired": paired(after, before, n_boot),
                                        "wins": int((after > before).sum()),
                                        "losses": int((after < before).sum())}
        result[name] = stats
    return result


def summarize_stages(rows):
    groups = defaultdict(list)
    for row in rows:
        for group in row["groups"]:
            groups[f"{row['set']}:{row['variant']}:{group}"].append(row)
    return {name: {"n": len(items),
                   "stages": dict(Counter(r["failure_stage"] for r in items)),
                   "lost_hit10": sum(r["rerank_lost_hit10"] for r in items),
                   "rescued_hit10": sum(r["rerank_rescued_hit10"] for r in items),
                   "final_recall10": float(np.mean([r["final_metrics"]["recall@10"] for r in items])),
                   "lexical_candidates_unchanged": sum(r["lexical_candidates_unchanged"] for r in items)}
            for name, items in sorted(groups.items())}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pilot_diagnostics(searches, events):
    """Final explicit judgments only. No-answer refers to shown results, not corpus."""
    by_id = {s["search_id"]: s for s in searches}
    labels, no_answer, _, changes = replay(
        events, lambda e: e.get("search_id") if e.get("search_id") in by_id else None)
    failures = []
    reasons = Counter()
    queries_by_reason = defaultdict(set)
    for sid, search in by_id.items():
        judged = [event for (event_sid, _), event in labels.items() if event_sid == sid]
        negative = [e for e in judged if e["verdict"] == "not_related"]
        positive = [e["result_key"] for e in judged if e["verdict"] == "related"]
        for e in negative:
            reason = e.get("reason") or "unspecified"
            reasons[reason] += 1
            queries_by_reason[reason].add(sid)
        if negative or sid in no_answer:
            failures.append({"search_id": sid, "flow": search.get("flow"),
                             "query": search.get("query"), "source_key": search.get("source_key"),
                             "filters": search.get("filters"), "related_keys": positive,
                             "negative_judgments": negative,
                             "no_answer_in_shown_results": sid in no_answer})
    return {"n_searches": len(by_id), "n_events": len(events), "n_final_labels": len(labels),
            "verdicts": dict(Counter(e["verdict"] for e in labels.values())),
            "negative_reasons": dict(reasons),
            "searches_per_negative_reason": {k: len(v) for k, v in queries_by_reason.items()},
            "no_answer_in_shown_results": len(no_answer), "label_changes": dict(changes),
            "note": "Repeated searches/judgments are not independent. Missing judgments are unknown. No training or global precision estimate."}, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=False)
    clean = "data/processed/finetune_split_clean.json"
    sample = "data/processed/paraphrase_rerank_sample500.json"
    cache_path = "data/processed/rerank_k_dev_scores.jsonl"
    keys = load_dev_keys(clean, sample)
    cache_rows = load_jsonl(cache_path)
    cache = {(r["set"], r["key"], r["qv"]): r for r in cache_rows}
    if len(cache) != len(cache_rows):
        raise ValueError("duplicate cache rows")
    paths = [clean, sample, cache_path, "pipeline/retrieval.py", "pipeline/search.py",
             "pipeline/audit_search_robustness.py", "SEARCH_ROBUSTNESS.md"]
    lexical, stages, stress = [], [], []
    for name, (corpus_path, query_path) in SETS.items():
        paths.extend([corpus_path, query_path])
        engine = SearchEngine(load_jsonl(corpus_path), bm25_fold="auto")
        engine.build_bm25()
        queries = [q for q in load_jsonl(query_path) if q["key"] in keys]
        for q in queries:
            relevant = {engine.key_to_idx[k] for k in q["relevant"]}
            for variant, text in variants(q["query"]).items():
                row = {"set": name, "key": q["key"], "variant": variant, "query": text,
                       "groups": slices(q), "same_as_original": text == q["query"],
                       **lexical_row(engine, text, relevant)}
                lexical.append(row)
                if variant in ("goc", "khong_dau"):
                    saved = cache[(name, q["key"], variant)]
                    if saved["query"] != text or set(saved["relevant"]) != set(q["relevant"]):
                        raise ValueError("cache query/label mismatch")
                    stages.append({"set": name, "key": q["key"], "variant": variant,
                                   "query": text, "groups": slices(q),
                                   "lexical_candidates_unchanged": row["candidate_list_unchanged"],
                                   **replay_stage(saved)})
        # Code-only probes have no relevance labels: measure unsupported votes only.
        codes = sorted({m.group() for q in queries
                        for m in re.finditer(r"\b(?:[A-Z]{2,}\d{3,}|[A-Za-z]+Exception)\b", q["query"])})
        for text in codes + [f"ZZQUNSEEN{n}XYZ" for n in range(20)]:
            scores, old = legacy_keyword(engine, text)
            new = engine.search(text, SearchConfig(use_dense=False, top_k=100))
            stress.append({"set": name, "query": text,
                           "type": "synthetic_oov" if text.startswith("ZZQUNSEEN") else "observed_identifier",
                           "positive_docs": int((scores > 0).sum()),
                           "unsupported_candidates_before": int((scores[old] == 0).sum()),
                           "unsupported_candidates_after": int((scores[new] == 0).sum())})
        print(f"{name}: {len(queries)} dev sources, lexical + cached stages done", flush=True)

    # Existing issue judgments are regression-only; do not tune on them.
    corpus_path = "data/processed/corpus.jsonl"
    gold_path = "data/processed/similar_gold_eval.jsonl"
    pool_path = "data/processed/pooling_judgments_v4_llm.csv"
    paths.extend([corpus_path, gold_path, pool_path])
    engine = SearchEngine(load_jsonl(corpus_path), bm25_fold="none")
    engine.build_bm25()
    for name, queries in (("issue_gold", queries_from_gold(gold_path)),
                          ("issue_pooling_v4", queries_from_pooling(pool_path))):
        for q in queries:
            idx = engine.key_to_idx[q["key"]]
            doc = engine.docs[idx]
            text = doc.get("text_dense") or doc.get("summary", "")
            relevant = {engine.key_to_idx[k] for k, grade in q["rel"].items()
                        if grade >= 1 and k in engine.key_to_idx}
            if not relevant:
                raise ValueError("missing positive labels")
            lexical.append({"set": name, "key": q["key"], "variant": "original_issue",
                            "query": text, "groups": slices({"query": text}),
                            **lexical_row(engine, text, relevant, exclude={idx})})
        print(f"{name}: {len(queries)} lexical regression queries done", flush=True)

    # Canonical-equivalence invariants and sensitivity to mixed diacritics are
    # comparisons WITHIN the new engine, separate from before/after bug-fix deltas.
    originals = {(r["set"], r["key"]): r for r in lexical if r["variant"] == "goc"}
    sensitivity = defaultdict(list)
    for row in lexical:
        original = originals.get((row["set"], row["key"]))
        if original and row["variant"] != "goc":
            sensitivity[f"{row['set']}:{row['variant']}"].append({
                "changed_top10": row["after_top10"] != original["after_top10"],
                "delta_recall10": row["after"]["recall@10"] - original["after"]["recall@10"],
                "delta_mrr10": row["after"]["mrr@10"] - original["after"]["mrr@10"]})
    sensitivity_summary = {}
    for name, items in sensitivity.items():
        sensitivity_summary[name] = {"n": len(items),
            "changed_top10": sum(r["changed_top10"] for r in items),
            **{m: {"mean": float(np.mean([r[m] for r in items])),
                   "wins": sum(r[m] > 0 for r in items), "losses": sum(r[m] < 0 for r in items)}
               for m in ("delta_recall10", "delta_mrr10")}}

    search_path, feedback_path = "data/pilot/searches.jsonl", "data/pilot/feedback.jsonl"
    paths.extend([search_path, feedback_path])
    pilot, pilot_failures = pilot_diagnostics(load_jsonl(search_path), load_jsonl(feedback_path))

    report = {"scope": "live lexical regression; historical hybrid/CE dev replay; no fresh neural inference",
              "split": "dev (issue judgments regression only)", "rerank": "top20 blend@0.3",
              "counts": {"lexical_rows": len(lexical), "stage_rows": len(stages), "stress_rows": len(stress)},
              "input_sha256": {p: sha256(p) for p in paths},
              "lexical_slices": summarize_lexical(lexical, args.n_boot),
              "cached_stage_slices": summarize_stages(stages),
              "lexical_variant_sensitivity": sensitivity_summary,
              "pilot_explicit_feedback": pilot,
              "stress": {kind: {"n": len(items),
                                  "queries_with_zero_padding_before": sum(r["unsupported_candidates_before"] > 0 for r in items),
                                  "queries_with_zero_padding_after": sum(r["unsupported_candidates_after"] > 0 for r in items)}
                         for kind in ("observed_identifier", "synthetic_oov")
                         if (items := [r for r in stress if r["type"] == kind])},
              "limitations": ["Historical cache has no full model/corpus fingerprint; hashes are recorded now only.",
                              "No fresh dense/CE scores; unchanged lexical candidates imply unchanged replay inputs, not new model validation.",
                              "Known-item miss does not prove all retrieved documents irrelevant.",
                              "Code-only/OOV probes have no relevance labels; zero lexical evidence is not semantic abstention.",
                              "Variants are synthetic, correlated by source; no combined-variant CI or global acceptance by average.",
                              "Issue gold is incomplete; pooling is predominantly LLM-labelled."]}
    for filename, rows in (("lexical_per_query.jsonl", lexical), ("stages_per_query.jsonl", stages),
                           ("stress_per_query.jsonl", stress), ("pilot_failures.jsonl", pilot_failures)):
        with (dest / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {dest / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
