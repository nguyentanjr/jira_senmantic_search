"""One fixed, grouped development experiment; uses only explicit AI preferences."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np

from retrieval import DenseIndex, SearchEngine, SearchConfig, load_jsonl
import embedding_cache


def connected_groups(triples, judgments):
    parent = {}
    def find(k):
        parent.setdefault(k, k)
        if parent[k] != k:
            parent[k] = find(parent[k])
        return parent[k]
    def join(keys):
        roots = [find(k) for k in keys]
        for root in roots:
            parent[root] = min(roots)
    for r in triples:
        join([r['key'], r['positive_key'], *r['neg_keys']])
    for r in judgments:
        join([r['query_key'], r['candidate_key']])
    return {k: find(k) for k in parent}


def fit_adapter(emb, index, preferences, alpha=1., margin=.05):
    """Fit score = cosine + sqrt(d) * (q*d) @ w; ridge penalty on w.

    Weight every anchor equally regardless of its number of negative pairs.
    Solve the small dual system; no implicit negatives or train-time centering.
    """
    if not preferences:
        raise ValueError('No training preferences outside held-out component')
    counts = defaultdict(int)
    for p in preferences:
        counts[p['query_key']] += 1
    differences = np.array([emb[index[p['query_key']]] *
        (emb[index[p['positive_key']]] - emb[index[p['negative_key']]])
        for p in preferences], dtype=np.float64)
    target = margin - differences.sum(axis=1)
    scale = np.sqrt(emb.shape[1])
    weight = np.sqrt([1. / counts[p['query_key']] for p in preferences])
    x = differences * scale * weight[:, None]
    y = target * weight
    return x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)


class FrozenDense:
    def __init__(self, emb, texts):
        self.emb = emb
        self.lookup = {t: i for i, t in enumerate(texts)}
        self.multiplier = np.ones(emb.shape[1], dtype=np.float32)
    def encode(self, texts, is_query):
        assert is_query
        return self.emb[[self.lookup[t] for t in texts]]
    def score(self, query):
        # Bilinear retrieval score, not a renormalized cosine or probability.
        return self.emb @ (query * self.multiplier)
    def set_adapter(self, w):
        self.multiplier = (1. + np.sqrt(self.emb.shape[1]) * w).astype(np.float32)


def metrics(ranking, labels):
    positive = {k for k, g in labels.items() if g > 0}
    top = ranking[:10]
    return dict(known_recall10=sum(k in positive for k in top)/len(positive) if positive else None,
        known_recall100=sum(k in positive for k in ranking[:100])/len(positive) if positive else None,
        judged_fraction10=sum(k in labels for k in top)/len(top) if top else 0.,
        known_zero_count10=sum(labels.get(k) == 0 for k in top))


def judged_ndcg(keys, labels):
    # Only the explicitly judged pool is ranked. Not corpus-wide nDCG.
    gains = [2**labels[k]-1 for k in keys]
    ideal = sorted(gains, reverse=True)
    discount = 1 / np.log2(np.arange(len(keys)) + 2)
    denom = float(np.dot(ideal, discount))
    return float(np.dot(gains, discount)/denom) if denom else None


def mean_fields(rows):
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', default='configs/relation_adapter_probe_20260923.json')
    ap.add_argument('--preferences', default='data/processed/relation_codex_review_20260923/ai_development_preferences.jsonl')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (out/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    review = Path('data/processed/relation_codex_review_20260923')
    judgments = load_jsonl(str(review/'judgments.jsonl'))
    preferences = load_jsonl(args.preferences)
    by_id = {j['review_id']: j for j in judgments}
    for p in preferences:
        for col, grade, keycol in [('positive_review_id', p['positive_grade'], 'positive_key'), ('negative_review_id', 0, 'negative_key')]:
            j = by_id[p[col]]
            assert j['grade'] == grade and j['confidence'] in protocol.get('allowed_confidence', ['high'])
            if col == 'positive_review_id':
                assert grade in protocol.get('allowed_positive_grades', [2])
            assert (j['query_key'], j['candidate_key']) == (p['query_key'], p[keycol])
    docs = load_jsonl('data/processed/corpus.jsonl')
    index = {d['key']: i for i, d in enumerate(docs)}
    texts = [d.get('text_dense', '') for d in docs]
    dense = DenseIndex('models/e5-base-task2', query_prefix='query: ', passage_prefix='query: ')
    identity = dense.cache_identity(texts)
    cache_path = embedding_cache.cache_path('data/cache', 'corpus', identity)
    emb = embedding_cache.load(cache_path, identity)
    if emb is None:
        raise RuntimeError('No matching verified cache; do not silently recompute corpus')
    assert np.allclose(np.linalg.norm(emb, axis=1), 1., atol=1e-5)
    # Same text/prefix on both sides lets the issue query reuse its corpus vector.
    frozen = FrozenDense(emb, texts)
    for q in {j['query_key'] for j in judgments}:
        assert np.array_equal(frozen.encode([texts[index[q]]], True)[0], emb[index[q]])
    groups = connected_groups(load_jsonl('data/processed/direction_pairs_v2_clean.jsonl'), judgments)
    by_query = defaultdict(dict)
    for j in judgments:
        by_query[j['query_key']][j['candidate_key']] = j['grade']
    engine = SearchEngine(docs, dense=frozen, bm25_fold='none')
    engine.build_bm25()
    config = SearchConfig(top_k=100)
    records, folds, preference_results = [], [], []
    for group in sorted({groups[q] for q in by_query}):
        train = [p for p in preferences if groups[p['query_key']] != group]
        held = [p for p in preferences if groups[p['query_key']] == group]
        train_keys = {p[k] for p in train for k in ('query_key','positive_key','negative_key')}
        test_keys = {k for q, labels in by_query.items() if groups[q] == group for k in [q, *labels]}
        assert train_keys.isdisjoint(test_keys)
        w = fit_adapter(emb, index, train, protocol['ridge_alpha'], protocol['target_margin'])
        np.save(out/f'heldout_{group}_adapter.npy', w, allow_pickle=False)
        fold = dict(group_id=group, training_preferences=len(train), heldout_preferences=len(held),
                    train_keys=sorted(train_keys), heldout_keys=sorted(test_keys), adapter_norm=float(np.linalg.norm(w)))
        for q, labels in sorted(by_query.items()):
            if groups[q] != group:
                continue
            r = dict(query_key=q, group_id=group, known_positives=sum(g>0 for g in labels.values()))
            for mode, weights in [('baseline', np.zeros_like(w)), ('adapter', w)]:
                frozen.set_adapter(weights)
                start = time.perf_counter()
                ranking = engine.search(texts[index[q]], config, exclude={index[q]})
                elapsed = (time.perf_counter()-start)*1000
                keys = [docs[i]['key'] for i in ranking]
                scores = frozen.score(emb[index[q]])
                pool = sorted(labels, key=lambda k: (-float(scores[index[k]]), k))
                r[mode] = metrics(keys, labels)
                r[mode]['judged_pool_dense_ndcg'] = judged_ndcg(pool, labels)
                r[mode+'_ranking'] = keys
                r[mode+'_search_ms'] = elapsed
            records.append(r)
        for p in held:
            difference = emb[index[p['query_key']]]*(emb[index[p['positive_key']]]-emb[index[p['negative_key']]])
            baseline = float(difference.sum())
            adapted = baseline + float(np.sqrt(emb.shape[1])*difference @ w)
            preference_results.append({**p, 'baseline_margin': baseline, 'adapter_margin': adapted,
                'baseline_correct': float(baseline > 0) + .5*float(baseline == 0),
                'adapter_correct': float(adapted > 0) + .5*float(adapted == 0)})
        local = [r for r in records if r['group_id'] == group]
        fold['metrics'] = {m: mean_fields([r[m] for r in local]) for m in ('baseline','adapter')}
        folds.append(fold)
        print(json.dumps({k:v for k,v in fold.items() if not k.endswith('keys')}), flush=True)
    overall = {m: mean_fields([r[m] for r in records]) for m in ('baseline','adapter')}
    accuracy = {}
    for m in ('baseline', 'adapter'):
        per_query = defaultdict(list)
        for p in preference_results:
            per_query[p['query_key']].append(p[m+'_correct'])
        accuracy[m] = float(np.mean([np.mean(v) for v in per_query.values()]))
    b, a = overall['baseline'], overall['adapter']
    passed = (a['known_recall10'] > b['known_recall10'] and a['known_recall100'] >= b['known_recall100']
        and accuracy['adapter'] > accuracy['baseline'] and all(
        f['metrics']['adapter']['known_recall10'] >= f['metrics']['baseline']['known_recall10']
        for f in folds if f['metrics']['baseline']['known_recall10'] is not None))
    summary = dict(protocol=protocol, queries=len(records), components=len(folds), preferences=len(preferences),
        positive_queries=sum(r['known_positives']>0 for r in records), metrics=overall,
        preference_accuracy_macro_query=accuracy, passes_research_gate=bool(passed), deployed=False,
        cache_identity=identity, cache_sha256=embedding_cache.file_hash(cache_path),
        limitations=['AI development labels, not independent test', 'Incomplete pool: unjudged is unknown, not grade 0',
                    'Few connected components; no claim of statistical significance',
                    'Frozen query cache timing excludes embedding inference'],
        sources_sha256={str(p): embedding_cache.file_hash(p) for p in [protocol_path, review/'judgments.jsonl',
            Path(args.preferences), Path('data/processed/corpus.jsonl'),
            Path('data/processed/direction_pairs_v2_clean.jsonl'), Path(__file__), Path('pipeline/retrieval.py')]})
    for name, rows in [('query_results',records), ('folds',folds), ('preference_results',preference_results)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ('metrics','preference_accuracy_macro_query','passes_research_gate','queries','components','positive_queries')},indent=2))


if __name__ == '__main__':
    main()
