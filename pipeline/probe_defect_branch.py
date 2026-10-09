"""One fixed auxiliary BM25 branch over summary + observed actual behavior."""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from retrieval import BM25, DenseIndex, SearchConfig, SearchEngine, load_jsonl, rrf_fuse, tokenize, top_k_idx
import embedding_cache
from probe_structured_rank import connected_groups, metrics, mean

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def defect_text(doc):
    actual=(doc.get('sections') or {}).get('actual','')
    return '\n'.join(x for x in (doc.get('summary',''),actual) if x).strip()

def fuse_rankings(primary, auxiliary, n_docs, k=60., auxiliary_weight=.5, limit=100):
    score=rrf_fuse([np.asarray(primary,dtype=np.int32),np.asarray(auxiliary,dtype=np.int32)],
                   n_docs,k=k,weights=[1.,auxiliary_weight])
    union=np.asarray(list(dict.fromkeys([*primary,*auxiliary])),dtype=np.int32)
    order=np.argsort(-score[union],kind='stable')
    return union[order][:limit]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    protocol_path=Path('configs/defect_branch_probe_20260923.json')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    review=Path('data/processed/relation_codex_review_20260923')
    judgments=load_jsonl(str(review/'judgments.jsonl')); triples=load_jsonl('data/processed/direction_pairs_v2_clean.jsonl')
    groups=connected_groups(triples,judgments); labels=defaultdict(dict)
    for j in judgments: labels[j['query_key']][j['candidate_key']]=j['grade']
    docs=load_jsonl('data/processed/corpus.jsonl'); index={d['key']:i for i,d in enumerate(docs)}
    texts=[d.get('text_dense','') for d in docs]
    dense=DenseIndex('models/e5-base-task2',query_prefix='query: ',passage_prefix='query: ')
    ident=dense.cache_identity(texts); cache=embedding_cache.cache_path('data/cache','corpus',ident)
    emb=embedding_cache.load(cache,ident)
    if emb is None: raise RuntimeError('Matching verified corpus cache required')
    dense.emb=emb; engine=SearchEngine(docs,dense=dense,bm25_fold='none'); engine.build_bm25()
    baseline_cfg=SearchConfig(top_k=100)
    defect=[defect_text(d) for d in docs]; defect_index=BM25([tokenize(t,fold='none') for t in defect])
    rows=[]
    for q in sorted(labels):
        qi=index[q]
        primary=engine.search(texts[qi],baseline_cfg,exclude={qi})
        scores=defect_index.score(tokenize(defect[qi],fold='none'))
        auxiliary=top_k_idx(scores,101); auxiliary=auxiliary[(auxiliary!=qi)&(scores[auxiliary]>0)][:100]
        fused=fuse_rankings(primary,auxiliary,len(docs))
        pkeys=[docs[i]['key'] for i in primary]; fkeys=[docs[i]['key'] for i in fused]
        rows.append({'query_key':q,'group_id':groups[q],'baseline':metrics(pkeys,labels[q]),
            'defect_branch':metrics(fkeys,labels[q]),'baseline_ranking':pkeys,
            'defect_branch_ranking':fkeys,'auxiliary_ranking':[docs[i]['key'] for i in auxiliary]})
    modes=('baseline','defect_branch'); fields=('known_recall10','known_recall100','known_zero_count10','judged_fraction10','judged_pool_ndcg')
    overall={m:{f:mean(rows,m,f) for f in fields} for m in modes}; folds=[]
    for g in sorted({r['group_id'] for r in rows}):
        subset=[r for r in rows if r['group_id']==g]
        folds.append({'group_id':g,'queries':len(subset),'metrics':{m:{f:mean(subset,m,f) for f in fields} for m in modes}})
    b,a=overall['baseline'],overall['defect_branch']
    passed=(a['known_recall10']>b['known_recall10'] and a['known_recall100']>=b['known_recall100']
        and a['known_zero_count10']<=b['known_zero_count10'] and a['judged_pool_ndcg']>=b['judged_pool_ndcg']
        and all(f['metrics']['defect_branch']['known_recall10']>=f['metrics']['baseline']['known_recall10']
                for f in folds if f['metrics']['baseline']['known_recall10'] is not None))
    changes=[]
    for r in rows:
        for k,g in labels[r['query_key']].items():
            br=r['baseline_ranking'].index(k)+1 if k in r['baseline_ranking'] else None
            nr=r['defect_branch_ranking'].index(k)+1 if k in r['defect_branch_ranking'] else None
            if br!=nr: changes.append({'query_key':r['query_key'],'candidate_key':k,'grade':g,'baseline_rank':br,'defect_branch_rank':nr})
    report={'queries':len(rows),'positive_queries':sum(any(v>0 for v in x.values()) for x in labels.values()),
        'components':len(folds),'metrics':overall,'folds':folds,'passes_research_gate':bool(passed),'deployed':False,
        'changed_judged_pairs':len(changes),'cache_identity':ident,'cache_sha256':embedding_cache.file_hash(cache),
        'limitations':['AI development labels, not independent test','Unjudged corpus items are unknown','One sequential exploratory hypothesis after prior failures'],
        'sources_sha256':{str(p):sha(p) for p in [protocol_path,review/'judgments.jsonl',Path('data/processed/corpus.jsonl'),Path('data/processed/direction_pairs_v2_clean.jsonl'),Path(__file__),Path('pipeline/retrieval.py')]}}
    (out/'query_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (out/'changed_judged_pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changes),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('queries','metrics','folds','passes_research_gate','changed_judged_pairs')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
