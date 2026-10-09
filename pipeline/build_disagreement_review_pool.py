"""Build a fresh development pool from full-text/defect-only disagreements.

Labels, ranks and branch provenance stay out of the review/dossier files.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from build_feedback_pairs import existing_eval_keys
from build_relation_review import context
from retrieval import BM25, DenseIndex, SearchConfig, SearchEngine, load_jsonl, tokenize, top_k_idx
import embedding_cache

TAG_RE=re.compile(r"\[([^\]]{1,80})\]")
GENERIC={'cms','api','ussd','app','web','webview','eu','agent','customer','client','operation','fe','be','ba','production','pre','staging'}

class FrozenDense:
    def __init__(self,emb,texts): self.emb=emb; self.lookup={t:i for i,t in enumerate(texts)}
    def encode(self,texts,is_query): return self.emb[[self.lookup[t] for t in texts]]
    def score(self,q): return self.emb@q

def family(doc):
    tags=[]
    for tag in TAG_RE.findall(doc.get('summary','')):
        words=[w for w in re.findall(r'[a-z0-9]+',tag.lower()) if w not in GENERIC]
        if words: tags.append('_'.join(words))
    if tags: return '+'.join(tags[:2])
    words=[w for w in re.findall(r'[a-z0-9]+',doc.get('summary','').lower()) if w not in GENERIC and len(w)>2]
    return '_'.join(words[:3]) or doc['key']

def defect_text(doc):
    return '\n'.join(x for x in (doc.get('summary',''),(doc.get('sections') or {}).get('actual','')) if x).strip()

def rid(q,c): return hashlib.sha256((q+'\0'+c).encode()).hexdigest()[:16]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--anchors',type=int,default=20)
    ap.add_argument('--per-branch',type=int,default=2); ap.add_argument('--shared',type=int,default=2); args=ap.parse_args()
    out=Path(args.out)
    if out.exists(): raise SystemExit('Output exists; use a new directory')
    docs=load_jsonl('data/processed/corpus.jsonl'); idx={d['key']:i for i,d in enumerate(docs)}; texts=[d.get('text_dense','') for d in docs]
    protected_paths=['data/processed/similar_gold_eval.jsonl','data/processed/pooling_judgments_v4_llm.csv',
        'data/processed/contrastive_jira_review.csv','data/processed/finetune_split_clean.json']
    excluded=existing_eval_keys(*protected_paths)
    extra_paths=['data/processed/similar_gold_train.jsonl','data/processed/gold_train_pairs.jsonl',
                 'data/processed/direction_pairs_v2_clean.jsonl','data/processed/relation_codex_review_20260923/judgments.jsonl']
    extra_paths += [
        'data/processed/disagreement_review_pool_20260923/judgments.jsonl',
        'data/processed/relation_reranker_failure_review_20260923/judgments.jsonl',
        'data/processed/relation_reranker_symmetric_top10_review_20260923/judgments.jsonl',
        'data/processed/relation_reranker_v2_symmetric_top10_review_20260923/judgments.jsonl',
    ]
    for p in extra_paths:
        for row in load_jsonl(p):
            for col in ('a','b','key','positive_key','query_key','candidate_key'):
                if row.get(col): excluded.add(str(row[col]).upper())
            excluded.update(str(k).upper() for k in row.get('neg_keys',[]))
    dense=DenseIndex('models/e5-base-task2',query_prefix='query: ',passage_prefix='query: ')
    ident=dense.cache_identity(texts); cache=embedding_cache.cache_path('data/cache','corpus',ident)
    emb=embedding_cache.load(cache,ident)
    if emb is None: raise SystemExit('Verified issue corpus cache missing')
    frozen=FrozenDense(emb,texts); engine=SearchEngine(docs,dense=frozen,bm25_fold='none'); engine.build_bm25()
    defect=[defect_text(d) for d in docs]; dbm=BM25([tokenize(t) for t in defect])
    eligible=[i for i,d in enumerate(docs) if d['key'] not in excluded
              and len((d.get('sections') or {}).get('actual','').strip())>=35
              and len(defect[i])>=70]
    # Stable candidate anchors across runs, but inspect every eligible issue before selection.
    candidates=[]; cfg=SearchConfig(top_k=20)
    for n,qi in enumerate(eligible):
        key=docs[qi]['key']; base=engine.search(texts[qi],cfg,exclude={qi}); bkeys=[docs[i]['key'] for i in base]
        s=dbm.score(tokenize(defect[qi])); dr=top_k_idx(s,21); dr=dr[(dr!=qi)&(s[dr]>0)][:20]; dkeys=[docs[i]['key'] for i in dr]
        b=[k for k in bkeys if k not in excluded]; d=[k for k in dkeys if k not in excluded]
        bs=[k for k in b if k not in set(d)]; ds=[k for k in d if k not in set(b)]; sh=[k for k in b if k in set(d)]
        if len(bs)>=args.per_branch and len(ds)>=args.per_branch and len(sh)>=args.shared:
            jac=len(set(b[:10])&set(d[:10]))/max(1,len(set(b[:10])|set(d[:10])))
            candidates.append((jac,family(docs[qi]),key,qi,b,d,bs,ds,sh))
    # Prefer largest disagreement; enforce unique family, no reused candidate, and low anchor cosine.
    selected=[]; used=set(); families=set()
    for item in sorted(candidates,key=lambda x:(x[0],hashlib.sha256(x[2].encode()).hexdigest())):
        jac,fam,key,qi,b,d,bs,ds,sh=item
        if fam in families or key in used: continue
        if any(float(emb[qi]@emb[x[3]])>.72 for x in selected): continue
        picks=[]
        for source in (bs[:args.per_branch],ds[:args.per_branch],sh[:args.shared]):
            available=[k for k in source if k not in used and k!=key]
            if len(available)<len(source): break
            picks.extend(available)
        if len(picks)!=2*args.per_branch+args.shared or len(set(picks))!=len(picks): continue
        selected.append(item); families.add(fam); used.update([key,*picks])
        if len(selected)==args.anchors: break
    if len(selected)<args.anchors: raise SystemExit(f'Only {len(selected)} independent anchors found')
    reviews=[]; side=[]
    for jac,fam,key,qi,b,d,bs,ds,sh in selected:
        groups=[('full_text_only',bs[:args.per_branch]),('defect_only',ds[:args.per_branch]),('shared',sh[:args.shared])]
        picks=[k for _,ks in groups for k in ks]
        # Stable blind order unrelated to branch/rank.
        picks.sort(key=lambda k:hashlib.sha256((key+'\0'+k).encode()).hexdigest())
        for cand in picks:
            review_id=rid(key,cand)
            reviews.append({'review_id':review_id,'query_key':key,'candidate_key':cand,
                'query_context':context(docs[qi]),'candidate_context':context(docs[idx[cand]]),
                'same_context':'','grade':'','reason':'','reviewer':'','notes':''})
            source=next(name for name,ks in groups if cand in ks)
            side.append({'review_id':review_id,'anchor_family':fam,'proposal_source':source,
                'full_text_rank':b.index(cand)+1 if cand in b else None,
                'defect_rank':d.index(cand)+1 if cand in d else None,
                'top10_jaccard':jac,'use':'development_review_only','label_source':None})
    out.mkdir(parents=True)
    cols=list(reviews[0])
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(reviews)
    (out/'provenance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in side),encoding='utf-8')
    # Dossiers only expose pair identities and full contexts, never retrieval source/rank.
    byq=defaultdict(list)
    for r in reviews: byq[r['query_key']].append(r['candidate_key'])
    docmap={d['key']:d for d in docs}; seen=set()
    for start in range(0,len(selected),4):
        keys=[x[2] for x in selected[start:start+4]]
        parts=['PAIRS TO JUDGE',*[q+': '+', '.join(byq[q]) for q in keys],'\nDOCUMENTS']
        for key in [k for q in keys for k in [q,*byq[q]]]:
            if key in seen: continue
            seen.add(key); parts.extend([f'\n--- {key} ---',context(docmap[key])])
        (out/f'dossier_{start//4+1}.txt').write_text('\n'.join(parts),encoding='utf-8')
    manifest={'anchors':len(selected),'pairs':len(reviews),'unique_documents':len(seen),
        'families':[{'key':x[2],'family':x[1],'top10_jaccard':x[0]} for x in selected],
        'selection':'lowest top10 Jaccard; unique family; pairwise anchor cosine <= 0.72; no document reused; 2 full-only + 2 defect-only + 2 shared',
        'excluded_documents':len(excluded),'labels_blank':True,'human_verified':False,
        'scope':'new AI development review pool, not independent test after grading by same agent',
        'cache_identity':ident,'cache_sha256':embedding_cache.file_hash(cache),
        'sources_sha256':{p:sha(p) for p in [*protected_paths,*extra_paths,'data/processed/corpus.jsonl',__file__]}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('anchors','pairs','unique_documents','excluded_documents','families')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
