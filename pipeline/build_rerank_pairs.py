# -*- coding: utf-8 -*-
"""Dung bo huan luyen cho reranker luong issue.

Positive: 133 cap trong `similar_gold_train.jsonl` (lien ket nguoi gan trong Jira,
grade 1/2). Day la tap TRAIN cua gold, tach roi tap eval.

Negative co hai nguon, LUON ghi ro `neg_source` de bao cao tach:
  - `confirmed`: nguoi dung pilot bam "khong lien quan" (verdict=not_related).
    Chac chan dung nhung rat it.
  - `mined`: ung vien xep hang cao trong top-K ma KHONG nam trong lien ket nao.
    Du so luong nhung la GIA DINH - co the dinh cap trung that chua ai lien ket.

Giam rui ro cua `mined` bang ba lop loai tru:
  1. moi issue cung ho (connected component cua do thi cap positive),
  2. moi issue co lien ket Jira truc tiep (`links`, ke ca Cloners),
  3. moi issue nam trong bat ky tap danh gia nao.

Chong ro ri: don vi chia la HO, khong phai cap. Ho nao cham gold_eval, pooling v4
hoac contrastive thi bi loai HOAN TOAN - xem [[contrastive-and-feedback-leak-check]].
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl
from search import FLOWS

SEED = 20260922


def components(pairs: list[tuple[str, str]]) -> dict[str, int]:
    """Gom cac issue noi voi nhau thanh ho; mot ho nam tron trong mot split."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    root_id: dict[str, int] = {}
    out: dict[str, int] = {}
    for k in list(parent):
        r = find(k)
        out[k] = root_id.setdefault(r, len(root_id))
    return out


def eval_keys(args) -> set[str]:
    """Khoa KHONG duoc xuat hien o bat ky phia nao cua mot dong huan luyen.

    Gold_eval va contrastive la cac tap CAP DOI XUNG: moi khoa vua la truy van
    vua la dap an, nen phai cam ca hai phia.

    Pooling chi cam `query_key`. `doc_key` KHONG cam: phep do cham quan he
    (truy van, tai lieu) cu the, nen dung tai lieu D lam negative cho mot truy
    van KHAC khong lam ro ri nhan cua D voi truy van pooling cua no - va mo hinh
    van thay van ban cua D o tang truy xuat. Cam ca doc_key la siet qua tay:
    no da xoa 30/39 negative nguoi dung da xac nhan.
    """
    keys: set[str] = set()
    with open(args.gold_eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys |= {r["a"].upper(), r["b"].upper()}
    with open(args.pooling, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            keys.add(r["query_key"].upper())
    if "contrastive" not in args.retired_sets:
        with open(args.contrastive, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                for c in ("issue_a", "issue_b", "source_a", "source_b"):
                    v = (r.get(c) or "").strip().upper()
                    if v:
                        keys.add(v)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--train-pairs", default="data/processed/similar_gold_train.jsonl")
    ap.add_argument("--gold-eval", default="data/processed/similar_gold_eval.jsonl")
    ap.add_argument("--pooling", default="data/processed/pooling_judgments_v4_llm.csv")
    ap.add_argument("--contrastive", default="data/processed/contrastive_jira_review.csv")
    ap.add_argument("--feedback", default="data/pilot/feedback.jsonl")
    ap.add_argument("--flow", default="issue", choices=["issue", "text"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--neg-per-query", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--retired-sets", default="",
                    help="tap danh gia da NGHI HUU, cho phep dung lam du lieu train. "
                         "Dung 'contrastive' khi bo contrastive chi con la so lieu "
                         "lich su: 6/7 issue nguon cua pilot nam trong do, cam no "
                         "lam mat 30/39 negative nguoi dung da xac nhan. MOT CHIEU: "
                         "da bat thi khong bao gio dung contrastive de nghiem thu nua.")
    ap.add_argument("--out", default="data/processed/rerank_pairs_issue.jsonl")
    args = ap.parse_args()

    docs_list = load_jsonl(args.corpus)
    docs = {d["key"].upper(): d for d in docs_list}

    rows = [json.loads(l) for l in open(args.train_pairs, encoding="utf-8") if l.strip()]
    pairs = [(r["a"].upper(), r["b"].upper()) for r in rows]
    fam = components(pairs)
    banned = eval_keys(args)

    bad_fams = {fam[k] for k in fam if k in banned}
    kept = [(r, a, b) for r, (a, b) in zip(rows, pairs)
            if fam[a] not in bad_fams and fam[b] not in bad_fams]
    print(f"cap positive: {len(rows)} -> giu {len(kept)} "
          f"(loai {len(bad_fams)} ho cham tap danh gia)")

    pos_of: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for r, a, b in kept:
        g = int(r.get("grade", 1))
        pos_of[a][b] = max(pos_of[a].get(b, 0), g)
        pos_of[b][a] = max(pos_of[b].get(a, 0), g)
    queries = sorted(pos_of)
    print(f"truy van train: {len(queries)} | ho: {len({fam[q] for q in queries})}")

    # Negative da nguoi xac nhan, tu pilot. Chi giu dong sach tap danh gia.
    confirmed: dict[str, set[str]] = collections.defaultdict(set)
    if os.path.exists(args.feedback):
        for line in open(args.feedback, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("flow") != args.flow or r.get("verdict") != "not_related":
                continue
            src = (r.get("source_key") or "").strip().upper()
            doc = (r.get("result_key") or "").strip().upper()
            if src and doc and src not in banned and doc not in banned:
                confirmed[src].add(doc)
    print(f"negative da xac nhan: {sum(len(v) for v in confirmed.values())} "
          f"tren {len(confirmed)} issue nguon")

    flow = FLOWS[args.flow]
    dense = DenseIndex(flow["model"], device=args.device, batch_size=args.batch_size,
                       query_prefix=flow["query_prefix"],
                       passage_prefix=flow["passage_prefix"])
    eng = SearchEngine(docs_list, dense=dense, reranker=None, bm25_fold=flow["bm25_fold"])
    eng.build_bm25()
    eng.build_dense(show_progress=False)
    cfg = SearchConfig(use_bm25=True, use_dense=True, rerank=False,
                       candidate_k=args.top_k, top_k=args.top_k)

    # Chia theo SO CAP POSITIVE chu khong theo so ho: ho to nho rat chenh nhau,
    # chia deu so ho lam val phinh len gan mot nua positive.
    rng = random.Random(SEED)
    pos_in_fam: dict[int, int] = collections.Counter()
    for _, a, b in kept:
        pos_in_fam[fam[a]] += 1
    fam_ids = sorted(pos_in_fam)
    rng.shuffle(fam_ids)
    target = sum(pos_in_fam.values()) * args.val_fraction
    val_fams: set[int] = set()
    got = 0
    for f_id in fam_ids:
        if got >= target:
            break
        val_fams.add(f_id)
        got += pos_in_fam[f_id]

    out_rows = []
    stats = collections.Counter()
    for q in sorted(set(queries) | set(confirmed)):
        if q not in docs:
            continue
        split = "val" if fam.get(q, -1) in val_fams else "train"
        qtext = docs[q].get("text_dense") or docs[q].get("summary", "")
        # Ba lop loai tru cho negative khai thac.
        linked = {str(l.get("target", "")).upper()
                  for l in (docs[q].get("links") or []) if l.get("target")}
        same_fam = {k for k, f in fam.items() if f == fam.get(q, -2)}
        protect = set(pos_of.get(q, {})) | linked | same_fam | banned | {q}

        for doc, grade in pos_of.get(q, {}).items():
            if doc in docs:
                out_rows.append({"query_key": q, "doc_key": doc, "label": 1,
                                 "grade": grade, "split": split,
                                 "neg_source": None, "rank": None})
                stats["positive"] += 1

        for doc in confirmed.get(q, ()):
            if doc in docs and doc not in pos_of.get(q, {}):
                out_rows.append({"query_key": q, "doc_key": doc, "label": 0,
                                 "grade": None, "split": split,
                                 "neg_source": "confirmed", "rank": None})
                stats["neg_confirmed"] += 1

        if q in pos_of:
            ex = {eng.key_to_idx[q]} if q in eng.key_to_idx else set()
            ranked = eng.search(qtext, cfg, exclude=ex)
            taken = 0
            for pos, idx in enumerate(ranked, 1):
                key = eng.keys[idx]
                if key in protect or key in confirmed.get(q, ()):
                    continue
                out_rows.append({"query_key": q, "doc_key": key, "label": 0,
                                 "grade": None, "split": split,
                                 "neg_source": "mined", "rank": pos})
                stats["neg_mined"] += 1
                taken += 1
                if taken >= args.neg_per_query:
                    break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n-> {args.out}: {len(out_rows)} dong")
    for k in ("positive", "neg_confirmed", "neg_mined"):
        print(f"  {k:15s} {stats[k]}")
    sp = collections.Counter((r["split"], "pos" if r["label"] else "neg") for r in out_rows)
    print(f"  split: {dict(sp)}")
    qs = {(r["split"], r["query_key"]) for r in out_rows}
    print(f"  truy van: train={sum(1 for s, _ in qs if s == 'train')} "
          f"val={sum(1 for s, _ in qs if s == 'val')}")
    tk = {r["query_key"] for r in out_rows} | {r["doc_key"] for r in out_rows}
    print(f"  KIEM TRA RO RI: giao voi tap danh gia = {len(tk & banned)} (phai la 0)")
    if args.retired_sets:
        print(f"  CANH BAO: da dung tap nghi huu lam train: {args.retired_sets} "
              f"-> khong duoc dung tap do de nghiem thu nua.")


if __name__ == "__main__":
    main()
