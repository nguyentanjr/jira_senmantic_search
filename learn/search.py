# -*- coding: utf-8 -*-
"""
Tim issue lich su tuong dong -- HAI LUONG, dinh tuyen theo dang truy van.

    python pipeline/search.py --query "khong dang nhap duoc sau khi doi mat khau"
    python pipeline/search.py --key RO-11708
    python pipeline/search.py --query RO-11708            # nhan ra ma issue -> luong issue
    python pipeline/search.py --key RO-11708 --rerank on  # ghi de, de thu

DINH TUYEN -- CHOT THEO SO DO NGAY 2026-09-11 (HANDOFF.md §5.4c)
  luong "text"  (ky su go mo ta tu do) -> hybrid + cross-encoder rerank top-20
      paraphrase, 500 truy van sach: R@10 hybrid 0,584 -> 0,634,
      +0,050 [+0,022; +0,078]; vung tu vung khac (jaccard < 0,2) +0,073.
  luong "issue" (goi y tu mot issue co san) -> hybrid, KHONG rerank
      gold, 134 truy van: nDCG@10 hybrid 0,564 -> rerank 0,503,
      -0,060 [-0,121; -0,004].
      dense = e5-task2 fine-tune -- chot 2026-09-14 theo quy tac dat truoc (chon tren
      gold, xac nhan MOT lan tren pooling v4, 39 truy van): hybrid nDCG@10 0,641 -> 0,732
      so voi e5 goc; so voi keyword +0,115 [+0,056; +0,178].

  luong "text" con dung BM25 them ban BO DAU (retrieval.tokenize fold="add"),
      chot 2026-09-14 (eval_bm25_fold.py, dev 500): truy van go khong dau
      hybrid+rerank R@10 0,480 -> 0,566; truy van co dau khong giam.
  luong "text" TRON diem cross-encoder (30%) voi thu tu hybrid (70%) thay vi de
      CE thay han thu tu -- chot 2026-09-14 (select_rerank_fusion.py, chon tren dev,
      test 2.675 mot lan): khong dau MRR@10 +0,041 / R@10 +0,026 so voi replace;
      co dau MRR@10 +0,014, R@10 -0,005 (khong co y nghia).

Tien to e5 giu DUNG cach da do: text = 'query: ' / 'passage: ' (bat doi xung,
nhu eval_paraphrase.py); issue = 'query: ' ca hai phia (nhu eval_similar.py).
Doi tien to la dung mot cau hinh CHUA duoc do.

PHAM VI: dung retrieval.SearchEngine -- chinh he thong da duoc do. Pipeline dup
(run_dup_e2e.py --mode search) co reranker rieng (chi thay phan base roi ap lai
bonus/penalty); no duoc do rieng theo luong (eval_dup_flows.py, 2026-09-12) va
dinh tuyen cung chieu: text -> bat doi xung + rerank, issue -> khong rerank.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution                                                     # noqa: E402
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,  # noqa: E402
                       SearchEngine, load_jsonl)
from run_ablation import CONFIGS, E5_PASSAGE, E5_QUERY                # noqa: E402

ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")
DEFAULT_RERANK_K = 20

FLOWS = {
    "text": {
        "config": "hybrid+rerank",
        "query_prefix": E5_QUERY, "passage_prefix": E5_PASSAGE,
        # BM25 them ban bo dau (2026-09-14, eval_bm25_fold.py, dev 500):
        # hybrid+rerank truy van khong dau R@10 0,480 -> 0,566 (+0,086 co y nghia),
        # truy van co dau 0,634 -> 0,636 (khong giam).
        # auto: truy van co dau -> chi muc thuong; khong dau -> chi muc them ban bo dau.
        # `add` cho moi truy van lam hong truy van NGAN co dau (viet tay: keyword
        # 0,620 -> 0,440); ket qua truy van khong dau giu nguyen nhu da do voi add.
        "bm25_fold": "auto",
        # Tron diem CE voi thu tu hybrid thay vi de CE thay han (2026-09-14,
        # select_rerank_fusion.py: chon blend@0.3 tren dev, kiem MOT lan tren test 2.675):
        # khong dau MRR@10 +0,041, R@10 +0,026; co dau R@10 -0,005 (ns), MRR +0,014.
        "rerank_fusion": "blend", "rerank_weight": 0.3,
        # Chon tren 4 lat dev ngay 2026-09-15: top-20 dat nguong khong giam
        # qua 0,01 R@10/MRR@10 tren moi lat, va giam CPU p50 30,7% so voi top-30.
        "rerank_k": DEFAULT_RERANK_K,
        # Lan thu cuoi (2026-09-14, final_eval.py: chon tren dev, test 2.675 mot lan):
        # e5 + reranker fine-tune tren cap "mo ta -> issue" (build_task_pairs.py).
        # Test so voi keyword: co dau R@10 +0,134 / MRR +0,111; khong dau +0,087 / +0,051.
        # Vong 2 (2026-09-14, final_eval2.py, chon tren dev 2 bo do, test mot lan):
        # e5-base train them cap "tieu de -> issue" (task_pairs_v2). So voi e5-task|ce-task|0.3
        # tren test: tieu de R@10 +0,060 (co dau) / +0,066 (khong dau); paraphrase -0,006 (ns) / +0,010.
        "model": "models/e5-base-task2", "reranker": "models/bge-reranker-base-task",
        "why": "mo ta tu do: rerank +0,050 R@10 so voi hybrid (paraphrase, co y nghia); "
               "BM25 bo dau cho truy van go khong dau; tron CE 30% voi thu tu hybrid",
    },
    "issue": {
        "config": "hybrid",
        "query_prefix": E5_QUERY, "passage_prefix": E5_QUERY,
        "bm25_fold": "none",       # van ban issue luon co dau; chua do luong nay
        "rerank_fusion": "replace", "rerank_weight": 0.5,   # chi dung khi --rerank on
        "rerank_k": DEFAULT_RERANK_K,  # chi co hieu luc khi ghi de --rerank on
        # Model chon theo quy tac chot TRUOC khi do (2026-09-14, HANDOFF §7 "Issue flow:
        # model selection"): chon tren gold 134 -- e5-task2 hybrid nDCG@10 +0,043, R@10 +0,071
        # so voi e5 goc (co y nghia); xac nhan MOT lan tren pooling v4 da mo rong (39 truy van):
        # nDCG@10 +0,091, R@10 +0,116 so voi e5 goc; so voi keyword +0,115 / +0,104 (co y nghia).
        # Reranker chi dung khi --rerank on (chua do voi model nay).
        "model": "models/e5-base-task2", "reranker": "BAAI/bge-reranker-base",
        "why": "tu mot issue: hybrid e5-task2 (pooling v4: nDCG@10 +0,115 so voi keyword); "
               "rerank -0,060 nDCG@10 so voi hybrid (gold, co y nghia) -> tat",
    },
}


def plan(query: str | None = None, key: str | None = None,
         rerank: str = "auto") -> dict:
    """Chon luong + cau hinh. Ham thuan de test duoc.

    `query` trung dang ma issue ("RO-11708") duoc coi la `key`: go ma issue vao
    o tim kiem la muon tim issue tuong tu voi no, khong phai tim chuoi ky tu do.
    `rerank`: auto (theo luong) / on / off (ghi de -- dung cau hinh chua do).
    """
    query = (query or "").strip()
    key = (key or "").strip()
    if query and not key and ISSUE_KEY_RE.match(query):
        key, query = query, ""
    if bool(query) == bool(key):
        raise ValueError("can dung MOT trong --query hoac --key")
    if rerank not in ("auto", "on", "off"):
        raise ValueError(f"rerank khong hop le: {rerank}")
    flow = "issue" if key else "text"
    f = FLOWS[flow]
    name = {"auto": f["config"], "on": "hybrid+rerank", "off": "hybrid"}[rerank]
    return {"flow": flow, "config": name, "query": query, "key": key,
            "query_prefix": f["query_prefix"], "passage_prefix": f["passage_prefix"],
            "bm25_fold": f["bm25_fold"],
            "rerank_fusion": f["rerank_fusion"], "rerank_weight": f["rerank_weight"],
            "rerank_k": f["rerank_k"],
            "model": f["model"], "reranker": f["reranker"],
            "measured": rerank == "auto", "why": f["why"]}


def _short(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n].rstrip() + " …"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="", help="mo ta tu do (hoac ma issue)")
    ap.add_argument("--key", default="", help="ma issue co san, vd RO-11708")
    ap.add_argument("--rerank", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rerank-k", type=int, default=None,
                    help=f"so ung vien CE cham; mac dinh theo luong ({DEFAULT_RERANK_K})")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--model", default="", help="ghi de model dense cua luong (chua do)")
    ap.add_argument("--reranker", default="", help="ghi de reranker cua luong (chua do)")
    args = ap.parse_args()

    try:
        pl = plan(args.query, args.key, args.rerank)
    except ValueError as e:
        sys.exit(str(e))
    model = args.model or pl["model"]
    rr_name = args.reranker or pl["reranker"]
    for p in (model, rr_name):
        if p.startswith("models/") and not os.path.exists(os.path.join(p, "config.json")):
            sys.exit(f"thieu trong so model {p} (chi co task_meta.json?). Tai model tu Colab "
                     f"ve dung thu muc nay, hoac train lai: pipeline/train_task_models.py")

    docs = load_jsonl(args.corpus)
    by_key = {d["key"]: i for i, d in enumerate(docs)}
    exclude = None
    if pl["flow"] == "issue":
        if pl["key"] not in by_key:
            sys.exit(f"{pl['key']} khong co trong corpus ({args.corpus})")
        src = docs[by_key[pl["key"]]]
        qtext = src.get("text_dense") or src.get("summary", "")
        exclude = {by_key[pl["key"]]}
    else:
        qtext = pl["query"]

    rerank_k = pl["rerank_k"] if args.rerank_k is None else args.rerank_k
    if CONFIGS[pl["config"]]["rerank"] and rerank_k < args.top_k:
        sys.exit("--rerank-k phai lon hon hoac bang --top-k")
    cfg = SearchConfig(name=pl["config"], top_k=args.top_k, rerank_k=rerank_k,
                       rerank_fusion=pl["rerank_fusion"], rerank_weight=pl["rerank_weight"],
                       **CONFIGS[pl["config"]])
    dense = DenseIndex(model, device="cpu", batch_size=32,
                       query_prefix=pl["query_prefix"], passage_prefix=pl["passage_prefix"])
    texts = [d.get("text_dense", "") for d in docs]
    if not os.path.exists(dense._cache_path(texts, "corpus")):
        print("chua co cache embedding cho tien to nay -- lan dau se nhung toan bo "
              f"{len(docs)} issue tren CPU (lau), cac lan sau doc tu cache.")
    reranker = CrossEncoderReranker(rr_name) if cfg.rerank else None
    eng = SearchEngine(docs, dense=dense, reranker=reranker, bm25_fold=pl["bm25_fold"])
    eng.build_bm25()
    eng.build_dense(show_progress=True)
    ranked = eng.search(qtext, cfg, exclude=exclude)

    head = f"[{pl['key']}] {_short(docs[by_key[pl['key']]].get('summary', ''), 120)}" \
        if pl["flow"] == "issue" else f'"{_short(pl["query"], 120)}"'
    print(f"\nTRUY VAN: {head}")
    print(f"luong={pl['flow']} | cau hinh={pl['config']}"
          + ("" if pl["measured"] else "  (GHI DE -- cau hinh nay CHUA duoc do cho luong nay)"))
    print(f"ly do: {pl['why']}\n")
    for rank, i in enumerate(ranked, 1):
        d = eng.docs[i]
        res = resolution.extract(d.get("text_comments", ""))
        print(f"{rank:>2}. {d['key']}  [{d.get('status', '')}/{d.get('resolution', '')}]  "
              f"{_short(d.get('summary', ''), 110)}")
        if res:
            print(f"    xu ly da ghi nhan: {_short(res, 220)}")
        print(f"    {d.get('url', '')}")


if __name__ == "__main__":
    main()
