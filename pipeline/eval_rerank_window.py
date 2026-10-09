# -*- coding: utf-8 -*-
"""Quet CUA SO rerank (k) x TRONG SO CE tren DEV.

Vi sao can script nay, du da co eval_rerank_k.py: script cu gioi han k <= 30 va
chi doi k voi trong so blend co dinh 0,3. Chan doan ngay 2026-09-20 cho thay
cua so top-20 moi la tran cua pipeline (fusion R@20 = 0,758 trong khi
R@50 = 0,824, R@100 = 0,864 tren paraphrase dev co dau), va cong thuc blend
chuan hoa hang hybrid THEO kich thuoc cua so:

    diem = w * minmax(ce) + (1 - w) * (1 - (hang - 1) / (k - 1))

nen mot tai lieu chi co the vao top-10 khi hang hybrid < 10 + (w/(1-w)) * (k-1).
Voi w = 0,3: k = 50 chi voi toi hang 31, k = 100 chi toi hang 52. Noi k ma giu
nguyen w se khong lay duoc het phan tang => phai quet k va w CUNG LUC.

Script chi ghi bang so de nguoi doc quyet dinh; KHONG sua pipeline/search.py.
Cham CE mot lan cho top-``--head`` roi danh gia moi (k, w) tren cung bo diem do,
nen them cau hinh khong ton them GPU.

Quy tac chon da dat TRUOC khi chay (giu nguyen tinh than eval_rerank_k.py):
- Chi doi cau hinh khi delta Recall@10 so voi (k=20, blend 0,3) duong va CI
  khong chua 0 tren CA HAI lat paraphrase (co dau, khong dau).
- Khong lat nao duoc giam qua 0,01 R@10 hoac MRR@10.
- Neu nhieu cau hinh dat, lay k nho nhat (re nhat khi chay that).
- DEV only. Xac nhan tren test la mot lan chay rieng, sau khi da chot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paraphrase import ci, paired
from eval_rerank_k import SETS, load_dev_keys, read_cache
from eval_rerankers import metrics_from_ranked
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig, SearchEngine,
                       fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS

METRICS = ("recall@10", "mrr@10", "ndcg@10")
BASELINE = (20, 0.3)          # cau hinh pilot dang chay


def evaluate(row: dict, rel: set[str], k: int, weight: float) -> dict:
    """Xep lai top-k bang blend(w) roi do; phan duoi k giu nguyen thu tu hybrid."""
    hybrid = np.asarray(row["hybrid"], dtype=object)
    ce = np.asarray(row["ce"], dtype=float)
    k = min(k, len(hybrid), len(ce))
    ranked = np.concatenate([fuse_rerank(hybrid[:k], ce[:k], "blend", weight),
                             hybrid[k:]])
    return metrics_from_ranked(ranked, rel)


def score_missing(args, cached: dict) -> dict:
    """Cham CE cho top-``head`` cua tung query dev, ghi cache tang dan."""
    rr = CrossEncoderReranker(args.reranker, device=args.device,
                              batch_size=args.ce_batch, fp16=args.device == "cuda")
    wanted = load_dev_keys(args.clean_keys, args.dev_keys)
    specs = []
    shared_model = None
    for set_name in args.sets:
        corpus_path, query_path = SETS[set_name]
        corpus = load_jsonl(corpus_path)
        corpus_keys = {d["key"] for d in corpus}
        queries = [q for q in load_jsonl(query_path) if q["key"] in wanted]
        if args.limit:
            queries = queries[:args.limit]
        variants = [(q, qv, q["query"] if qv == "goc" else strip_diacritics(q["query"]))
                    for q in queries for qv in ("goc", "khong_dau")]
        todo = [(q, qv, text) for q, qv, text in variants
                if (set_name, q["key"], qv) not in cached]
        if todo:
            dense = DenseIndex(args.model, device=args.device, batch_size=64,
                               query_prefix="query: ", passage_prefix="passage: ")
            if shared_model is not None:
                dense._model = shared_model
            eng = SearchEngine(corpus, bm25_fold="auto", dense=dense)
            eng.build_bm25(); eng.build_dense(show_progress=False)
            shared_model = dense._model
            cfg = SearchConfig(name="hybrid", top_k=args.head, **CONFIGS["hybrid"])
            text_of = {d["key"]: d.get("text_dense", "") for d in corpus}
            pending = []
            for q, qv, text in todo:
                rel = {k for k in q["relevant"] if k in corpus_keys}
                if not rel:
                    continue
                keys = [eng.keys[i] for i in eng.search(text, cfg)]
                pending.append({"set": set_name, "key": q["key"], "qv": qv,
                                "query": text, "relevant": sorted(rel), "hybrid": keys})
            os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
            for start in range(0, len(pending), args.score_chunk):
                chunk = pending[start:start + args.score_chunk]
                pairs = [(row["query"], text_of[key])
                         for row in chunk for key in row["hybrid"]]
                scores = rr.score_pairs(pairs, show_progress=False)
                offset = 0
                with open(args.cache, "a", encoding="utf-8") as f:
                    for row in chunk:
                        n = len(row["hybrid"])
                        row["ce"] = [round(float(x), 6) for x in scores[offset:offset + n]]
                        offset += n
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        cached[(row["set"], row["key"], row["qv"])] = row
                print(f"cache {len(cached)} rows", flush=True)
        for q, qv, _ in variants:
            row = cached.get((set_name, q["key"], qv))
            if row is not None:
                specs.append((row, set(row["relevant"])))
        print(f"{set_name}: {len(specs)} luot da co diem CE", flush=True)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 20, 30, 50, 100])
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--head", type=int, default=100, help="so tai lieu duoc cham CE")
    ap.add_argument("--model", default="models/e5-base-task2")
    ap.add_argument("--reranker", default="models/bge-reranker-base-task")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--cache", default="data/processed/rerank_window_dev_scores.jsonl")
    ap.add_argument("--out", default="data/processed/rerank_window_dev.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: so query moi set")
    ap.add_argument("--ce-batch", type=int, default=64)
    ap.add_argument("--score-chunk", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    if max(args.ks) > args.head:
        raise SystemExit(f"--ks khong duoc vuot --head ({args.head})")
    if not all(0.0 < w <= 1.0 for w in args.weights):
        raise SystemExit("--weights phai nam trong (0, 1]")
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} da ton tai; doi ten hoac xoa de chay lai")

    cached = read_cache(args.cache)
    specs = score_missing(args, cached)

    combos = [(k, w) for k in sorted(args.ks) for w in sorted(args.weights)]
    arrays: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row, rel in specs:
        sl = f"{row['set']}:{row['qv']}"
        for k, w in combos:
            for metric, value in evaluate(row, rel, k, w).items():
                arrays[sl][f"k{k}_w{w}"][metric].append(value)
    arrays = {sl: {name: {m: np.asarray(v) for m, v in metrics.items()}
                   for name, metrics in by_combo.items()}
              for sl, by_combo in arrays.items()}

    base = f"k{BASELINE[0]}_w{BASELINE[1]}"
    if any(base not in by_combo for by_combo in arrays.values()):
        raise SystemExit(f"thieu cau hinh moc {base}: them k=20 va w=0.3 vao --ks/--weights")
    quality = {sl: {name: {m: ci(v, args.n_boot) for m, v in metrics.items()}
                    for name, metrics in by_combo.items()}
               for sl, by_combo in arrays.items()}
    paired_vs_base = {sl: {name: {m: paired(metrics[m], by_combo[base][m], args.n_boot)
                                  for m in METRICS}
                           for name, metrics in by_combo.items() if name != base}
                      for sl, by_combo in arrays.items()}

    para = [sl for sl in arrays if sl.startswith("paraphrase")]
    eligible = []
    for k, w in combos:
        name = f"k{k}_w{w}"
        if name == base:
            continue
        gain = all(paired_vs_base[sl][name]["recall@10"]["delta"] > 0
                   and paired_vs_base[sl][name]["recall@10"]["lo"] > 0 for sl in para)
        safe = all(paired_vs_base[sl][name][m]["delta"] >= -0.01
                   for sl in paired_vs_base for m in ("recall@10", "mrr@10"))
        if gain and safe:
            eligible.append(name)
    recommended = min(eligible, key=lambda n: (int(n[1:n.index("_")]), n),
                      default=base)

    report = {"scope": "DEV only; chon cau hinh, chua xac nhan tren test",
              "baseline": base, "head": args.head, "model": args.model,
              "reranker": args.reranker, "n_rows": len(specs),
              "rule": ("doi khi delta R@10 > 0 va CI khong chua 0 tren CA HAI lat "
                       "paraphrase, khong lat nao giam qua 0,01 R@10/MRR@10; "
                       "nhieu cau hinh dat thi lay k nho nhat"),
              "quality": quality, "paired_vs_baseline": paired_vs_base,
              "eligible": eligible, "recommended": recommended}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for sl in sorted(quality):
        print(f"\n== {sl}")
        for k, w in combos:
            name = f"k{k}_w{w}"
            r10 = quality[sl][name]["recall@10"]
            delta = (paired_vs_base[sl][name]["recall@10"]["delta"]
                     if name != base else 0.0)
            mark = "  <- moc" if name == base else ""
            print(f"  k={k:>3} w={w:<4} R@10 {r10['mean']:.4f} "
                  f"[{r10['lo']:.4f}; {r10['hi']:.4f}]  delta {delta:+.4f}{mark}")
    print(f"\ndat tieu chi: {eligible or 'khong co'}")
    print(f"de xuat: {recommended}\n-> {args.out}")


if __name__ == "__main__":
    main()
