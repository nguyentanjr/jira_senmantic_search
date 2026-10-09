# -*- coding: utf-8 -*-
"""CLI cho he thong Duplicate / Similar Bug Retrieval.

    # tim bug tuong tu voi mot issue da co
    python pipeline/run_dup_e2e.py --mode search --query-key RO-10006

    # tim bang mo ta tu do
    python pipeline/run_dup_e2e.py --mode search --query "khong dang nhap duoc sau khi doi mat khau"

    # do tren tap gold (199 cap ky su tu ghi trong comment)
    python pipeline/run_dup_e2e.py --mode eval

    # ablation tung phase
    python pipeline/run_dup_e2e.py --mode ablation

    # soat ket qua: trung ro rang / false positive / hard negative / bo sot
    python pipeline/run_dup_e2e.py --mode audit --limit 25

    # kiem tra xu ly mau thuan tren cac vi du chuan
    python pipeline/run_dup_e2e.py --mode selfcheck

Doi cau hinh KHONG can sua code:
    --set scoring.use_structured=false
    --set decision.t_duplicate=0.82 --set candidates.pool_k=200
    --config configs/dup_default.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

# Gioi han luong + tat song song tokenizer TRUOC khi torch duoc nap: hai
# model cung mot tien trinh tren CPU rat de vot RAM va bi ket thuc khong
# traceback (bai hoc da ghi trong pipeline/run_ablation.py).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dup import DuplicatePipeline, load_config          # noqa: E402
from pipeline.dup._compat import load_jsonl                      # noqa: E402
from pipeline.dup.config import ablation_presets                 # noqa: E402
from pipeline.dup.evaluate import (audit_leakage, build_queries,  # noqa: E402
                                   build_queries_by_cue, decision_metrics,
                                   evaluate, evaluate_by_group)

METRIC_COLS = ["queries", "recall@1", "recall@5", "recall@10",
               "precision@5", "mrr@10", "ndcg@10"]


def _load_gold(cfg, split: str) -> list[dict]:
    """split: all | eval | train."""
    if split == "all":
        return load_jsonl(cfg.gold_path)
    p = f"data/processed/gold_{split}_pairs.jsonl"
    if not os.path.exists(p):
        print(f"  (khong co {p}, dung toan bo {cfg.gold_path})")
        return load_jsonl(cfg.gold_path)
    return load_jsonl(p)


def _print_row(name: str, m: dict, width: int = 26) -> None:
    print(f"  {name:<{width}}" + "".join(
        f"{str(m.get(c, '-')):<11}" for c in METRIC_COLS))


def _print_header(width: int = 26) -> None:
    print(f"  {'config':<{width}}" + "".join(f"{c:<11}" for c in METRIC_COLS))
    print("  " + "-" * (width + 11 * len(METRIC_COLS)))


# ------------------------------------------------------------------ modes
def mode_search(cfg, args) -> None:
    p = DuplicatePipeline(cfg).build()
    if args.query_key:
        res = p.search_by_key(args.query_key, top_k=cfg.decision.top_k)
        src = next((d for d in p.docs if d["key"] == args.query_key), {})
        print(f"\nQUERY {args.query_key}: {src.get('summary','')[:150]}\n")
    else:
        res = p.search(args.query, top_k=cfg.decision.top_k)
        print(f"\nQUERY: {args.query}\n")
    pl = res["plan"]
    print(f"  luong={pl['flow']} | e5 {'doi xung' if pl['symmetric'] else 'bat doi xung'}"
          f" | rerank={pl['rerank']}")
    if not pl["labels_measured"]:
        print("  (nhan duplicate/review/related CHUA duoc do cho luong nay -- "
              "chi doc thu tu, dung tin nhan)")
    print()
    for r in res["results"]:
        print(f"  [{r['label']:<9}] {r['score']:.3f}  {r['bugId']}  "
              f"{r['summary'][:95]}")
        print(f"      {r['explanation']}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print(f"\n-> {args.json_out}")


def mode_eval(cfg, args) -> None:
    p = DuplicatePipeline(cfg).build()
    gold = _load_gold(cfg, args.split)
    qs = build_queries(gold, set(p.key_to_idx))
    if args.limit:
        qs = qs[: args.limit]
    print(f"\ntap danh gia: {len(qs)} truy van tu {len(gold)} cap gold "
          f"(split={args.split})")

    print("\n-- KIEM TRA RO RI NHAN --")
    for k, v in audit_leakage(p, qs, load_jsonl(cfg.gold_path)).items():
        print(f"  {k:<40} {v}")

    t = time.time()
    m = evaluate(p, qs, ks=(1, 5, 10), progress_every=args.progress_every)
    m["config"] = cfg.name
    print(f"\n-- KET QUA ({time.time()-t:.0f}s) --")
    _print_header()
    _print_row(cfg.name, m)

    if args.by_cue:
        # Tach theo LOAI CUE. Xem evaluate.CUE_CLASS: 2/3 tap gold la "tuong
        # tu" chu khong phai "trung", nen mot con so gop lai khong noi len
        # duoc he thong manh o viec nao.
        print("\n-- THEO LOAI CUE --")
        print("  (n nho -> khoang tin cay rong; doc theo huong, dung doc"
              " tung chu so)")
        _print_header(34)
        by_cue = build_queries_by_cue(gold, set(p.key_to_idx))
        m_cue = {}
        for cls in ("duplicate", "similar", "unverified", "unknown", "all"):
            if not by_cue.get(cls):
                continue
            mm = evaluate(p, by_cue[cls], ks=(1, 5, 10))
            if mm.get("queries"):
                m_cue[cls] = mm
                _print_row(cls, mm, 34)
        m["by_cue"] = m_cue

    if args.by_group:
        print("\n-- THEO NHOM TRUY VAN --")
        by = {d["key"]: d for d in p.docs}
        _print_header(34)
        for g, mm in evaluate_by_group(p, qs, by).items():
            _print_row(g, mm, 34)

    print("\n-- LATENCY (ms) --")
    print("  " + json.dumps(p.timing.summary()))

    out = {"config": cfg.to_dict(), "metrics": m,
           "latency_ms": p.timing.summary()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")


def mode_ablation(cfg, args) -> None:
    presets = ablation_presets(cfg)
    names = args.configs or list(presets)
    docs = load_jsonl(cfg.corpus_path)
    gold = _load_gold(cfg, args.split)

    rows, lat = [], {}
    for name in names:
        c = presets[name]
        print(f"\n=== {name} ===")
        p = DuplicatePipeline(c, docs=docs).build(show_progress=False)
        qs = build_queries(gold, set(p.key_to_idx))
        if args.limit:
            qs = qs[: args.limit]
        t = time.time()
        m = evaluate(p, qs, ks=(1, 5, 10), progress_every=args.progress_every)
        m["config"] = name
        m["wall_s"] = round(time.time() - t, 1)
        rows.append(m)
        lat[name] = p.timing.summary()
        print(f"  recall@10={m.get('recall@10')} mrr@10={m.get('mrr@10')} "
              f"({m['wall_s']}s)", flush=True)
        # Moi pipeline giu 4388 BugRepresentation + index rieng. Chay 6 cau
        # hinh lien tiep ma khong tha ra thi bo nho leo len vai GB, va khi
        # cau hinh cuoi con nap them cross-encoder thi de bi ket thuc khong
        # traceback (dung canh bao da ghi trong pipeline/run_ablation.py).
        del p
        gc.collect()

    print("\n" + "=" * 100)
    print("ABLATION TUNG PHASE")
    print("=" * 100)
    _print_header()
    for m in rows:
        _print_row(m["config"], m)
    if len(rows) > 1:
        base = rows[0]
        print(f"\nchenh so voi '{base['config']}':")
        for m in rows[1:]:
            d = [f"{k} {m.get(k,0)-base.get(k,0):+.4f}"
                 for k in ("recall@10", "mrr@10", "ndcg@10")]
            print(f"  {m['config']:<28} " + "  ".join(d))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"results": rows, "latency_ms": lat,
                   "base_config": cfg.to_dict()}, fh, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")


def mode_audit(cfg, args) -> None:
    """In Top-K kem phan loai de NGUOI soat -- buoc 8 trong quy trinh."""
    p = DuplicatePipeline(cfg).build()
    gold = _load_gold(cfg, args.split)
    qs = build_queries(gold, set(p.key_to_idx))[: args.limit or 25]
    by = {d["key"]: d for d in p.docs}

    buckets = {"trung ro rang": [], "false positive nghi ngo": [],
               "hard negative": [], "bo sot (false negative)": []}
    for q in qs:
        res = p.search_by_key(q["key"], top_k=cfg.decision.top_k)
        rel = set(q["relevant"])
        found = [r for r in res["results"] if r["bugId"] in rel]
        print(f"\n{'='*100}\nQUERY {q['key']}: {by[q['key']]['summary'][:110]}")
        print(f"  dap an gold: {sorted(rel)}")
        for rank, r in enumerate(res["results"][:5], 1):
            mark = "GOLD" if r["bugId"] in rel else "    "
            print(f"  {rank}. {mark} [{r['label']:<9}] {r['score']:.3f} "
                  f"{r['bugId']}  {r['summary'][:80]}")
            print(f"        {r['explanation'][:200]}")
            item = {"query": q["key"], "bug": r["bugId"], "rank": rank,
                    "label": r["label"], "score": r["score"],
                    "gold": r["bugId"] in rel,
                    "summary_q": by[q["key"]]["summary"][:110],
                    "summary_b": r["summary"][:110],
                    "explanation": r["explanation"]}
            if r["bugId"] in rel and r["label"] == "duplicate":
                buckets["trung ro rang"].append(item)
            elif r["bugId"] not in rel and r["label"] == "duplicate":
                buckets["false positive nghi ngo"].append(item)
            elif r["negativeSignals"] and r["score"] >= cfg.decision.t_related:
                buckets["hard negative"].append(item)
        if not found:
            buckets["bo sot (false negative)"].append(
                {"query": q["key"], "relevant": sorted(rel)})

    print("\n" + "=" * 100)
    print("TONG HOP SOAT")
    for k, v in buckets.items():
        print(f"  {k:<28} {len(v)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(buckets, fh, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


def mode_tune(cfg, args) -> None:
    """Do trong so semantic/lexical TREN TAP TRAIN, bao cao tren tap eval.

    Bat buoc phai tach: neu do trong so tren chinh tap dung de bao cao thi
    con so bao cao la con so da duoc chon de dep, khong phai con so tong quat.
    Tap train o day la cac CUM trung lap rieng biet voi tap eval
    (xem build_training_pairs.py -- chia theo cum lien thong).
    """
    p = DuplicatePipeline(cfg).build()
    tr = build_queries(_load_gold(cfg, "train"), set(p.key_to_idx))
    ev = build_queries(_load_gold(cfg, "eval"), set(p.key_to_idx))
    print(f"\ntune tren {len(tr)} truy van TRAIN; bao cao tren {len(ev)} truy van EVAL")

    grid = [float(x) for x in (args.grid or
                               ["0.0", "0.2", "0.3", "0.4", "0.5", "0.6",
                                "0.8", "1.0"])]
    best, rows = None, []
    for ws in grid:
        cfg.scoring.weights.semantic = ws
        cfg.scoring.weights.lexical = round(1.0 - ws, 3)
        m = evaluate(p, tr, ks=(1, 5, 10))
        m["w_semantic"] = ws
        rows.append(m)
        print(f"  w_sem={ws:.2f} w_lex={1-ws:.2f}  "
              f"recall@10={m.get('recall@10')} mrr@10={m.get('mrr@10')}")
        if best is None or m.get("mrr@10", 0) > best.get("mrr@10", 0):
            best = m

    cfg.scoring.weights.semantic = best["w_semantic"]
    cfg.scoring.weights.lexical = round(1.0 - best["w_semantic"], 3)
    print(f"\nchon tren TRAIN: w_semantic={best['w_semantic']} "
          f"(mrr@10 train={best['mrr@10']})")
    held = evaluate(p, ev, ks=(1, 5, 10))
    print("\n-- BAO CAO TREN TAP EVAL (chua tung dung de chon) --")
    _print_header()
    _print_row("tuned", held)

    out = {"grid": rows, "chosen_w_semantic": best["w_semantic"],
           "train_metrics": best, "eval_metrics": held}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    if args.save_config:
        cfg.save(args.save_config)
        print(f"-> {args.save_config}")
    print(f"-> {args.out}")


def mode_tune_threshold(cfg, args) -> None:
    """Quet t_duplicate / t_related TREN TAP TRAIN, bao cao tren tap EVAL.

    Luu y ve cach doc ket qua: precision o day la CAN DUOI (tap gold khong
    day du), nen tri tuyet doi khong co y nghia -- chi dung de so sanh giua
    cac nguong voi nhau. Vi vay ham chon nguong theo F1 can-duoi, roi in ca
    duong cong ra de nguoi doc tu nhin diem gay.
    """
    p = DuplicatePipeline(cfg).build()
    tr = build_queries(_load_gold(cfg, "train"), set(p.key_to_idx))
    ev = build_queries(_load_gold(cfg, "eval"), set(p.key_to_idx))
    print(f"\nquet nguong tren {len(tr)} truy van TRAIN; "
          f"bao cao tren {len(ev)} truy van EVAL")

    grid = [float(x) for x in (args.grid or
                               ["0.60", "0.65", "0.70", "0.74", "0.78",
                                "0.82", "0.86", "0.90"])]
    rows, best = [], None
    print(f"\n  {'t_dup':<8}{'dup_recall':<12}{'dup_prec(can duoi)':<20}"
          f"{'f1':<9}{'#duplicate':<12}")
    for t in grid:
        cfg.decision.t_duplicate = t
        m = decision_metrics(p, tr, top_k=cfg.decision.top_k)
        m["t_duplicate"] = t
        rows.append(m)
        print(f"  {t:<8.2f}{m['dup_recall']:<12}"
              f"{m['dup_precision_lower_bound']:<20}"
              f"{m['dup_f1_lower_bound']:<9}{m['n_labeled_duplicate']:<12}")
    # CACH CHON -- day la cho de sai nhat cua ca buoc nay.
    #
    # Toi da hoa F1 la SAI o bai toan nay. Precision o tren la can duoi (tap
    # gold khong day du) nen no bi dem thieu mot cach co he thong, con recall
    # thi khong. F1 vi vay luon thien ve recall va keo nguong xuong day:
    # tren du lieu that no chon 0.65, sinh 266 nhan 'duplicate' o tap eval
    # trong khi chi 10.9% trong so do la gold.
    #
    # Dung yeu cau nghiep vu ("false positive dat hon false negative"), tieu
    # chi la SAN PRECISION: lay nguong THAP NHAT ma precision con dat muc san
    # -- tuc lay het recall co the, mien la khong pha muc do sach da cam ket.
    if args.objective == "f1":
        best = max(rows, key=lambda m: m["dup_f1_lower_bound"])
        why = f"f1 can duoi={best['dup_f1_lower_bound']}"
    else:
        ok = [m for m in rows
              if m["dup_precision_lower_bound"] >= args.precision_floor]
        best = min(ok, key=lambda m: m["t_duplicate"]) if ok else \
            max(rows, key=lambda m: m["dup_precision_lower_bound"])
        why = (f"nguong thap nhat dat san precision >= {args.precision_floor}"
               if ok else "khong nguong nao dat san -- lay precision cao nhat")

    cfg.decision.t_duplicate = best["t_duplicate"]
    print(f"\nchon tren TRAIN: t_duplicate={best['t_duplicate']}  ({why})")
    print(f"  recall={best['dup_recall']} "
          f"precision(can duoi)={best['dup_precision_lower_bound']}")

    held = decision_metrics(p, ev, top_k=cfg.decision.top_k)
    print("\n-- CHAT LUONG NHAN TREN TAP EVAL (chua tung dung de chon) --")
    for k, v in held.items():
        print(f"  {k:<28} {v}")
    rank = evaluate(p, ev, ks=(1, 5, 10))
    print("\n-- XEP HANG TREN EVAL (khong doi theo nguong, dung de doi chieu) --")
    _print_header()
    _print_row(cfg.name, rank)

    out = {"grid": rows, "chosen": best, "eval_decision": held,
           "eval_ranking": rank, "config": cfg.to_dict()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    if args.save_config:
        cfg.save(args.save_config)
        print(f"-> {args.save_config}")
    print(f"-> {args.out}")


SELFCHECK_CASES = [
    ("From Date = To Date", "From Date < To Date", "operator_conflict"),
    ("Agent = TERMINATED khong thuc hien duoc giao dich",
     "Agent = LOCKED khong thuc hien duoc giao dich", "status_conflict"),
    ("Bam Save tren man hinh cau hinh kenh",
     "Bam Submit tren man hinh cau hinh kenh", "action_conflict"),
    ("Nhap so tien = 0 he thong bao loi",
     "Nhap so tien = 1 he thong bao loi", "numeric_conflict"),
    ("Man hinh hien thi thong bao thanh cong",
     "Man hinh khong hien thi thong bao thanh cong", "negation_conflict"),
    ("Truyen thieu truong requestTime API bao loi",
     "Truyen thieu truong accessChannelID API bao loi", "entity_conflict"),
    ("[USSD EU][Show balance] khong hien thi so du",
     "[USSD supper agent][check balance] khong hien thi so du", "actor_conflict"),
    ("Chuyen tien sang ngan hang CRDB bi loi timeout",
     "Chuyen tien sang ngan hang NMB bi loi timeout", "provider_conflict"),
]

SELFCHECK_POSITIVE = [
    ("[APP EU][Bundle] khong mua duoc goi cuoc",
     "[EU_APP][Buy Bundle] khong mua duoc goi cuoc"),
    ("nguoi dung khong dang nhap duoc sau khi doi mat khau",
     "tai khoan bi tu choi xac thuc sau khi cap nhat mat khau"),
]


def mode_selfcheck(cfg, args) -> None:
    """Kiem tra rieng phase 1-4 tren cac vi du chuan (khong can corpus)."""
    from pipeline.dup.extract import StructuredExtractor
    from pipeline.dup.features import PairFeatureExtractor
    from pipeline.dup.preprocess import Preprocessor
    from pipeline.dup.scoring import HybridScorer

    pre = Preprocessor(cfg.preprocess)
    ex = StructuredExtractor(cfg.extract, pre)
    fx = PairFeatureExtractor(cfg.scoring)
    sc = HybridScorer(cfg.scoring)

    print("\n-- PHASE 1: token co giu toan tu / so / phu dinh khong --")
    for t in ("From Date = To Date", "From Date < To Date",
              "so tien = 0", "so tien = 1", "khong hien thi"):
        toks = [x for x in pre.tokenize(t)
                if x.startswith(("op_", "num_", "neg_"))]
        print(f"  {t:<28} -> {toks}")

    print("\n-- PHASE 4: cap MAU THUAN (van ban rat giong) --")
    ok = 0
    for a_txt, b_txt, want in SELFCHECK_CASES:
        ra, rb = ex.extract_query(a_txt, "A"), ex.extract_query(b_txt, "B")
        f = fx.extract(ra, rb, semantic=0.97, lexical=0.95)
        bd = sc.score(f)
        hit = f.get(want) > 0
        ok += hit
        print(f"  [{'OK ' if hit else 'MISS'}] {want:<20} score={bd.final:.3f} "
              f"gate={bd.gate_applied}")
        print(f"         A: {a_txt[:62]}")
        print(f"         B: {b_txt[:62]}")
        print(f"         tru: {bd.negative_signals()}")
    print(f"  --> bat duoc {ok}/{len(SELFCHECK_CASES)} tin hieu mau thuan")

    print("\n-- PHASE 4: cap DUONG (khong duoc bi phat) --")
    for a_txt, b_txt in SELFCHECK_POSITIVE:
        ra, rb = ex.extract_query(a_txt, "A"), ex.extract_query(b_txt, "B")
        f = fx.extract(ra, rb, semantic=0.90, lexical=0.70)
        bd = sc.score(f)
        print(f"  score={bd.final:.3f} cong={bd.positive_signals()} "
              f"tru={bd.negative_signals()}")
        print(f"     A: {a_txt[:70]}\n     B: {b_txt[:70]}")


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="search",
                    choices=["search", "eval", "ablation", "audit", "selfcheck",
                             "tune", "tune-threshold"])
    ap.add_argument("--config", default=None, help="file JSON cau hinh")
    # action="extend": cho phep LAP LAI --set. Voi nargs="*" thuong, lan
    # --set sau GHI DE lan truoc, nen "--set rerank.enabled=true
    # --set rerank.kind=sklearn" chi con ve trai thu hai -- reranker khong
    # he duoc bat ma bao cao van in ra binh thuong.
    ap.add_argument("--set", dest="overrides", action="extend", nargs="+",
                    default=[], help="ghi de: --set scoring.use_structured=false")
    ap.add_argument("--query", default="")
    ap.add_argument("--query-key", default="")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["all", "eval", "train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--by-group", action="store_true")
    ap.add_argument("--by-cue", action="store_true",
                    help="tach ket qua theo loai cue (duplicate/similar/...)")
    ap.add_argument("--progress-every", type=int, default=0)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--objective", default="precision-floor",
                    choices=["precision-floor", "f1"],
                    help="tune-threshold: tieu chi chon nguong")
    ap.add_argument("--precision-floor", type=float, default=0.40,
                    help="tune-threshold: san precision (can duoi) phai dat")
    ap.add_argument("--grid", nargs="*", default=None,
                    help="tune: cac gia tri w_semantic can thu")
    ap.add_argument("--save-config", default="",
                    help="tune: ghi config da chon ra file JSON")
    ap.add_argument("--out", default="data/processed/dup_e2e.json")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.top_k:
        cfg.decision.top_k = args.top_k

    if args.mode == "search" and not (args.query or args.query_key):
        sys.exit("can --query hoac --query-key")

    print(f"config: {cfg.name}  corpus={cfg.corpus_path}")
    {"search": mode_search, "eval": mode_eval, "ablation": mode_ablation,
     "audit": mode_audit, "selfcheck": mode_selfcheck,
     "tune": mode_tune,
     "tune-threshold": mode_tune_threshold}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
