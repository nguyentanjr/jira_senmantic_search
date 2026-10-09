# -*- coding: utf-8 -*-
"""
MO RONG bo pooling bang top-K cua MOT he moi -- moi dong da cham giu nguyen.

    python pipeline/extend_pooling_set.py --model models/e5-base-task2 --name hybrid_e5task2

    # sau do NGUOI DUNG chay bang GEMINI_API_KEY cua ho (them --dry-run de xem truoc):
    python pipeline/pooling_llm_grade.py fill --model gemini-3.5-flash \
        --inp data/processed/pooling_judgments_v4.csv \
        --out data/processed/pooling_judgments_v4_llm.csv \
        --cache data/processed/pooling_llm_cache_v4.jsonl

VI SAO
Pool v3 chi gom top-10 cua keyword / semantic / hybrid voi e5 goc. He moi keo
tai lieu ngoai pool len top-10 va eval tinh cac tai lieu do la 0 -> so sanh
thien vi chong lai he moi (compare_models.py in ty le "top-10 chua cham").
Chuan TREC: gop top-K cua he moi vao pool, cham phan con thieu bang DUNG quy
trinh da cham pool cu, roi tinh lai MOI he tren pool moi.

QUY TAC -- chot truoc khi do (HANDOFF.md §7, "Issue flow: model selection")
  - Dong cua file goc giu nguyen tung o, ke ca nhan LLM cu va dong "?".
  - Chi them cap (truy van, tai lieu) chua co dong nao; bo chinh truy van.
  - Dong moi trong moi truy van duoc sort roi XAO theo seed -> thu tu khong lo
    thu hang. File gan nhan khong ghi cau hinh nao tim ra tai lieu nao; anh xa
    nam rieng trong pooling_runs<suffix>.jsonl.
  - Cot query_* cua dong moi chep tu dong goc cua cung truy van -> LLM thay
    dung mo ta A nhu luc cham v3.
  - Cham dong moi bang `pooling_llm_grade.py fill`: vi du chi lay tu nhan NGUOI;
    nhan LLM cu khong lam vi du va khong bi cham lai (LLM_SOURCES).

GIOI HAN (ghi khi bao cao): moi luot goi LLM chi gom tai lieu cua he moi,
khong tron voi tai lieu cua cac he cu nhu luc cham v3.

Dau ra (khong ghi de file co san):
  data/processed/pooling_judgments_v4.csv   dong v3 + dong moi (grade trong)
  data/processed/pooling_runs_v4.jsonl      runs v3 + runs he moi (GIU RIENG)
  data/processed/pooling_stats_v4.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_pooling_set import FIELDS, judgment_row                        # noqa: E402
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl  # noqa: E402
from run_ablation import CONFIGS, E5_QUERY                                # noqa: E402

D = "data/processed"


def plan_new_pairs(top_by_query: dict[str, list[str]],
                   judged_by_query: dict[str, set[str]], seed: int) -> dict[str, list[str]]:
    """Tai lieu can them cho tung truy van: chua co dong nao, khong phai chinh truy van.

    Thu tu dau ra KHONG phu thuoc thu hang (sort roi xao) -> nguoi/LLM cham khong
    doan duoc tai lieu nao he moi xep dau.
    """
    rng = random.Random(seed)
    out = {}
    for q, top in top_by_query.items():
        seen = judged_by_query.get(q, set())
        cand = sorted({d for d in top if d != q and d not in seen})
        rng.shuffle(cand)
        out[q] = cand
    return out


def merge_rows(base_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Dong goc giu nguyen (thu tu trong moi truy van); dong moi noi sau khoi cua truy van."""
    by_q: dict[str, list[dict]] = {}
    for r in base_rows:
        by_q.setdefault(r["query_key"], []).append(r)
    extra: dict[str, list[dict]] = {}
    for r in new_rows:
        if r["query_key"] not in by_q:
            raise ValueError(f"truy van {r['query_key']} khong co trong file goc")
        extra.setdefault(r["query_key"], []).append(r)
    return [r for q, rs in by_q.items() for r in rs + extra.get(q, [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=f"{D}/corpus.jsonl")
    ap.add_argument("--base-judgments", default=f"{D}/pooling_judgments_v3_llm.csv")
    ap.add_argument("--base-runs", default=f"{D}/pooling_runs_v3.jsonl")
    ap.add_argument("--model", required=True, help="model dense cua he moi")
    ap.add_argument("--config", default="hybrid", help="cau hinh trong run_ablation.CONFIGS")
    ap.add_argument("--name", required=True, help="ten he moi trong runs, vd hybrid_e5task2")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--suffix", default="_v4")
    ap.add_argument("--out-dir", default=D)
    ap.add_argument("--desc-chars", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=15,
                    help="chi de uoc luong so luot goi LLM -- bang --batch-size cua fill")
    args = ap.parse_args()

    j_path = os.path.join(args.out_dir, f"pooling_judgments{args.suffix}.csv")
    r_path = os.path.join(args.out_dir, f"pooling_runs{args.suffix}.jsonl")
    s_path = os.path.join(args.out_dir, f"pooling_stats{args.suffix}.json")
    for p in (j_path, r_path):
        if os.path.exists(p):
            sys.exit(f"{p} da ton tai -- doi --suffix, khong ghi de")

    with open(args.base_judgments, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        base_rows, fields = list(rd), list(rd.fieldnames or [])
    missing = [c for c in FIELDS if c not in fields]
    if missing:
        sys.exit(f"{args.base_judgments} thieu cot {missing}")
    base_runs = load_jsonl(args.base_runs)
    if any(r["config"] == args.name for r in base_runs):
        sys.exit(f"{args.base_runs} da co cau hinh ten {args.name}")
    queries = list(dict.fromkeys(r["query_key"] for r in base_runs))
    first_row: dict[str, dict] = {}
    for r in base_rows:
        first_row.setdefault(r["query_key"], r)
    if set(queries) != set(first_row):
        sys.exit("tap truy van cua runs va file gan nhan khong khop")

    docs = load_jsonl(args.corpus)
    by_key = {d["key"]: d for d in docs}
    cfg = SearchConfig(name=args.config, top_k=args.depth, **CONFIGS[args.config])
    # Luong issue: ca noi dung issue <-> ca noi dung issue -> tien to DOI XUNG,
    # giong build_pooling_set.py / eval_similar.py / compare_models.py.
    dense = DenseIndex(args.model, device="cpu", batch_size=32,
                       query_prefix=E5_QUERY, passage_prefix=E5_QUERY)
    eng = SearchEngine(docs, dense=dense)
    eng.build_bm25()
    if CONFIGS[args.config]["use_dense"]:
        print("dang nhung corpus (doc tu cache neu da co)...", flush=True)
        eng.build_dense(show_progress=False)

    top, new_runs = {}, []
    for q in queries:
        d = by_key[q]
        qtext = d.get("text_dense") or d.get("summary", "")
        ranked = eng.search(qtext, cfg, exclude={eng.key_to_idx[q]})[:args.depth]
        top[q] = [eng.keys[i] for i in ranked]
        new_runs += [{"query_key": q, "config": args.name, "rank": rank, "doc_key": k}
                     for rank, k in enumerate(top[q], 1)]

    judged: dict[str, set[str]] = {}
    for r in base_rows:
        judged.setdefault(r["query_key"], set()).add(r["doc_key"])
    plan = plan_new_pairs(top, judged, args.seed + 2)

    new_rows = []
    for q in queries:
        qb = first_row[q]
        for k in plan[q]:
            row = {c: "" for c in fields}
            row.update(judgment_row(by_key[q], by_key[k], args.desc_chars, qb["query_desc"]))
            for c in ("query_summary", "query_desc", "query_error_codes"):
                row[c] = qb[c]
            new_rows.append(row)
    rows = merge_rows(base_rows, new_rows)

    with open(j_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(r_path, "w", encoding="utf-8") as f:
        for r in base_runs + new_runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_new = {q: len(plan[q]) for q in queries}
    n_top = sum(len(v) for v in top.values())
    stats = {
        "base_judgments": args.base_judgments, "base_runs": args.base_runs,
        "model": args.model, "config": args.config, "name": args.name,
        "depth": args.depth, "shuffle_seed": args.seed + 2, "n_queries": len(queries),
        "n_base_rows": len(base_rows), "n_new_rows": sum(n_new.values()),
        "top_already_judged": round(1 - sum(n_new.values()) / n_top, 4),
        "queries_with_new_rows": sum(1 for v in n_new.values() if v),
        "new_rows_per_query": n_new,
        "new_rows_with_resolution": sum(int(r["doc_has_resolution"]) for r in new_rows),
        "est_llm_calls": sum(math.ceil(v / args.batch_size) for v in n_new.values()),
    }
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"\n{args.name}: top-{args.depth} cua {len(queries)} truy van "
          f"| da cham san {stats['top_already_judged']:.1%}")
    print(f"dong moi can cham: {stats['n_new_rows']} ({stats['queries_with_new_rows']} truy van, "
          f"toi da {max(n_new.values())}/truy van) | uoc tinh {stats['est_llm_calls']} luot goi LLM")
    print(f"-> {j_path}\n-> {r_path}   (GIU RIENG)\n-> {s_path}")
    llm_out = os.path.splitext(j_path)[0] + "_llm.csv"
    cache = os.path.join(args.out_dir, f"pooling_llm_cache{args.suffix}.jsonl")
    print("\nBuoc tiep -- NGUOI DUNG chay (can GEMINI_API_KEY; them --dry-run de xem truoc):\n"
          f"  python pipeline/pooling_llm_grade.py fill --model gemini-3.5-flash "
          f"--inp {j_path} --out {llm_out} --cache {cache}")


if __name__ == "__main__":
    main()
