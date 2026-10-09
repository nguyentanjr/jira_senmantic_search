# -*- coding: utf-8 -*-
"""
Phan tich THEO NHOM TRUY VAN cho luong mo ta tu do, tren TEST -- mo ta, KHONG chon.

    python pipeline/group_analysis_text.py --device cuda
    python pipeline/group_analysis_text.py --device cuda --limit 20     # chay thu, khong ghi file

Thiet ke chot truoc khi chay: HANDOFF.md §7 "Follow-up analyses" (muc A).

VI SAO CHAY LAI TEST
final2_eval_test.json chi luu so tong, khong co thu hang tung truy van. Script nay
chay lai DUNG cac cau hinh da chot (khong he moi, khong doi tham so) de chia nhom.
Day la LAN XEM THU HAI cua test luong text: ket qua chi de mo ta, KHONG duoc dung
de doi cau hinh -- ghi ro khi bao cao.

CAU HINH (bon buoc cua problem.txt, truoc / sau fine-tune)
  keyword | semantic, hybrid voi e5 goc | semantic, hybrid voi e5-task2 |
  hybrid+rerank: e5-task2 + ce-task, blend 0.3 (dang chay)
  BM25 fold auto, tien to 'query: ' / 'passage: ', rerank top-30.
NHOM (group_stats.py): tin hieu lexical tren CHINH chuoi truy van (ban khong dau
giu nhan cua ban goc); kem nhom theo `lexical_group` cua issue dap an.

Dau ra: data/processed/group_text_test.json, group_text_test_ranks.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bm25_fold import answer_rank                                     # noqa: E402
from eval_rerankers import make_split, metrics_from_ranked                 # noqa: E402
from final_eval2 import QVERS, RERANK_K, SETS                              # noqa: E402
from group_stats import (GROUPS, group_label, group_report, has_signal,    # noqa: E402
                         print_report)
from retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,     # noqa: E402
                       SearchEngine, fuse_rerank, load_jsonl, strip_diacritics)
from run_ablation import CONFIGS                                           # noqa: E402

DENSE = {"e5": "intfloat/multilingual-e5-base", "e5-task2": "models/e5-base-task2"}
RERANKER = "models/bge-reranker-base-task"
DEPLOYED = "hybrid+rerank|e5-task2|ce-task|0.3"
METRICS = ("recall@10", "mrr@10", "ndcg@10")
GROUP_BY = ("q_signal", "lexical_group")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ce-batch", type=int, default=128)
    ap.add_argument("--clean-keys", default="data/processed/finetune_split_clean.json")
    ap.add_argument("--dev-keys", default="data/processed/paraphrase_rerank_sample500.json")
    ap.add_argument("--limit", type=int, default=0, help="chay thu: khong ghi file")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out or f"data/processed/group_text_{args.split}.json"
    ranks_out = os.path.splitext(out)[0] + "_ranks.jsonl"
    if os.path.exists(out) and not args.force and not args.limit:
        sys.exit(f"{out} da ton tai -- khong chay lai test (dung --force neu that su can)")
    for p in (DENSE["e5-task2"], RERANKER):
        if not os.path.isdir(p):
            sys.exit(f"thieu model {p}")

    with open(args.clean_keys, encoding="utf-8") as f:
        clean = json.load(f)["paraphrase_clean_keys"]
    with open(args.dev_keys, encoding="utf-8") as f:
        wanted = set(make_split(clean, json.load(f)["paraphrase_clean_keys"])[args.split])

    cfg_kw = SearchConfig(name="keyword", top_k=100, **CONFIGS["keyword"])
    cfg = {c: SearchConfig(name=c, top_k=100, **CONFIGS[c]) for c in ("semantic", "hybrid")}
    data = {}
    for s, (cpath, qpath) in SETS.items():
        corpus = load_jsonl(cpath)
        qs = [q for q in load_jsonl(qpath) if q["key"] in wanted]
        if args.limit:
            qs = qs[: args.limit]
        base = SearchEngine(corpus, bm25_fold="auto")
        base.build_bm25()
        items = []
        for q in qs:
            rel = {base.key_to_idx[k] for k in q["relevant"] if k in base.key_to_idx}
            if not rel:
                continue
            flags = {"q_signal": has_signal(q["query"]),
                     "lexical_group": bool(q.get("lexical_group"))}
            for qv, text in (("goc", q["query"]), ("khong_dau", strip_diacritics(q["query"]))):
                items.append({"key": q["key"], "qv": qv, "text": text, "rel": rel,
                              "jac_bin": q.get("jac_bin"), **flags})
        ranked = {"keyword": [base.search(it["text"], cfg_kw) for it in items]}
        for dn, path in DENSE.items():
            t0 = time.perf_counter()
            eng = SearchEngine(corpus, bm25_fold="auto",
                               dense=DenseIndex(path, device=args.device, batch_size=64,
                                                query_prefix="query: ", passage_prefix="passage: "))
            eng._bm25, eng._bm25_add = base._bm25, base._bm25_add
            eng.build_dense(show_progress=True)
            for c in cfg:
                ranked[f"{c}|{dn}"] = [eng.search(it["text"], cfg[c]) for it in items]
            print(f"  {s} / {dn}: {time.perf_counter() - t0:.0f}s", flush=True)
            del eng
            gc.collect()
        data[s] = {"corpus": corpus, "items": items, "ranked": ranked}
        print(f"{s}: {len(items) // 2} truy van x 2 ban", flush=True)

    t0 = time.perf_counter()
    rr = CrossEncoderReranker(RERANKER, device=args.device, batch_size=args.ce_batch, fp16=True)
    for s, dd in data.items():
        hy = dd["ranked"]["hybrid|e5-task2"]
        pairs = [(i, int(d)) for i, h in enumerate(hy) for d in h[:RERANK_K]]
        sc = rr.score_pairs([(dd["items"][i]["text"], dd["corpus"][d].get("text_dense", ""))
                             for i, d in pairs], show_progress=True)
        fused, k = [], 0
        for h in hy:
            n = len(h[:RERANK_K])
            fused.append(np.concatenate([fuse_rerank(h[:RERANK_K], sc[k:k + n], "blend", 0.3),
                                         h[RERANK_K:]]))
            k += n
        dd["ranked"][DEPLOYED] = fused
    print(f"  reranker: {time.perf_counter() - t0:.0f}s", flush=True)
    del rr
    gc.collect()

    names = list(next(iter(data.values()))["ranked"])
    result = {"split": args.split, "second_look_at_test": args.split == "test",
              "note": "chay lai cau hinh da chot de chia nhom; chi mo ta, khong dung de chon",
              "configs": names, "dense": DENSE, "reranker": RERANKER, "bm25_fold": "auto",
              "groups": list(GROUPS), "group_by": list(GROUP_BY), "n_boot": args.n_boot,
              "slices": {}}
    rows = []
    for s, dd in data.items():
        for qv in QVERS:
            idx = [i for i, it in enumerate(dd["items"]) if it["qv"] == qv]
            scores = {}
            for n in names:
                ms = [metrics_from_ranked(dd["ranked"][n][i], dd["items"][i]["rel"]) for i in idx]
                scores[n] = {m: np.asarray([x[m] for x in ms]) for m in METRICS}
            sl = f"{s}:{qv}"
            result["slices"][sl] = {}
            for by in GROUP_BY:
                labels = [group_label(dd["items"][i][by]) for i in idx]
                rep = group_report(scores, labels, "keyword", METRICS, args.n_boot)
                result["slices"][sl][by] = rep
                print_report(f"{sl} | nhom theo {by}", rep)
        for i, it in enumerate(dd["items"]):
            row = {"set": s, "key": it["key"], "qv": it["qv"], "q_signal": it["q_signal"],
                   "lexical_group": it["lexical_group"], "jac_bin": it["jac_bin"]}
            for n in names:
                r = answer_rank(dd["ranked"][n][i][:100], it["rel"])
                row[f"rank:{n}"] = None if r is None else int(r)
            rows.append(row)

    if args.limit:
        print("\n--limit: khong ghi file")
        return
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(ranks_out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n-> {out}\n-> {ranks_out}")


if __name__ == "__main__":
    main()
