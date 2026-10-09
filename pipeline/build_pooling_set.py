# -*- coding: utf-8 -*-
"""
Dung BO POOLING de nguoi gan nhan (bo danh gia 2 -- bo bao cao chinh thuc).

    python pipeline/build_pooling_set.py                    # 40 truy van, sau 10
    python pipeline/build_pooling_set.py --n-queries 25 --depth 8

    # giu nguyen pool da co, chi dung lai cot text cho nguoi gan nhan
    python pipeline/build_pooling_set.py \
        --from-runs data/processed/pooling_runs_v2.jsonl --suffix _v3

VI SAO PHAI CO BO NAY
Tap gold dao tu comment co ba khiem khuyet khong sua duoc bang cach chinh model:
  1. thien lech ve phia lexical -- ky su tim ra cap trung bang chinh keyword
     search cua Jira, nen tap gold gan nhu chi chua cap keyword da giai duoc;
  2. NHAN THIEU -- mot truy van co the co 5 issue lien quan, gold chi biet 1;
     he thong tra ve cai thu 2 van bi cham la miss;
  3. khong scale -- 19,4 cap/1000 issue va dang giam.

Khiem khuyet (2) la thu duy nhat POOLING chua duoc, va no thien vi CO HUONG:
he thong nao tra ve nhieu tai lieu lien quan chua duoc gan nhan thi bi phat
nang hon. Retrieval ngu nghia dung la loai he thong do -- nen khoang cach BM25
dang do duoc nhieu kha nang bi thoi phong.

CACH LAM (chuan TREC)
Gop top-K cua MOI cau hinh retrieval thanh mot "pool", roi nguoi gan nhan tren
pool do. Moi cau hinh deu co co hoi dong gop tai lieu duong, nen khong cau hinh
nao bi phat vi tim ra thu ma cau hinh khac bo sot.

CHONG THIEN VI KHI GAN NHAN -- BAT BUOC
File giao cho nguoi KHONG chua thong tin cau hinh nao tim ra tai lieu nao, va
thu tu trong moi truy van da bi XAO. Neu de lo, nguoi gan nhan se vo thuc uu ai
cau hinh dung dau. Anh xa cau hinh -> thu hang duoc giu rieng trong
`pooling_runs.jsonl` va chi dung luc tinh diem.

Cung ly do do: KHONG dua nhan LLM vao file gan nhan. LLM da do duoc precision
25% tren chinh mien nay (xem llm_label.py), nen goi y cua no se neo nguoi gan
nhan ve phia sai nhieu hon la giup.

THANG NHAN: HOI VE GIA TRI SU DUNG, KHONG HOI VE "CUNG DEFECT"

    2 = giup truc tiep   cung van de hoac cung co che, VA co ghi nhan cach xu ly
    1 = dang doc         lien quan ve chuc nang / luong, co the co manh moi
    0 = khong giup gi
    ? = khong du thong tin

Cau hoi khi gan nhan la:
    "Neu toi dang dieu tra issue A va he thong dua issue B len, B co giup toi
     khong?"

KHONG hoi "hai issue nay co cung underlying defect khong". Da do: ba nguoi gan
nhan doc lap (ky su, Gemini flash-lite, Claude Opus 5) cho ba phan bo khac han
nhau tren cung 53 cap -- ky su goi 8 cap la trung lap, Gemini 28, Claude 32,
kappa lan luot 0,179 va 0,006. Moi ben deu tu nhat quan; cai thieu la mot dinh
nghia chung, va no khong ton tai.

Cau hoi ve gia tri su dung thi khong vuong chuyen do: ky su phan duoc ngay,
va no do dung thu de tai huu ich.

Ly do sau xa: tren Jira QA noi bo, TRUNG LAP THAT GAN NHU KHONG CO. Do tren
chinh corpus nay -- 75 cap co cue "trung/duplicate" tren 10.925 bug (0,69%),
44% cach nhau <= 1 ngay, 45% cung nguoi bao. Do la QA chu dong tao ticket song
song khi quet mot man hinh qua nhieu portal, khong phai ai do bao lai loi cu.
Bug da fix thi khong ai submit lai. Bai toan DBRD kinh dien (Bugzilla/Mozilla,
nguoi dung cuoi, ty le trung lap 10-30%) khong ton tai o day.

Dau ra:
  data/processed/pooling_judgments.csv   -> nguoi dien cot `grade`
  data/processed/pooling_runs.jsonl      -> anh xa cau hinh/thu hang (GIU RIENG)
  data/processed/pooling_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resolution                                                        # noqa: E402
from build_corpus import CODE_MARK, SECTION_CANON, SECTION_RE             # noqa: E402
from retrieval import DenseIndex, SearchConfig, SearchEngine, load_jsonl  # noqa: E402
from run_ablation import CONFIGS, E5_QUERY                                # noqa: E402

FIELDS = ["query_key", "doc_key", "grade", "note",
          "query_summary", "doc_summary", "query_desc", "doc_desc",
          "query_error_codes", "doc_error_codes", "doc_status",
          "doc_resolution", "doc_has_resolution", "doc_url"]

#: dong dai hon nguong nay trong section la log dan vao, khong chen lai
MAX_RESTORE_LINE = 300


def short(s: str, n: int = 400) -> str:
    s = " ".join((s or "").split())
    return s[:n]


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _heading_index(lines: list[str], canon: str) -> int | None:
    for i, ln in enumerate(lines):
        m = SECTION_RE.match(ln)
        if m and SECTION_CANON.get(" ".join(m.group(1).lower().split())) == canon:
            return i
    return None


def _restore(biz: str, val: str, canon: str) -> str:
    """Chen lai cac dong cua section ma `text_business` da danh mat.

    `split_business_technical` xep dong nao co traceId/SQL vao khoi ky thuat,
    ke ca dong nguoi viet nhu "- Bao loi System busy, traceId: ...". Section
    thi parse tu mo ta goc nen con giu. Moi dong thieu duoc dat ngay sau dong
    lien truoc no trong section (hoac sau tieu de) de giu thu tu goc.
    """
    lines = biz.split("\n") if biz else []
    norms = [_norm(x) for x in lines]
    prev = _heading_index(lines, canon)
    # tim tu CHINH dong tieu de: "Actual: noi dung" viet cung dong
    start = prev if prev is not None else 0
    tail = []
    for vl in val.split("\n"):
        nv = _norm(vl)
        if not nv:
            continue
        hit = next((i for i in range(start, len(lines))
                    if norms[i] == nv or norms[i].endswith(" " + nv)), None)
        if hit is not None:
            prev, start = hit, hit + 1
            continue
        if len(nv) > MAX_RESTORE_LINE:
            continue
        if prev is None:
            tail.append(vl.strip())
            continue
        lines.insert(prev + 1, vl.strip())
        norms.insert(prev + 1, nv)
        prev += 1
        start = prev + 1
    if tail:
        label = "Thực tế" if canon == "actual" else "Mong muốn"
        lines += [f"{label}:"] + tail
    return "\n".join(lines)


def describe(d: dict, limit: int) -> str:
    """Mo ta nghiep vu day du cua issue, giu xuong dong.

    KHONG dung rieng `sections["actual"]`: parser chi bat duoc khi co tieu de
    "Thuc te:"; 18% corpus viet kieu "buoc 1... =>> hien thi ..." thi rong,
    va phan truoc tieu de dau tien bi bo. Cung khong du neu thieu "Mong muon":
    nhieu cap chi lien quan qua ma loi/thong bao mong doi.
    """
    text = d.get("text_business", "") or ""
    secs = d.get("sections") or {}
    for canon in ("actual", "expected"):
        if secs.get(canon):
            text = _restore(text, secs[canon], canon)
    # section parse tu mo ta goc nen con dau khoi code "\x00CODE\x00"
    text = text.replace(CODE_MARK, "")
    text = "\n".join(" ".join(ln.split()) for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …[cắt]"


def error_codes(d: dict, n: int = 8) -> str:
    return ", ".join(((d.get("signals") or {}).get("error_code") or [])[:n])


def judgment_row(q: dict, d: dict, desc_chars: int, q_desc: str | None = None) -> dict:
    """Mot dong gan nhan (truy van q, ung vien d). Dung chung voi
    extend_pooling_set.py de dong them vao pool cung dinh dang, cung cach cat text."""
    # Phai hien resolution: thang nhan moi hoi ve gia tri su dung, ma
    # khong nhin thay phan xu ly da ghi nhan thi khong gan duoc nhan 2.
    res = resolution.extract(d.get("text_comments", ""))
    return {
        "query_key": q["key"], "doc_key": d["key"], "grade": "", "note": "",
        "query_summary": short(q.get("summary", ""), 200),
        "doc_summary": short(d.get("summary", ""), 200),
        "query_desc": describe(q, desc_chars) if q_desc is None else q_desc,
        "doc_desc": describe(d, desc_chars),
        "query_error_codes": error_codes(q),
        "doc_error_codes": error_codes(d),
        "doc_status": "/".join(x for x in (d.get("status"), d.get("resolution")) if x),
        "doc_resolution": short(res, 600),
        "doc_has_resolution": int(bool(res)),
        "doc_url": d.get("url", ""),
    }


def pick_queries(docs: list[dict], n: int, seed: int) -> list[dict]:
    """Lay mau phan tang theo `lexical_group`.

    De tai cam ket phan tich rieng nhom co ma loi/API (lexical co loi the) va
    nhom thuan nghiep vu (semantic co loi the). Lay ngau nhien thuan thi hai
    nhom lech nhau va lat cat do mat y nghia.
    """
    rng = random.Random(seed)
    ok = [d for d in docs
          if len(d.get("text_business", "")) >= 250
          and (d.get("sections") or {}).get("actual")]
    groups = defaultdict(list)
    for d in ok:
        groups[bool(d.get("lexical_group"))].append(d)
    out = []
    for g in (True, False):
        take = min(n // 2, len(groups[g]))
        out += rng.sample(groups[g], take)
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--configs", nargs="*",
                    default=["keyword", "semantic", "hybrid"])
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--depth", type=int, default=10,
                    help="lay top-K cua moi cau hinh vao pool")
    ap.add_argument("--seed", type=int, default=13)
    # Doi ten khi file cu dang mo trong Excel (Windows khoa ghi).
    ap.add_argument("--suffix", default="",
                    help="hau to cho ten file dau ra, vd _v2")
    ap.add_argument("--from-runs", default="",
                    help="dung lai pool tu pooling_runs*.jsonl co san, bo qua "
                         "retrieval -- giu nguyen tap cap, chi dung lai cot text")
    ap.add_argument("--desc-chars", type=int, default=1500)
    args = ap.parse_args()

    docs = load_jsonl(args.corpus)
    by_key = {d["key"]: d for d in docs}
    configs, depth = args.configs, args.depth
    runs, pool = [], defaultdict(set)

    if args.from_runs:
        runs = load_jsonl(args.from_runs)
        for r in runs:
            pool[r["query_key"]].add(r["doc_key"])
        # runs duoc ghi theo thu tu truy van -> thu tu xao ben duoi trung ban goc
        queries = [by_key[k] for k in dict.fromkeys(r["query_key"] for r in runs)]
        configs = list(dict.fromkeys(r["config"] for r in runs))
        depth = max(r["rank"] for r in runs)
        print(f"corpus {len(docs)} | {len(queries)} truy van tu {args.from_runs}")
    else:
        queries = pick_queries(docs, args.n_queries, args.seed)
        if not queries:
            sys.exit("khong chon duoc truy van nao")
        print(f"corpus {len(docs)} | {len(queries)} truy van "
              f"| top-{depth} moi cau hinh | {len(configs)} cau hinh")

        # Truy van la ca noi dung issue <-> tai lieu la ca noi dung issue: DOI XUNG.
        dense = DenseIndex(args.model, device="cpu", batch_size=32,
                           query_prefix=E5_QUERY, passage_prefix=E5_QUERY)
        eng = SearchEngine(docs, dense=dense)
        eng.build_bm25()
        if any(CONFIGS[c]["use_dense"] for c in configs):
            print("  dang nhung corpus (doc tu cache neu da co)...")
            eng.build_dense(show_progress=False)

        for qi, q in enumerate(queries, 1):
            qtext = q.get("text_dense") or q.get("summary", "")
            self_idx = {eng.key_to_idx[q["key"]]} if q["key"] in eng.key_to_idx else set()
            for name in configs:
                cfg = SearchConfig(name=name, top_k=depth, **CONFIGS[name])
                ranked = eng.search(qtext, cfg, exclude=self_idx)[:depth]
                for rank, idx in enumerate(ranked, 1):
                    k = eng.keys[idx]
                    runs.append({"query_key": q["key"], "config": name,
                                 "rank": rank, "doc_key": k})
                    pool[q["key"]].add(k)
            if qi % 10 == 0:
                print(f"    ... {qi}/{len(queries)}", flush=True)

    rng = random.Random(args.seed + 1)
    rows = []
    for q in queries:
        cand = sorted(pool[q["key"]])
        rng.shuffle(cand)                 # xao: khong de lo cau hinh nao dung dau
        q_desc = describe(q, args.desc_chars)
        for k in cand:
            rows.append(judgment_row(q, by_key[k], args.desc_chars, q_desc))

    os.makedirs(args.out_dir, exist_ok=True)
    sfx = args.suffix
    j_path = os.path.join(args.out_dir, f"pooling_judgments{sfx}.csv")
    r_path = os.path.join(args.out_dir, f"pooling_runs{sfx}.jsonl")
    s_path = os.path.join(args.out_dir, f"pooling_stats{sfx}.json")

    # File gan nhan co the da duoc dien -- ghi de la mat cong nguoi gan.
    if os.path.exists(j_path):
        with open(j_path, encoding="utf-8-sig", newline="") as f:
            n_graded = sum(1 for r in csv.DictReader(f) if (r.get("grade") or "").strip())
        if n_graded:
            sys.exit(f"{j_path} da co {n_graded} dong duoc gan nhan -- "
                     "doi --suffix hoac --out-dir, khong ghi de")

    with open(j_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(r_path, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sizes = [len(pool[q["key"]]) for q in queries]
    uniq_by_cfg = Counter()
    seen_pairs = defaultdict(set)
    for r in runs:
        seen_pairs[(r["query_key"], r["doc_key"])].add(r["config"])
    for cfgs in seen_pairs.values():
        if len(cfgs) == 1:
            uniq_by_cfg[next(iter(cfgs))] += 1
    n_res = sum(r["doc_has_resolution"] for r in rows)
    stats = {
        "n_queries": len(queries), "depth": depth,
        "judgments_with_resolution": n_res,
        "judgments_empty_doc_desc": sum(1 for r in rows if not r["doc_desc"]),
        "judgments_doc_desc_clipped": sum(1 for r in rows if r["doc_desc"].endswith("…[cắt]")),
        "configs": configs, "n_judgments": len(rows),
        "pool_size_min": min(sizes), "pool_size_max": max(sizes),
        "pool_size_mean": round(sum(sizes) / len(sizes), 1),
        "lexical_group_queries": sum(1 for q in queries if q.get("lexical_group")),
        "unique_contribution": dict(uniq_by_cfg),
    }
    with open(s_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"\ncan gan nhan : {len(rows)} phan quyet")
    print(f"pool moi truy van: min={stats['pool_size_min']} "
          f"tb={stats['pool_size_mean']} max={stats['pool_size_max']}")
    print(f"truy van nhom lexical: {stats['lexical_group_queries']}/{len(queries)}")
    print("tai lieu CHI mot cau hinh tim ra (do do gia tri rieng cua tung cau hinh):")
    for k, v in uniq_by_cfg.most_common():
        print(f"  {k:<12}{v:>6}")
    print(f"\n-> {j_path}   (dien cot `grade`: 2 / 1 / 0 / ?)")
    print(f"-> {r_path}   (GIU RIENG, khong mo khi gan nhan)")
    print(f"-> {s_path}")


if __name__ == "__main__":
    main()
