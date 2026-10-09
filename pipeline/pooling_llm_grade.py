# -*- coding: utf-8 -*-
"""
Cho LLM dien `grade`/`note` con trong cua bo pooling, THEO CHUAN cua cac dong
nguoi gan nhan da dien -- khong theo dinh nghia rieng cua LLM.

    python pipeline/pooling_llm_grade.py calibrate --dry-run
    python pipeline/pooling_llm_grade.py calibrate     # LLM co theo chuan khong?
    python pipeline/pooling_llm_grade.py fill          # dien phan con trong
    python pipeline/pooling_llm_grade.py review        # cham lai dong REVIEW (file da fill)

NGUYEN TAC
  1. Dong da co grade la CHUAN. Khong bao gio ghi de, ke ca khi LLM nghi khac.
  2. LLM chi hoc chuan qua VI DU: moi dong da gan, tru REVIEW va tru nhan do
     chinh LLM dien (LLM_SOURCES). Prompt khong
     dua them tieu chi nao ngoai ten bon muc cua thang nhan -- muc do khat khe,
     ly do len 2 hay xuong 0 deu lay tu vi du. Vi du cua CHINH truy van dang
     cham duoc tach rieng lam tham chieu gan nhat.
  3. Dong REVIEW (note so issue voi sai truy van) khong dung lam vi du va giu
     nguyen trong dau ra de nguoi sua.
  4. Moi truy van mot luot goi, ket qua tra ve THEO doc_key va duoc kiem tra
     du / khong thua / khong trung. Sai dinh dang thi goi lai kem thong bao
     loi; van sai thi de trong, khong doan.
  5. `calibrate` lan luot giau vi du cua tung truy van co nhan, cho LLM cham
     lai cac dong do va so voi nhan co san. Chua chay buoc nay thi chua biet
     LLM co theo chuan hay khong -- va con so nay phai di kem khi bao cao.

Dau ra (file dau vao KHONG bi sua):
  data/processed/pooling_judgments_v3_llm.csv    them cot label_source, llm_model
  data/processed/pooling_llm_calibration_v3.json
  data/processed/pooling_llm_calibration_v3.csv  tung cap nguoi vs LLM
  data/processed/pooling_llm_cache_v3.jsonl      phan hoi da hop le (de chay tiep)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_label import make_caller  # noqa: E402

GRADES = ("0", "1", "2", "?")
JSON_RE = re.compile(r"\{.*\}", re.S)

#: Dong da gan nhung note so voi SAI truy van (soat ngay 2026-09-11): note cua
#: RO-10840 / RO-17986 ta RO-6025 / RO-8406; 5 dong dau cua RO-19734 lap luan
#: nhu truy van la RO-11708. Dung lam vi du thi LLM hoc luon cai sai.
REVIEW = {
    ("RO-11708", "RO-10840"), ("RO-11708", "RO-17986"),
    ("RO-19734", "RO-22318"), ("RO-19734", "RO-17551"),
    ("RO-19734", "RO-18539"), ("RO-19734", "RO-18536"),
    ("RO-19734", "RO-19134"),
}

HEADER = """Bạn điền nhãn cho bộ đánh giá tìm kiếm issue Jira (QA nội bộ ví điện tử).

Câu hỏi cho mỗi ứng viên B: "Nếu tôi đang điều tra issue A và hệ thống đưa B lên, B có giúp tôi không?"
  2 = giúp trực tiếp
  1 = đáng đọc, có liên quan
  0 = không giúp gì
  ? = không đủ thông tin để quyết định

CHUẨN CHẤM LÀ CỦA NGƯỜI GÁN NHÃN, KHÔNG PHẢI CỦA BẠN.
- Các VÍ DỤ bên dưới do người gán nhãn chấm. Chấm ứng viên mới đúng như cách họ đã chấm: cùng mức khắt khe, cùng loại lý do để lên 2 hay xuống 0.
- Không tự đặt tiêu chí mới. Phân vân giữa hai mức thì chọn mức mà ví dụ giống nhất đã được chấm.
- Mô tả của B quá ít để so thì dùng "?", không đoán.
- Note: 1–2 câu tiếng Việt, cùng văn phong với ví dụ, nêu điểm giống và khác cụ thể giữa A và B (luồng, điều kiện, mã lỗi). Chỉ nói về đúng A và B đang xét, chỉ dùng thông tin có trong mô tả — không suy diễn trạng thái, luồng hay mã lỗi không được nêu.

QUY TẮC NGƯỜI GÁN ĐÃ CHỐT (áp dụng cho mọi ứng viên):
1. B khác luồng nghiệp vụ / channel / API / portal so với A nhưng cùng cơ chế lỗi: CHỈ được 2 khi B có mục "Xử lý đã ghi nhận" nêu nguyên nhân hoặc cách sửa áp dụng được cho A. "Xử lý đã ghi nhận: (không có)" thì tối đa 1.
2. Loại lỗi quan trọng hơn trường dữ liệu / màn hình: cùng loại lỗi nhưng ở trường khác = 1; cùng trường hoặc cùng màn hình nhưng loại lỗi khác = 0."""

FOOTER = """Với mỗi ứng viên, xác định same_flow TRƯỚC rồi mới chấm grade:
  same_flow = true CHỈ KHI A và B cùng chức năng/thao tác (vd cùng Create Reversal Outbound, cùng Pay Tanz QR) VÀ cùng kênh (USSD / APP / WEB / API) VÀ cùng portal nếu có (Enterprise / Client Operation / Internal Operation / Channel).
  Khác thao tác (Create / Reject / Approve…), khác kênh, khác portal hoặc khác API → same_flow = false.
  same_flow CHỈ dùng cho quy tắc 1 (giới hạn mức 2). Khác flow KHÔNG phải lý do để chấm 0: chọn 0 hay 1 vẫn theo loại lỗi (quy tắc 2) — cùng loại lỗi thì ít nhất là 1 dù khác portal/kênh.

Trả về DUY NHẤT một JSON, không kèm chữ nào khác:
{"judgments": [{"doc_key": "RO-...", "same_flow": false, "grade": "1", "note": "..."}]}
Đúng một phần tử cho mỗi ứng viên trong mục ỨNG VIÊN CẦN CHẤM — không thêm, không bớt, giữ nguyên doc_key."""


# ------------------------------------------------------------------ io
def load_rows(path: str) -> tuple[list[dict], list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])


def is_labeled(r: dict) -> bool:
    return (r.get("grade") or "").strip() in GRADES


def group_by_query(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["query_key"], []).append(r)
    return out


#: Nhan do LLM dien. Khong bao gio lam vi du (chuan cham se troi theo chinh LLM);
#: khi fill mot file DA fill (vd pool mo rong v4) thi giu nguyen, khong cham lai.
LLM_SOURCES = ("llm", "llm_review")


def examples_of(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if is_labeled(r) and (r["query_key"], r["doc_key"]) not in REVIEW
            and (r.get("label_source") or "") not in LLM_SOURCES]


# -------------------------------------------------------------- prompt
def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + " …"


def render_query(r: dict, desc_chars: int) -> str:
    return (f"A = {r['query_key']}: {r['query_summary']}\n"
            f"Mã lỗi: {r.get('query_error_codes') or '-'}\n"
            f"Mô tả A:\n{_clip(r['query_desc'], desc_chars)}")


def render_doc(r: dict, desc_chars: int, res_chars: int) -> str:
    parts = [f"B = {r['doc_key']}: {r['doc_summary']}",
             f"Trạng thái: {r.get('doc_status') or '-'} | "
             f"Mã lỗi: {r.get('doc_error_codes') or '-'}",
             f"Mô tả B:\n{_clip(r['doc_desc'], desc_chars)}"]
    # Luon ghi dong nay: quy tac 1 can LLM biet ro la KHONG co cach sua.
    res = (r.get("doc_resolution") or "").strip()
    parts.append("Xử lý đã ghi nhận: " + (_clip(res, res_chars) if res else "(không có)"))
    return "\n".join(parts)


def render_labeled(r: dict) -> str:
    return (render_doc(r, 300, 200)
            + f"\n→ grade {r['grade'].strip()} | note: {(r.get('note') or '').strip()}")


def build_prompt(qrow: dict, targets: list[dict], examples: list[dict],
                 desc_chars: int = 1500, fmt_err: str = "") -> str:
    q = qrow["query_key"]
    same = [e for e in examples if e["query_key"] == q]
    other = [e for e in examples if e["query_key"] != q]
    parts = [HEADER]
    if other:
        blocks = []
        for rs in group_by_query(other).values():
            blocks.append("### Truy vấn " + render_query(rs[0], 500) + "\n\n"
                          + "\n\n".join(render_labeled(r) for r in rs))
        parts += ["## VÍ DỤ NGƯỜI GÁN ĐÃ CHẤM (các truy vấn khác)",
                  "\n\n=====\n\n".join(blocks)]
    parts += ["## ISSUE ĐANG ĐIỀU TRA", render_query(qrow, desc_chars)]
    if same:
        parts += ["## ỨNG VIÊN CỦA CHÍNH A ĐÃ ĐƯỢC NGƯỜI GÁN CHẤM "
                  "(tham chiếu gần nhất, không chấm lại)",
                  "\n\n".join(render_labeled(r) for r in same)]
    parts += [f"## ỨNG VIÊN CẦN CHẤM ({len(targets)})",
              "\n\n---\n\n".join(render_doc(t, desc_chars, 600) for t in targets),
              FOOTER]
    if fmt_err:
        parts.append(f"LƯU Ý: lần trả lời trước không hợp lệ ({fmt_err}). "
                     "Trả lại đầy đủ, đúng định dạng.")
    return "\n\n".join(parts)


# --------------------------------------------------------------- parse
def parse_judgments(text: str, expected: set[str]) -> tuple[dict, str]:
    """-> ({doc_key: (grade, note)}, loi). Sai bat ky dieu gi thi tra {} + loi."""
    m = JSON_RE.search(text or "")
    if not m:
        return {}, "khong tim thay JSON"
    try:
        items = json.loads(m.group(0)).get("judgments")
    except (json.JSONDecodeError, AttributeError):
        return {}, "JSON khong doc duoc"
    if not isinstance(items, list):
        return {}, "thieu mang judgments"
    out: dict[str, tuple[str, str]] = {}
    for it in items:
        if not isinstance(it, dict):
            return {}, "phan tu khong phai object"
        k = str(it.get("doc_key", "")).strip()
        g = str(it.get("grade", "")).strip()
        if k not in expected:
            return {}, f"doc_key khong nam trong danh sach: {k}"
        if k in out:
            return {}, f"doc_key bi lap: {k}"
        if g not in GRADES:
            return {}, f"grade khong hop le o {k}: {g}"
        sf = it.get("same_flow")
        if isinstance(sf, str):
            sf = {"true": True, "false": False}.get(sf.strip().lower())
        if not isinstance(sf, bool):
            return {}, f"same_flow khong hop le o {k}: {it.get('same_flow')}"
        out[k] = (g, str(it.get("note", "")).strip(), sf)
    missing = expected - out.keys()
    if missing:
        return {}, f"thieu {len(missing)} doc_key, vd {sorted(missing)[:3]}"
    return out, ""


# ---------------------------------------------------------------- core
RULE1_TAG = "[Quy tắc 1: khác flow, không có cách sửa ghi nhận → 1]"


def enforce_rule1(row: dict, grade: str, note: str, same_flow: bool) -> tuple[str, str]:
    """Quy tac 1 do nguoi gan chot: khac flow ma khong co cach sua -> toi da 1.

    Ap bang code vi lan calibrate thu hai LLM vi pham 10/45 dong du prompt da
    hien "(khong co)". LLM chi con tra loi cau hep hon -- co cung flow khong --
    va moi dong bi ha deu mang RULE1_TAG de soat lai.
    """
    has_res = bool((row.get("doc_resolution") or "").strip())
    if grade == "2" and not same_flow and not has_res:
        return "1", (note + " " + RULE1_TAG).strip()
    return grade, note


def fill_rows(rows: list[dict], judge, model: str, batch_size: int = 0) -> Counter:
    """Dien dong trong theo tung truy van. `judge(qrow, targets, examples)`
    tra ve ({doc_key: (grade, note, same_flow)}, loi). Dong da gan khong bao
    gio bi dong.

    `batch_size`: chia ung vien cua mot truy van thanh nhieu luot goi. Do khi
    fill: truy van 29 ung vien thi JSON tra ve bi cat (sai dinh dang ca 4 lan),
    trong khi calibrate chi toi 17 ung vien/luot va khong loi lan nao.
    """
    examples = examples_of(rows)
    stat = Counter()
    for r in rows:
        if is_labeled(r) and (r.get("label_source") or "") in LLM_SOURCES:
            # dau vao da qua fill (vd pool mo rong): giu nguyen nhan va nguon goc
            stat["llm_cu"] += 1
            continue
        r["llm_grade_raw"], r["llm_same_flow"] = "", ""
        if is_labeled(r):
            src = "review" if (r["query_key"], r["doc_key"]) in REVIEW else "user"
            r["label_source"], r["llm_model"] = src, ""
            stat[src] += 1
        else:
            r["label_source"], r["llm_model"] = "", ""
    for q, qrows in group_by_query(rows).items():
        targets = [r for r in qrows if not is_labeled(r)]
        if not targets:
            continue
        size = batch_size or len(targets)
        for i in range(0, len(targets), size):
            chunk = targets[i:i + size]
            got, err = judge(qrows[0], chunk, examples)
            if err:
                stat["loi_truy_van"] += 1
                print(f"  {q}: BO QUA {len(chunk)} ung vien ({err})", flush=True)
                continue
            for r in chunk:
                g, n, sf = got[r["doc_key"]]
                r["llm_grade_raw"], r["llm_same_flow"] = g, int(sf)
                r["grade"], r["note"] = enforce_rule1(r, g, n, sf)
                r["label_source"], r["llm_model"] = "llm", model
                stat["llm"] += 1
                stat["ha_theo_quy_tac_1"] += r["grade"] != g
    return stat


def regrade_review_rows(rows: list[dict], judge, model: str) -> Counter:
    """Cham lai cac dong label_source=review trong file DA fill.

    Vi du chi lay tu dong `user` -- KHONG lay dong `llm`, neu khong chuan cham
    se troi theo chinh LLM. Nhan cu duoc giu o review_old_grade/review_old_note.
    """
    examples = [r for r in rows if r.get("label_source") == "user"]
    stat = Counter()
    for q, qrows in group_by_query(rows).items():
        targets = [r for r in qrows if r.get("label_source") == "review"]
        if not targets:
            continue
        got, err = judge(qrows[0], targets, examples)
        if err:
            stat["loi_truy_van"] += 1
            print(f"  {q}: BO QUA {len(targets)} dong review ({err})", flush=True)
            continue
        for r in targets:
            g, n, sf = got[r["doc_key"]]
            r["review_old_grade"], r["review_old_note"] = r["grade"], r["note"]
            r["llm_grade_raw"], r["llm_same_flow"] = g, int(sf)
            r["grade"], r["note"] = enforce_rule1(r, g, n, sf)
            r["label_source"], r["llm_model"] = "llm_review", model
            stat["llm_review"] += 1
    return stat


def calibrate_rows(rows: list[dict], judge) -> list[dict]:
    """Giau vi du cua tung truy van, cho LLM cham lai, so voi nhan nguoi."""
    examples = examples_of(rows)
    by_q = group_by_query(rows)
    out = []
    for q, own in group_by_query(examples).items():
        others = [e for e in examples if e["query_key"] != q]
        got, err = judge(by_q[q][0], own, others)
        if err:
            print(f"  {q}: BO QUA ({err})", flush=True)
            continue
        for r in own:
            g, n, sf = got[r["doc_key"]]
            ge, ne = enforce_rule1(r, g, n, sf)
            out.append({"query_key": q, "doc_key": r["doc_key"],
                        "user_grade": r["grade"].strip(), "llm_grade": ge,
                        "llm_grade_raw": g, "llm_same_flow": int(sf),
                        "user_note": (r.get("note") or "").strip(), "llm_note": ne})
    return out


def weighted_kappa(pairs: list[tuple[str, str]], cats=("0", "1", "2")) -> float:
    """Kappa trong so bac hai -- lech 0<->2 bi phat nang hon lech 1 bac."""
    pairs = [(a, b) for a, b in pairs if a in cats and b in cats]
    n, k = len(pairs), len(cats)
    if not n:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    ra = [sum(obs[i]) for i in range(k)]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else 1.0


# ---------------------------------------------------------------- llm
def make_judge(args, mode: str):
    """Boc LLM: cache theo (mode, model, truy van, tap doc_key), retry, han muc."""
    cache: dict[tuple, dict] = {}
    if os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    cache[(x["mode"], x["model"], x["query_key"],
                           tuple(x["doc_keys"]), x.get("prompt_sha", ""))] = x
    call = None if args.dry_run else make_caller(args)
    n_prompts = [0, 0]

    def judge(qrow, targets, examples):
        keys = sorted(t["doc_key"] for t in targets)
        prompt = build_prompt(qrow, targets, examples, args.desc_chars)
        # Khoa theo NOI DUNG prompt: sua quy tac hay nhan mau thi phai goi lai,
        # khong duoc tra ket qua cham theo chuan cu.
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        ck = (mode, args.model, qrow["query_key"], tuple(keys), sha)
        if ck in cache:
            return {k: tuple(v) for k, v in cache[ck]["judgments"].items()}, ""
        n_prompts[0] += 1
        n_prompts[1] += len(prompt)
        if args.dry_run:
            if args.dump_prompt and n_prompts[0] == 1:
                with open(args.dump_prompt, "w", encoding="utf-8") as f:
                    f.write(prompt)
            return {}, "dry-run"
        fmt_err = ""
        for attempt in range(args.retries):
            prompt = build_prompt(qrow, targets, examples, args.desc_chars, fmt_err)
            t0 = time.time()
            print(f"  {qrow['query_key']}: goi LLM lan {attempt + 1} "
                  f"({len(targets)} ung vien)...", flush=True)
            try:
                text = call(prompt)
            except Exception as e:                               # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                print(f"    loi API sau {time.time() - t0:.0f}s: {msg[:200]}", flush=True)
                # Free tier co 2 loai 429; loai theo NGAY thi cho bao lau cung vo ich.
                if "PerDay" in msg:
                    # In nguyen van: "limit: 0" nghia la model khong co free
                    # tier -- cho sang ngay mai cung vo ich, phai doi model.
                    sys.exit(f"\nHET HAN MUC NGAY cho {args.model}. Chay lai "
                             "sau (cache giu phan da xong) hoac doi --model."
                             f"\nloi goc: {msg[:1500]}")
                time.sleep(args.quota_wait
                           if ("RESOURCE_EXHAUSTED" in msg or "429" in msg)
                           else min(2 ** attempt, 30))
                fmt_err = ""
                continue
            finally:
                if args.sleep:
                    time.sleep(args.sleep)
            got, fmt_err = parse_judgments(text, set(keys))
            print(f"    tra loi sau {time.time() - t0:.0f}s: "
                  + (f"SAI DINH DANG ({fmt_err})" if fmt_err else "hop le"), flush=True)
            if not fmt_err:
                rec = {"mode": mode, "model": args.model, "prompt_sha": sha,
                       "query_key": qrow["query_key"], "doc_keys": keys,
                       "judgments": {k: list(v) for k, v in got.items()}}
                with open(args.cache, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache[ck] = rec
                return got, ""
        return {}, fmt_err or "loi API sau khi retry"

    judge.n_prompts = n_prompts
    return judge


def _report_dry(judge, n_calls_label: str) -> None:
    n, chars = judge.n_prompts
    print(f"\n--dry-run: {n} luot goi {n_calls_label}, "
          f"~{chars:,} ky tu input (~{chars // 3:,} token, tieng Viet ~3 ky tu/token)")


# ---------------------------------------------------------------- cmds
def cmd_calibrate(args) -> None:
    rows, _ = load_rows(args.inp)
    ex = examples_of(rows)
    print(f"vi du hop le: {len(ex)} dong / {len(group_by_query(ex))} truy van "
          f"(bo {sum(1 for r in rows if (r['query_key'], r['doc_key']) in REVIEW)} dong REVIEW)")
    judge = make_judge(args, "calibrate")
    res = calibrate_rows(rows, judge)
    if args.dry_run:
        _report_dry(judge, "(moi truy van co nhan 1 luot)")
        return
    if not res:
        sys.exit("khong co ket qua nao")

    pairs = [(r["user_grade"], r["llm_grade"]) for r in res]
    exact = sum(a == b for a, b in pairs) / len(pairs)
    kw = weighted_kappa(pairs)
    conf = Counter(pairs)
    summary = {"model": args.model, "n": len(pairs), "exact_agreement": round(exact, 3),
               "weighted_kappa": round(kw, 3),
               "confusion_user_x_llm": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
               "per_query": {}}
    for q in dict.fromkeys(r["query_key"] for r in res):
        sub = [(r["user_grade"], r["llm_grade"]) for r in res if r["query_key"] == q]
        summary["per_query"][q] = {"n": len(sub),
                                   "exact": round(sum(a == b for a, b in sub) / len(sub), 3)}
    with open(args.calib_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(args.calib_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    n_cap = sum(r["llm_grade"] != r["llm_grade_raw"] for r in res)
    summary["capped_by_rule1"] = n_cap
    with open(args.calib_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\nLLM vs NGUOI  model={args.model}  n={len(pairs)}  "
          f"(ha 2->1 theo quy tac 1: {n_cap} dong)")
    print(f"  trung khop tuyet doi : {exact:.1%}")
    print(f"  weighted kappa       : {kw:.3f}   (<0.4 kem | 0.4-0.6 vua | 0.6-0.8 tot | >0.8 rat tot)")
    print("  nguoi -> LLM         : " + "  ".join(f"{a}->{b}:{v}" for (a, b), v in sorted(conf.items())))
    for q, s in summary["per_query"].items():
        print(f"  {q:<10} n={s['n']:<3} trung={s['exact']:.0%}")
    print(f"\n-> {args.calib_json}\n-> {args.calib_csv}   (xem cot user_note / llm_note o dong lech)")


def cmd_fill(args) -> None:
    if os.path.abspath(args.out) == os.path.abspath(args.inp):
        sys.exit("--out trung --inp: khong ghi vao file nhan goc")
    if os.path.exists(args.out) and not args.overwrite and not args.dry_run:
        sys.exit(f"{args.out} da ton tai (co the da duoc sua tay) -- them --overwrite")
    rows, fields = load_rows(args.inp)
    if args.limit_queries:
        keep = set(list(group_by_query(rows))[: args.limit_queries])
        rows = [r for r in rows if r["query_key"] in keep]

    # Excel tung chen rac vao mot o query_summary (vd "+E4:H4") -- lay gia tri
    # pho bien nhat trong truy van cho ca prompt lan dau ra.
    for qrows in group_by_query(rows).values():
        common = Counter(r["query_summary"] for r in qrows).most_common(1)[0][0]
        for r in qrows:
            r["query_summary"] = common

    judge = make_judge(args, "fill")
    todo = {(r["query_key"], r["doc_key"]) for r in rows if not is_labeled(r)}
    stat = fill_rows(rows, judge, args.model, args.batch_size)
    if args.dry_run:
        _report_dry(judge, f"cho {len(todo)} dong trong")
        return

    out_fields = fields + [c for c in ("label_source", "llm_model", "llm_grade_raw",
                                       "llm_same_flow") if c not in fields]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    left = sum(1 for r in rows if not is_labeled(r))
    print(f"\nnguoi (user) {stat['user']} | review {stat['review']} | LLM lan nay {stat['llm']} "
          f"| LLM cu giu nguyen {stat['llm_cu']} | con trong {left} "
          f"| truy van loi {stat['loi_truy_van']} "
          f"| ha 2->1 theo quy tac 1 {stat['ha_theo_quy_tac_1']}")
    print("phan bo grade LLM lan nay: " + str(Counter(
        r["grade"] for r in rows
        if r["label_source"] == "llm" and (r["query_key"], r["doc_key"]) in todo)))
    print("phan bo grade user: " + str(Counter(r["grade"] for r in rows if r["label_source"] == "user")))
    print(f"-> {args.out}")
    if stat["review"]:
        print(f"CHU Y: {stat['review']} dong label_source=review van giu nhan cu "
              "(note so voi sai truy van) -- sua tay truoc khi chay eval.")


def cmd_review(args) -> None:
    rows, fields = load_rows(args.inp)
    if "label_source" not in fields:
        sys.exit(f"{args.inp} chua qua buoc fill (thieu cot label_source)")
    todo = sum(1 for r in rows if r["label_source"] == "review")
    print(f"dong review can cham lai: {todo}")
    if not todo:
        return
    judge = make_judge(args, "review")
    stat = regrade_review_rows(rows, judge, args.model)
    if args.dry_run:
        _report_dry(judge, f"cho {todo} dong review")
        return

    bk = os.path.splitext(args.inp)[0] + ".before_review.csv"
    if not os.path.exists(bk):
        shutil.copy2(args.inp, bk)
    out_fields = fields + [c for c in ("review_old_grade", "review_old_note")
                           if c not in fields]
    try:
        with open(args.inp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        sys.exit(f"{args.inp} dang mo (Excel?) -- dong file roi chay lai")
    print(f"\ncham lai {stat['llm_review']} dong | truy van loi {stat['loi_truy_van']} "
          f"| sao luu: {bk}")
    for r in rows:
        if r["label_source"] == "llm_review":
            print(f"  {r['query_key']} {r['doc_key']}: {r['review_old_grade']} -> {r['grade']}"
                  f"  | {r['note'][:120]}")
    print(f"-> {args.inp}")


def main() -> None:
    D = "data/processed"
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--inp", default=f"{D}/pooling_judgments_v3.csv")
        p.add_argument("--cache", default=f"{D}/pooling_llm_cache_v3.jsonl")
        p.add_argument("--provider", default="gemini",
                       choices=["openai", "anthropic", "gemini"])
        p.add_argument("--base-url", default="")
        p.add_argument("--api-key-env", default="")
        p.add_argument("--model", default="gemini-3.5-flash-lite")
        # Model co "thinking" tieu token output truoc khi tra JSON; 8192 lam
        # JSON bi cat o truy van dong ung vien.
        p.add_argument("--max-tokens", type=int, default=16384)
        p.add_argument("--desc-chars", type=int, default=1500)
        p.add_argument("--retries", type=int, default=3)
        p.add_argument("--sleep", type=float, default=5.0)
        p.add_argument("--quota-wait", type=float, default=65.0)
        p.add_argument("--timeout", type=float, default=180.0,
                       help="giay; qua thoi gian thi huy luot goi va thu lai")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--dump-prompt", default="",
                       help="(dry-run) ghi prompt dau tien ra file de doc")

    c = sub.add_parser("calibrate", help="do muc LLM theo chuan cua nhan nguoi")
    common(c)
    c.add_argument("--calib-json", default=f"{D}/pooling_llm_calibration_v3.json")
    c.add_argument("--calib-csv", default=f"{D}/pooling_llm_calibration_v3.csv")
    c.set_defaults(func=cmd_calibrate)

    f = sub.add_parser("fill", help="dien grade/note con trong")
    common(f)
    f.add_argument("--out", default=f"{D}/pooling_judgments_v3_llm.csv")
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--limit-queries", type=int, default=0)
    f.add_argument("--batch-size", type=int, default=15,
                   help="toi da so ung vien moi luot goi (calibrate da chay toi 17)")
    f.set_defaults(func=cmd_fill)

    v = sub.add_parser("review", help="cham lai cac dong REVIEW trong file da fill")
    common(v)
    v.set_defaults(func=cmd_review, inp=f"{D}/pooling_judgments_v3_llm.csv")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
