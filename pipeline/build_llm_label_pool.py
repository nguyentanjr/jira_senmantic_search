# -*- coding: utf-8 -*-
"""
Dung POOL va PROMPT de LLM gan nhan quan he cap issue (tien-gan-nhan).

    python pipeline/build_llm_label_pool.py

LLM KHONG phai gold truth cuoi cung. Quy trinh:

    pool cap ung vien -> LLM gan nhan (2 chieu) -> loc theo tinh nhat quan
    -> nguoi audit mot phan -> bao cao kappa -> silver/gold

HAI TANG, BAO CAO RIENG
Do tren chinh corpus nay: chi 6,3% issue co "Root Cause" trong comment, 13,7%
co root cause HOAC solution, va 27% khong co comment nao. Tren cac pool cap ung
vien, ty le CA HAI phia deu co bang chung chi 10-16%.

Hau qua: neu bat LLM tra loi "cung underlying defect khong" cho moi cap -- va
cho phep no tra UNCERTAIN khi thieu bang chung, dung nhu phai the -- thi 85-90%
cap se ra UNCERTAIN. Nen tach lam hai tang, hoi hai cau hoi KHAC NHAU:

  TANG A (co can cu)  ca hai phia co root cause / solution.
                      Hoi: "cung underlying defect khong?"
                      Day la nhan manh nhat co the co ma khong phai di hoi ky su.

  TANG B (hanh vi)    phan con lai. Root cause la bang chung HOI CUU va phan lon
                      vang mat, nen no khong the lam dinh nghia van hanh tai thoi
                      diem truy van. Hoi thu quyet dinh duoc tu van ban:
                      "cung chuc nang, cung dieu kien, cung bieu hien loi?"
                      Nhan tang B phai duoc goi dung ten trong bao cao:
                      "behavioral duplicate", KHONG phai "same defect".
                      Day cung la dinh nghia trieu chung Bugzilla/Eclipse dung
                      khi danh dau duplicate (ho cung chua co patch luc do).

CHONG RO RI -- BAT BUOC
Tap gold hien tai duoc DAO RA tu chinh cau nhac trong comment ("trung bug
RO-10049"). Neu dua comment tho cho LLM thi no doc duoc cau ay va chi viec chep
lai nhan cu -- tap nhan "doc lap" tro thanh ban sao cua tap cu.
  1. che moi ma issue  (RO-xxxx -> <ISSUE>)
  2. bo moi dong chua tu khoa cue (trung / tuong tu / giong bug / da log bug...)
  3. KHONG dua truong `cue`, `links`, `dup_of` vao prompt

HAI CHIEU DE HIEU CHUAN
Quan he nay doi xung, nen moi cap duoc hoi hai lan: (A,B) va (B,A). Bat dong
khi dao la phep do hieu chuan truc tiep, khong ton them nhan nao. Loc theo
"nhat quan khi dao" chat hon nhieu so voi loc theo confidence LLM tu khai --
confidence tu khai hieu chuan kem, va no tu tin nhat dung luc bi danh lua boi
van ban giong nhau.

Dau ra:
  data/processed/llm_label_requests.jsonl   moi dong mot lan goi (2 dong / cap)
  data/processed/llm_label_pool.jsonl       metadata cap (khong co prompt)
  data/processed/llm_pool_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ISSUE_RE = re.compile(r"\b[A-Z]{2,10}-\d+\b")
CUE_RE = re.compile(
    r"trùng|tương tự|giống\s*bug|giống\s*lỗi|đã\s*log\s*bug|đã\s*có\s*bug"
    r"|duplicate|dup\b", re.I)
RC_RE = re.compile(
    r"(root\s*cause|nguyên nhân|nguyen nhan|solution|cách xử lý|đã fix|fix:)",
    re.I)


def mask(s: str) -> str:
    return ISSUE_RE.sub("<ISSUE>", s or "")


def drop_cue_lines(s: str) -> str:
    """Bo dong chua tu khoa cue. Tha bo nham con hon de lot nhan cu."""
    return "\n".join(ln for ln in (s or "").splitlines()
                     if not CUE_RE.search(ln))


def rc_excerpt(comments: str, max_chars: int = 500) -> str:
    """Trich doan tu cho xuat hien root cause / solution tro di."""
    c = drop_cue_lines(comments)
    m = RC_RE.search(c)
    return mask(c[m.start():m.start() + max_chars].strip()) if m else ""


def has_evidence(d: dict) -> bool:
    """Bang chung phai con song SAU khi cat cue, vi day la thu prompt hien ra.

    Kiem tren van ban tho thi phan tang sai: mot phan dang ke cai goi la "root
    cause" trong corpus nay that ra chi la tham chieu cheo sang issue khac --
    'Root Cause va Solution tuong tu RO-11923'. Cau do vua duoc tinh la bang
    chung, vua CHINH LA cau cue sinh ra nhan gold. Cat cue xong thi khong con
    gi, va cap do roi vao tang A voi mot prompt rong. Do tren pool: 38/158 cap
    tang A bi dung loi nay truoc khi sua.
    """
    return bool(rc_excerpt(d.get("text_comments", "")))


def render_issue(d: dict, label: str) -> str:
    sec = d.get("sections") or {}
    f = d.get("facets") or {}
    parts = [f"### Issue {label}",
             f"Tiêu đề: {mask(d.get('summary', ''))}"]
    for key, name in (("steps", "Các bước"), ("actual", "Kết quả thực tế"),
                      ("expected", "Kết quả mong đợi")):
        v = mask(drop_cue_lines(sec.get(key, "")))[:700].strip()
        if v:
            parts.append(f"{name}: {v}")
    rc = rc_excerpt(d.get("text_comments", ""))
    if rc:
        parts.append(f"Nguyên nhân / cách xử lý đã ghi nhận: {rc}")
    meta = [f"{k}={f.get(k)}" for k in ("severity", "test_env", "defect_type")
            if f.get(k)]
    sig = d.get("signals") or {}
    if sig.get("error_code"):
        meta.append("error_code=" + ",".join(sig["error_code"][:5]))
    if meta:
        parts.append("Thông tin khác: " + "; ".join(meta))
    return "\n".join(parts)


QUESTION = {
    "A": ("Hai issue này có cùng một lỗi nền (underlying defect) không? "
          "Cả hai đều có ghi nhận nguyên nhân/cách xử lý — hãy dựa vào đó."),
    "B": ("Hai issue này có mô tả cùng một lỗi quan sát được không: cùng chức "
          "năng, cùng điều kiện, cùng biểu hiện sai? Không có ghi nhận nguyên "
          "nhân, nên KHÔNG suy đoán về nguyên nhân."),
}

RULES = """Quy tắc bắt buộc:
- Văn bản giống nhau KHÔNG có nghĩa là trùng lặp. Hai issue có thể dùng gần như
  cùng một câu chữ mà vẫn là hai lỗi khác nhau.
- Thực thể khác nhau (ngân hàng, đối tác, kênh, tài khoản) KHÔNG tự động có
  nghĩa là hai lỗi khác nhau: chúng vẫn có thể cùng một lỗi nền ở tầng dưới.
  Cũng không tự động có nghĩa là cùng lỗi.
- Nếu không đủ bằng chứng để kết luận, trả UNCERTAIN. UNCERTAIN là câu trả lời
  hợp lệ và được khuyến khích, không phải thất bại.

Nhãn:
  DUPLICATE  cùng một lỗi (theo câu hỏi ở trên)
  RELATED    khác lỗi, nhưng liên quan về chức năng / luồng nghiệp vụ /
             service / module / biểu hiện
  UNRELATED  không liên quan
  UNCERTAIN  không đủ bằng chứng để chọn một trong ba nhãn trên

Chỉ trả về JSON, không thêm chữ nào khác:
{"label": "...", "confidence": 0.0, "reason": "<tối đa 25 từ>"}"""


def build_prompt(da: dict, db: dict, tier: str) -> str:
    return "\n\n".join([
        "Bạn đang phân loại quan hệ giữa hai bug report của một hệ thống ví "
        "điện tử. Nội dung trộn tiếng Việt và tiếng Anh.",
        QUESTION[tier],
        render_issue(da, "A"),
        render_issue(db, "B"),
        RULES,
    ])


def load_pairs(path: str, ka: str = "a", kb: str = "b") -> list[tuple]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            a, b = d.get(ka), d.get(kb)
            if a and b and a != b:
                out.append((a, b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--include-gold", action="store_true", default=True,
                    help="them cap gold vao pool de do dong thuan LLM vs cue "
                         "(cue KHONG duoc dua vao prompt)")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    docs = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs[d["key"]] = d

    sources = {
        "near_duplicates": load_pairs(f"{args.out_dir}/near_duplicates.jsonl"),
        "suspected": load_pairs(
            f"{args.out_dir}/suspected_duplicates_for_review.jsonl"),
        "hard_negative": load_pairs(f"{args.out_dir}/hard_negatives.jsonl",
                                    "anchor", "negative"),
    }
    if args.include_gold:
        sources["gold_check"] = load_pairs(
            f"{args.out_dir}/dup_candidates_from_text.jsonl")

    seen, pool = set(), []
    for src, pairs in sources.items():
        for a, b in pairs:
            if a not in docs or b not in docs:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            x, y = key
            tier = "A" if (has_evidence(docs[x]) and has_evidence(docs[y])) \
                else "B"
            pool.append({"pair_id": f"P{len(pool):05d}", "a": x, "b": y,
                         "source": src, "tier": tier})
    if args.max_pairs:
        pool = pool[: args.max_pairs]

    os.makedirs(args.out_dir, exist_ok=True)
    pool_path = os.path.join(args.out_dir, "llm_label_pool.jsonl")
    req_path = os.path.join(args.out_dir, "llm_label_requests.jsonl")
    stats_path = os.path.join(args.out_dir, "llm_pool_stats.json")

    with open(pool_path, "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_req = 0
    with open(req_path, "w", encoding="utf-8") as f:
        for p in pool:
            da, db = docs[p["a"]], docs[p["b"]]
            for order, (x, y) in (("ab", (da, db)), ("ba", (db, da))):
                f.write(json.dumps({
                    "request_id": f"{p['pair_id']}_{order}",
                    "pair_id": p["pair_id"], "order": order,
                    "tier": p["tier"], "a": p["a"], "b": p["b"],
                    "prompt": build_prompt(x, y, p["tier"]),
                }, ensure_ascii=False) + "\n")
                n_req += 1

    from collections import Counter
    stats = {
        "n_pairs": len(pool),
        "n_requests": n_req,
        "by_tier": dict(Counter(p["tier"] for p in pool)),
        "by_source": dict(Counter(p["source"] for p in pool)),
        "by_tier_source": {f"{p}|{s}": c for (p, s), c in
                           Counter((x["tier"], x["source"]) for x in pool).items()},
        "est_prompt_chars_p50": sorted(
            len(build_prompt(docs[p["a"]], docs[p["b"]], p["tier"]))
            for p in pool)[len(pool) // 2] if pool else 0,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    print(f"cap trong pool : {len(pool)}")
    print(f"luot goi LLM   : {n_req}  (moi cap hoi 2 chieu)")
    print(f"tang A (co can cu) : {stats['by_tier'].get('A', 0)}")
    print(f"tang B (hanh vi)   : {stats['by_tier'].get('B', 0)}")
    print("theo nguon:")
    for k, v in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {k:<16}{v:>6}")
    print(f"do dai prompt p50  : ~{stats['est_prompt_chars_p50']} ky tu")
    print(f"\n-> {pool_path}\n-> {req_path}\n-> {stats_path}")


if __name__ == "__main__":
    main()
