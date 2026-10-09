# -*- coding: utf-8 -*-
"""
Do bien the PROMPT tren tap 53 cap da co nhan nguoi.

    python pipeline/llm_prompt_eval.py --variants v1_strict v2_binary \
        --provider gemini --model gemini-3.5-flash-lite --sleep 1.5

Ly do ton tai:
Cau hinh dau tien (prompt v0, gemini-3.5-flash-lite) cho kappa = 0,179 tren 53
cap audit -- muc "rat kem". Ma tran nham lan chi ro kieu hong:

    nguoi noi DUPLICATE: 8   -> LLM bat 7   (recall 88%)
    LLM noi DUPLICATE  : 28  -> dung 7      (precision 25%)

LLM goi thua trung lap gap 3,5 lan. No khong hong ngau nhien: 80,8% on dinh khi
dao chieu, tuc SAI DEU. Gia thuyet: prompt v0 doi xu bon nhan ngang nhau va
khong bat neu bang chung, nen model chon nhan "manh" nhat khi hai van ban giong
nhau -- dung dieu prompt cam bang loi.

CANH BAO VE PHUONG PHAP -- PHAI DOC TRUOC KHI TRICH SO
53 cap nay vua la tap de TINH CHINH prompt, vua la tap DO. Lam vay la overfit:
con kappa tot nhat o day la CAN TREN, khong phai uoc luong khong thien lech.
Bien the nao thang o day BAT BUOC phai duoc xac nhan lai tren mot mau audit
MOI truoc khi dua vao luan van. Xem `llm_label.py audit` de lay mau moi.

Cung ly do do: khong nhet bat ky cap nao trong 53 cap nay vao prompt lam vi du
few-shot roi do tren chinh chung.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from argparse import Namespace
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_llm_label_pool import QUESTION, render_issue          # noqa: E402
from llm_label import LABELS, make_caller, norm_label, parse_reply  # noqa: E402

# --------------------------------------------------------------- variants
RULES_V0 = """Quy tắc bắt buộc:
- Văn bản giống nhau KHÔNG có nghĩa là trùng lặp. Hai issue có thể dùng gần như
  cùng một câu chữ mà vẫn là hai lỗi khác nhau.
- Thực thể khác nhau (ngân hàng, đối tác, kênh, tài khoản) KHÔNG tự động có
  nghĩa là hai lỗi khác nhau: chúng vẫn có thể cùng một lỗi nền ở tầng dưới.
  Cũng không tự động có nghĩa là cùng lỗi.
- Nếu không đủ bằng chứng để kết luận, trả UNCERTAIN.

Nhãn:
  DUPLICATE  cùng một lỗi (theo câu hỏi ở trên)
  RELATED    khác lỗi, nhưng liên quan về chức năng / luồng nghiệp vụ /
             service / module / biểu hiện
  UNRELATED  không liên quan
  UNCERTAIN  không đủ bằng chứng

Chỉ trả về JSON:
{"label": "...", "confidence": 0.0, "reason": "<tối đa 25 từ>"}"""

# v1: khong doi xu bon nhan ngang nhau nua. Mac dinh la RELATED, va DUPLICATE
# la mot ket luan phai TRA GIA: chi duoc chon khi neu ten duoc co che chung.
RULES_V1 = """Cách quyết định (làm đúng thứ tự này):

BƯỚC 1. Mặc định nhãn là RELATED. Hai issue nằm trong danh sách này vì chúng đã
giống nhau về từ vựng — điều đó KHÔNG phải bằng chứng.

BƯỚC 2. Chỉ nâng lên DUPLICATE nếu bạn nêu được TRONG `reason` cơ chế lỗi cụ
thể mà cả hai cùng chia sẻ — ví dụ cùng một trường bị thiếu, cùng một API trả
sai, cùng một điều kiện dữ liệu, cùng một dòng logic hỏng. Nếu `reason` của bạn
chỉ nói "cùng màn hình", "cùng module", "cùng triệu chứng", "mô tả giống nhau"
thì nhãn đúng là RELATED, không phải DUPLICATE.

BƯỚC 3. Hạ xuống UNRELATED nếu hai issue không chung chức năng lẫn luồng nghiệp
vụ. Trả UNCERTAIN nếu thông tin quá sơ sài để nói được gì.

Hai lỗi khác nhau trên cùng một màn hình là chuyện RẤT phổ biến trong dữ liệu
này: cùng chức năng, cùng cách hành văn, nhưng khác điều kiện hoặc khác bước.
Đó là RELATED.

Ngược lại, thực thể khác nhau (ngân hàng, đối tác, kênh, tài khoản) KHÔNG tự
động loại bỏ DUPLICATE: hai ngân hàng vẫn có thể hỏng vì cùng một lỗi nền.

Nhãn: DUPLICATE / RELATED / UNRELATED / UNCERTAIN

Chỉ trả về JSON:
{"label": "...", "confidence": 0.0, "reason": "<nêu cơ chế chung nếu chọn DUPLICATE, tối đa 25 từ>"}"""

# v2: nhu v1 nhung bo hai nhan khong ai dung. Tren pool near_duplicates, ca
# nguoi lan LLM deu khong dung UNRELATED/UNCERTAIN (nguoi: 0/0, LLM: 2/0) --
# bai toan thuc chat la nhi phan.
RULES_V2 = RULES_V1.replace(
    """BƯỚC 3. Hạ xuống UNRELATED nếu hai issue không chung chức năng lẫn luồng nghiệp
vụ. Trả UNCERTAIN nếu thông tin quá sơ sài để nói được gì.

""", "").replace(
    "Nhãn: DUPLICATE / RELATED / UNRELATED / UNCERTAIN",
    "Nhãn: chỉ được chọn DUPLICATE hoặc RELATED.")

# v3: v1 + doi chieu NGU CANH da boc san.
#
# Ly do: v1/v2 that bai theo dung mot kieu -- 17 cap nguoi goi RELATED thi LLM
# goi DUPLICATE, khong doi qua ca hai bien the. Doc lai cac cap do thi thay
# chung thuong cung module nhung KHAC man hinh/chuc nang:
#     [CMS agent][Dasboard commission]  vs  [CMS agent][Dasboard Transaction]
# Model doc tieu de nhu mot chuoi van xuoi va thay "gan nhu giong het". No
# khong tach duoc doan [Module][Chuc nang] ra khoi phan mo ta.
#
# Nen boc san bang `domain.parse_tags` va dat thanh bang doi chieu ro rang,
# thay vi hy vong model tu nhan ra. Trong so trong `dup/config.py` da noi
# ro muc do quan trong: actor_conflict 0.20, channel_conflict 0.15 (manh) --
# con provider_conflict chi 0.06 (yeu, vi khac ngan hang van co the cung root
# cause). Prompt phai phan anh dung thu bac do.
RULES_V3 = RULES_V1.replace(
    """Hai lỗi khác nhau trên cùng một màn hình là chuyện RẤT phổ biến trong dữ liệu
này: cùng chức năng, cùng cách hành văn, nhưng khác điều kiện hoặc khác bước.
Đó là RELATED.""",
    """DÙNG BẢNG "ĐỐI CHIẾU NGỮ CẢNH" Ở TRÊN — đây là phần quan trọng nhất:

- KÊNH khác nhau, hoặc TÁC NHÂN khác nhau  -> hai luồng nghiệp vụ khác nhau.
  Gần như chắc chắn là RELATED, không phải DUPLICATE.
- CHỨC NĂNG khác nhau (khác màn hình, khác nghiệp vụ trong cùng module)
  -> RELATED. Ví dụ "Dashboard commission" và "Dashboard transaction" là hai
  màn hình khác nhau, dù tiêu đề đọc gần như giống hệt và phần mô tả lỗi có
  thể chép nguyên si từ nhau.
- Chỉ khi KÊNH, TÁC NHÂN và CHỨC NĂNG đều trùng thì mới xét tiếp đến
  DUPLICATE — và vẫn phải nêu được cơ chế lỗi chung ở BƯỚC 2.

Hai lỗi khác nhau trên cùng một màn hình cũng RẤT phổ biến: cùng chức năng,
cùng cách hành văn, nhưng khác điều kiện hoặc khác bước. Đó cũng là RELATED.""")

VARIANTS = {"v0_baseline": RULES_V0, "v1_strict": RULES_V1,
            "v2_binary": RULES_V2, "v3_module": RULES_V3}
#: bien the nao can chen bang doi chieu ngu canh vao prompt
NEEDS_CONTEXT = {"v3_module"}


def render_context(da: dict, db: dict) -> str:
    """Bang doi chieu kenh / tac nhan / chuc nang, boc tu tieu de."""
    from domain import parse_tags
    ta, tb = parse_tags(da.get("summary", "")), parse_tags(db.get("summary", ""))
    lines = ["### Đối chiếu ngữ cảnh (bóc tự động từ tiêu đề)"]
    for key, name in (("channel", "Kênh"), ("actor", "Tác nhân"),
                      ("function", "Chức năng")):
        sa, sb = ta.get(key) or set(), tb.get(key) or set()
        fa = ", ".join(sorted(sa)) or "(không rõ)"
        fb = ", ".join(sorted(sb)) or "(không rõ)"
        if not sa or not sb:
            verdict = "KHÔNG ĐỦ THÔNG TIN"
        elif sa == sb:
            verdict = "GIỐNG"
        else:
            only = sorted(sa ^ sb)
            verdict = "KHÁC -> " + ", ".join(only)
        lines.append(f"{name:<10} A: {fa:<28} B: {fb:<28} {verdict}")
    return "\n".join(lines)

HEAD = ("Bạn đang phân loại quan hệ giữa hai bug report của một hệ thống ví "
        "điện tử. Nội dung trộn tiếng Việt và tiếng Anh.")


def build_prompt(da: dict, db: dict, tier: str, variant: str) -> str:
    parts = [HEAD, QUESTION[tier], render_issue(da, "A"), render_issue(db, "B")]
    if variant in NEEDS_CONTEXT:
        parts.append(render_context(da, db))
    parts.append(VARIANTS[variant])
    return "\n\n".join(parts)


def kappa(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    n = len(pairs) or 1
    agree = sum(1 for a, h in pairs if a == h) / n
    ca, ch = Counter(a for a, _ in pairs), Counter(h for _, h in pairs)
    pe = sum(ca[k] / n * ch[k] / n for k in set(ca) | set(ch))
    return agree, (agree - pe) / (1 - pe) if pe < 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/processed/llm_label_audit.csv")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--variants", nargs="+", default=["v1_strict"],
                    choices=sorted(VARIANTS))
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--max-tokens", type=int, default=300)
    # Free tier thuong 10-15 RPM. --sleep 1.2 (~27 RPM) lam moi vai luot lai
    # dinh 429 va phai cho 65s -- cham hon nhieu so viec di cham tu dau.
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--quota-wait", type=float, default=65.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/processed/llm_prompt_eval.json")
    args = ap.parse_args()

    with open(args.audit, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get("human_label") or "").strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("khong co dong nao co human_label")

    docs = {}
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs[d["key"]] = d

    call = make_caller(args)
    import time

    result = {"n_pairs": len(rows), "model": args.model, "variants": {}}
    base = [(norm_label(r["llm_label"]), norm_label(r["human_label"]))
            for r in rows]
    ag, kp = kappa(base)
    print(f"tap do: {len(rows)} cap da co nhan nguoi   model={args.model}")
    print(f"\n{'bien the':<14}{'dong thuan':>12}{'kappa':>9}"
          f"{'dup_recall':>12}{'dup_prec':>10}{'#DUP':>7}")
    print(f"{'(da chay v0)':<14}{ag:>11.1%}{kp:>9.3f}", end="")
    hd = sum(1 for _, h in base if h == "DUPLICATE")
    tp = sum(1 for a, h in base if a == h == "DUPLICATE")
    ld = sum(1 for a, _ in base if a == "DUPLICATE")
    print(f"{tp/max(1,hd):>12.0%}{tp/max(1,ld):>10.0%}{ld:>7}")

    for v in args.variants:
        if v == "v0_baseline":
            continue
        # Ghi tung cap ra dia NGAY. Lan chay dau khong lam vay: 106 luot goi
        # mat 84 phut vi dinh han muc lien tuc, va vi ket qua chi duoc ghi o
        # cuoi nen dung giua chung la mat trang. Co file nay thi chay lai chi
        # goi phan con thieu.
        cache_path = f"{os.path.splitext(args.out)[0]}.{v}.cache.jsonl"
        cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                line = line.strip()
                if line:
                    x = json.loads(line)
                    if "API_ERROR" not in (x["ab"], x["ba"]):
                        cache[x["pair_id"]] = x
            print(f"  {v}: dung lai {len(cache)} cap tu {cache_path}")
        cfh = open(cache_path, "a", encoding="utf-8")
        got, raw = [], []
        for i, r in enumerate(rows, 1):
            da, db = docs.get(r["a"]), docs.get(r["b"])
            if not (da and db):
                continue
            hit = cache.get(r["pair_id"])
            if hit:
                raw.append(hit)
                lab = hit["ab"] if hit["ab"] == hit["ba"] else "DISAGREE"
                got.append((lab, norm_label(r["human_label"])))
                continue
            labs = []
            for x, y in ((da, db), (db, da)):
                p = build_prompt(x, y, r["tier"], v)
                rep = ""
                for attempt in range(args.retries):
                    try:
                        rep = call(p)
                        break
                    except Exception as e:                    # noqa: BLE001
                        msg = f"{type(e).__name__}: {e}"
                        # Phan biet hai loai 429. Free tier co ca hai:
                        #   ...RequestsPerMinute...  -> cho la qua
                        #   ...RequestsPerDay...     -> CHO BAO NHIEU CUNG VO
                        #                               ICH, het la het ngay hom do
                        # Doc nham loai thu hai thanh loai thu nhat da lam mot
                        # lan chay treo 84 phut roi khong ra ket qua nao.
                        if "PerDay" in msg:
                            sys.exit(
                                "\nHET HAN MUC NGAY cho model "
                                f"{args.model} (free tier: 500 luot/ngay/model)."
                                "\nCho sang ngay mai, doi --model khac (moi model"
                                " co ngan sach rieng), hoac bat billing.")
                        time.sleep(args.quota_wait
                                   if ("RESOURCE_EXHAUSTED" in msg
                                       or "429" in msg)
                                   else min(2 ** attempt, 30))
                labs.append(parse_reply(rep)["label"] if rep else "API_ERROR")
                if args.sleep:
                    time.sleep(args.sleep)
            # Chi lay cap NHAT QUAN hai chieu, dung nhu duong chay that.
            lab = labs[0] if labs[0] == labs[1] else "DISAGREE"
            rec = {"pair_id": r["pair_id"], "ab": labs[0], "ba": labs[1],
                   "human": norm_label(r["human_label"])}
            raw.append(rec)
            cfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cfh.flush()
            got.append((lab, norm_label(r["human_label"])))
            if i % 5 == 0:
                print(f"    ... {i}/{len(rows)}", flush=True)
        cfh.close()

        usable = [(a, h) for a, h in got if a in LABELS]
        ag, kp = kappa(usable)
        hd = sum(1 for _, h in usable if h == "DUPLICATE")
        tp = sum(1 for a, h in usable if a == h == "DUPLICATE")
        ld = sum(1 for a, _ in usable if a == "DUPLICATE")
        n_dis = sum(1 for a, _ in got if a == "DISAGREE")
        print(f"{v:<14}{ag:>11.1%}{kp:>9.3f}"
              f"{tp/max(1,hd):>12.0%}{tp/max(1,ld):>10.0%}{ld:>7}"
              f"   (bo {n_dis} cap bat dong hai chieu)")
        result["variants"][v] = {
            "n_usable": len(usable), "n_disagree": n_dis,
            "agreement": round(ag, 4), "kappa": round(kp, 4),
            "dup_recall": round(tp / max(1, hd), 4),
            "dup_precision": round(tp / max(1, ld), 4),
            "n_llm_duplicate": ld, "raw": raw,
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")
    print("\nNHAC: 53 cap nay vua tinh chinh vua do -> con kappa tren la CAN"
          " TREN.\nBien the thang phai xac nhan lai tren mau audit MOI.")


if __name__ == "__main__":
    main()
