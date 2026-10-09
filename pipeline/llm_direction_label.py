# -*- coding: utf-8 -*-
"""Gan NHAN CHIEU cho tung issue bang LLM, chay mot lan offline.

Khac han viec LLM tien-gan-nhan da bi bac bo (kappa 0,14-0,19): o day LLM doc
MOT issue va tra loi mot cau hoi don ("mo ta thieu mot hanh vi hay thua mot
hanh vi?"), khong phai phan xet quan he giua hai issue. Va nhan nay KHONG duoc
dung lam chuan cham diem.

Ban dau nhan chi la tin hieu cho ``direction_filter.demote`` (bo loc hau ky, do
ra delta bang 0 nen khong bat). Nay nhan doi vai: lam DU LIEU TRAIN cho bo ba
nguoc chieu (neo / cung chieu / nguoc chieu) de fine-tune e5, vi do do
2026-09-22 cho thay e5 goc xep "gui thua sms" gan "gui thieu sms" hon (cos
0,921) ca cap dong nghia dung (0,791). Vi the DO CHINH XAC cua nhan bay gio
quan trong hon do PHU: train bang nhan sai la day model sai.

Quy trinh:
    label   goi LLM cho tung issue, cache JSONL, chay lai thi bo qua phan xong
    check   doi chieu voi luat tu vung tren phan ca hai cung gan nhan, va xuat
            mau ngau nhien de ky su cham tay
    sample  rut mau CO DINH (theo --seed) cua mot truc de cham tay do chinh xac
    score   so nhan hien tai voi bo da cham, tach ro "da sua" / "con sai" /
            "MAT nhan dung" de biet lan siet prompt duoc hay hong

Sau moi lan sua mo ta truc trong AXIS_DESC/RULES, chay lai RIENG truc do:
    python pipeline/llm_direction_label.py label --relabel-axis send_count,permission
roi cham lai bang ``score``. Nhan cu luon duoc chup lai ra *.pre-* truoc do.

Khoa API doc tu bien moi truong (mac dinh GEMINI_API_KEY); script khong bao gio
ghi khoa ra file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_direction_pairs import AXES, classify, judging_text

SEED = 20260922
MODEL = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"

AXIS_DESC = {
    "send_count": ("send_count", "SO BAN TIN da gui, khong xet noi dung ban tin. "
                                 "missing = KHONG GUI ban tin/thong bao dang le "
                                 "phai gui den mot doi tuong; extra = GUI THEM "
                                 "ban tin khong dang co (gui lap, gui double, "
                                 "hoac gui cho doi tuong khong dang nhan)"),
    "permission": ("permission", "QUYEN theo role/phan quyen, khong xet trang "
                                 "thai doi tuong hay dau vao. "
                                 "blocked = chu the CO quyen theo role/phan quyen "
                                 "nhung bi chan hoac bi bao khong co quyen; "
                                 "not_enforced = chu the KHONG co quyen theo "
                                 "role/phan quyen van truy cap/thuc hien duoc"),
    "button_state": ("button_state", "enable_wrong = nut/truong dang le phai khoa "
                                     "nhung van bam/nhap duoc; disable_wrong = dang le "
                                     "duoc dung nhung bi khoa"),
    "limit_enforce": ("limit_enforce", "BAT BUOC co ca hai: (1) bang chung ve mot "
                                       "HAN MUC/NGUONG cu the (gia tri toi thieu, "
                                       "toi da, so lan, tich luy ngay/thang) va (2) "
                                       "hanh vi CHAN sai. not_blocked = giao dich VI "
                                       "PHAM han muc ma he thong VAN CHO THUC HIEN; "
                                       "over_blocked = giao dich KHONG vi pham han muc "
                                       "(hoac issue khang dinh ro la hop le) nhung bi "
                                       "chan hoac bi bao vuot han muc"),
    "otp_required": ("otp_required", "missing_otp = dang le phai hoi OTP nhung khong hoi; "
                                     "extra_otp = khong can OTP nhung van hoi"),
    "result_status": ("result_status", "BAT BUOC co ca hai: (1) bang chung ve KET QUA "
                                       "THAT cua thao tac va (2) THONG BAO/KET QUA he "
                                       "thong tra ra, va hai cai do TRAI NGUOC nhau. "
                                       "false_success = ket qua that la THAT BAI/chua "
                                       "hoan tat nhung he thong bao hoac xu ly nhu "
                                       "THANH CONG; false_failure = ket qua that la "
                                       "THANH CONG (hoac dau vao/trang thai duoc khang "
                                       "dinh ro la hop le) nhung he thong bao LOI"),
}

RULES = """Quy tac:
- Can cu vao HIEN TRANG (tieu de + actual), khong can cu vao ky vong (expected).
- Neu TIEU DE va ACTUAL noi nguoc nhau, chon theo cau nao neu BANG CHUNG cu the
  (so lieu, log, trang thai DB, ma loi, mo ta thao tac da xay ra). Vi du tieu de
  "Loi khong gui sms khi change pin thanh cong" nhung actual "Change PIN thanh
  cong ma gui SMS ve" -> actual mo ta viec da xay ra, chon theo actual. Neu ca
  hai deu chung chung, hoac actual chi chep lai HANH VI MONG MUON (vd actual
  viet "he thong thuc hien gui SMS" trong khi tieu de bao "khong gui SMS")
  -> tra "none".
- Issue khong thuoc cap nao, hoac khong ro chieu, hoac noi ca hai chieu -> "none".
- Khong doan. Khong ro thi "none".
- RIENG result_status, PHAI chi ra duoc ket qua THAT va thong bao he thong TRAI
  NGUOC nhau. Thieu mot trong hai -> "none". Cu the tra "none" khi:
  * Issue chi noi "hien thong bao loi / System busy / loi 500" TRAN TRUI, khong
    kem theo bat ky dau hieu nao cho thay thao tac dang le phai thanh cong.
    NGUOC LAI, neu issue noi thao tac DA chay dung roi moi bao loi ("upload file
    THANH CONG nhung bao system busy", "nhap thong tin HOP LE van bao loi",
    "giao dich CO THAT nhung bao not found") thi do CHINH LA mau thuan can tim
    -> false_failure, KHONG phai none.
  * Ket qua that va thong bao KHOP nhau, chi SAI MA LOI hoac sai noi dung thong
    bao (vd: bo trong truong bat buoc thi bao loi la dung, chi sai ma loi;
    "tra sai ma loi", "hien sai message", "sai error_code" -> none).
  * Thao tac that bai that va he thong bao that bai.
  * Van de DUY NHAT la mot truong trang thai trong DB bi luu sai hoac khong
    duoc cap nhat (status, queue_status, request_status, inquiry_times...),
    con ket qua bao cho nguoi dung thi KHOP voi thuc te. "Cap nhat sai trang
    thai", "khong update status" tu no KHONG phai mot chieu.
    NGUOC LAI, neu he thong bao THANH CONG ma mot phan viec thuc su KHONG xay
    ra (khong tru tien/phi, khong tai ve file, khong luu ban ghi, nguoi nhan
    khong nhan duoc tien) thi do la false_success, KHONG phai none.
  * Issue chi noi he thong CHUA KIEM TRA mot trang thai hay mot dieu kien, va
    KHONG noi he thong da tra ket qua gi (vd "chua check trang thai partnerID").
    NGUOC LAI, neu issue noi ro he thong TRA VE THANH CONG cho mot thao tac dang
    le phai that bai (dau vao khong ton tai, du lieu khong hop le) thi do la
    false_success.
  Chon ben theo KET QUA THAT, khong theo chu trong tieu de: "tao merchant THANH
  CONG nhung errorType bao Fail" la false_failure, khong phai false_success.
- RIENG limit_enforce, PHAI chi ra duoc mot HAN MUC/NGUONG cu the va hanh vi
  chan sai. Thieu mot trong hai -> "none". Cu the tra "none" khi:
  * Giao dich VI PHAM han muc that va he thong CO chan, chi sai thong bao hoac
    sai ma loi (vd "vi pham han muc thang bao loi system busy", "hien thi thong
    bao sai khi vi pham han muc", "amount trong message sai"). Chan la DUNG,
    chi thong bao sai -> none.
  * Loi nam o viec khong kiem tra TRANG THAI doi tuong (clientID/partnerID
    dang REGISTERED, LOCKED, Blocked, khac Active) - do la loi check trang
    thai, khong lien quan han muc.
  * Loi hien thi: trang man hinh, khong hien thong bao, sai giao dien.
  * Khong co nguong nao duoc noi den, chi la validate dinh dang/do dai dau vao.
- RIENG send_count, truc nay CHI DEM so ban tin da gui. Tra "none" trong cac
  truong hop sau:
  * Ban tin DA duoc gui nhung noi dung sai hoac thieu truong: thieu balance,
    thieu param, thieu fee, thieu thong tin trong SMS, sai so lieu, sai ngon
    ngu, sai ma SMS, noi dung null/rong, gui sai noi dung so voi tai lieu.
    Day la loi NOI DUNG ban tin, khong phai loi so luong ban tin.
  * Khong lien quan den ban tin/thong bao gui cho nguoi dung (vd insert trung
    ban ghi trong DB, tru tien hai lan, tao trung giao dich).
  * Issue noi ca hai chieu: vua gui thieu cho doi tuong nay vua gui thua cho
    doi tuong kia.
- RIENG permission, truc nay CHI noi ve quyen theo role/phan quyen. Tra "none"
  trong cac truong hop sau:
  * Loi nam o viec kiem tra TRANG THAI doi tuong: clientID/tai khoan dang
    Block, Lock, Dormant, Cancel, Registered, khac Active; merchant khac
    Active; user inactive; tai khoan bi khoa do nhap sai PIN/OTP qua so lan.
    Du issue viet "van thuc hien duoc" thi day van la loi check trang thai.
  * Loi nam o viec kiem tra DAU VAO hoac DIEU KIEN nghiep vu: do dai truong,
    dinh dang, so dien thoai khac nha mang, sai PIN, sai OTP, han muc,
    "chua validate", "chua check" mot dieu kien khong phai phan quyen.
  Khi chon ben, chi can cu vao CHU THE co quyen hay khong, khong suy tu chu
  "phan quyen" hay "khong co quyen" trong thong bao loi: role dang le DUOC
  phep ma bi chan -> blocked; role dang le KHONG duoc phep ma van lam duoc
  -> not_enforced."""


PROMPT = """Ban doc mot issue Jira cua he thong vi dien tu va phan loai CHIEU cua loi.

Chi chon mot trong cac cap duoi day, va chi khi issue noi RO rang:
{axes}

{rules}

Issue:
---
{text}
---

Tra ve DUNG mot dong JSON, khong giai thich:
{{"axis": "<ten truc hoac none>", "side": "<ten ben hoac none>", "confident": true/false}}"""


def axes_block() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in AXIS_DESC.values())


def make_prompt(doc: dict) -> str:
    return PROMPT.format(axes=axes_block(), rules=RULES,
                         text=judging_text(doc)[:1500])


BATCH_PROMPT = """Ban doc {n} issue Jira cua he thong vi dien tu va phan loai CHIEU
cua loi cho TUNG issue.

Chi chon mot trong cac cap duoi day, va chi khi issue noi RO rang:
{axes}

{rules}

{items}

Tra ve DUNG {n} dong JSON, moi dong cho mot issue theo dung thu tu tren,
khong giai thich, khong danh so lai:
{{"i": 1, "axis": "<truc hoac none>", "side": "<ben hoac none>", "confident": true/false}}"""


def make_batch_prompt(docs: list[dict]) -> str:
    """Gop nhieu issue vao mot luot goi: han muc tang mien phi tinh theo SO
    REQUEST nen goi tung issue mot se mat hang gio cho 10.925 issue."""
    items = "\n\n".join(f"### Issue {i}\n{judging_text(d)[:1200]}"
                        for i, d in enumerate(docs, 1))
    return BATCH_PROMPT.format(n=len(docs), axes=axes_block(), rules=RULES,
                               items=items)


def parse_batch(raw: str, n: int) -> list[tuple[str, str, bool]] | None:
    """Tra ve dung n ket qua, hoac None neu khong khop -> goi lai tung cai."""
    out: dict[int, tuple[str, str, bool]] = {}
    for line in (raw or "").splitlines():
        line = line.strip().strip("`")
        start, end = line.find("{"), line.rfind("}")
        if start < 0 or end < 0:
            continue
        try:
            obj = json.loads(line[start:end + 1])
        except json.JSONDecodeError:
            continue
        try:
            idx = int(obj.get("i", 0))
        except (TypeError, ValueError):
            continue
        axis = str(obj.get("axis", "none")).strip().lower()
        side = str(obj.get("side", "none")).strip().lower()
        if axis not in AXES or side not in AXES.get(axis, {}):
            axis, side = "none", ""
        if 1 <= idx <= n:
            out[idx] = (axis, side, bool(obj.get("confident")))
    if len(out) != n:
        return None
    return [out[i] for i in range(1, n + 1)]


def parse(raw: str) -> tuple[str, str, bool]:
    text = (raw or "").strip().strip("`")
    if text.startswith("json"):
        text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return ("PARSE_ERROR", "", False)
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return ("PARSE_ERROR", "", False)
    axis = str(obj.get("axis", "none")).strip().lower()
    side = str(obj.get("side", "none")).strip().lower()
    if axis not in AXES or side not in AXES.get(axis, {}):
        return ("none", "", bool(obj.get("confident")))
    return (axis, side, bool(obj.get("confident")))


def read_cache(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["key"]: r for r in (json.loads(l) for l in f if l.strip())}


TRANSIENT = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "500", "INTERNAL",
             "deadline", "timeout", "Timeout")


def cmd_label(args) -> None:
    import itertools
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from google import genai
    from google.genai import types

    # Nhieu khoa ngan cach bang dau phay -> xoay vong de chia han muc.
    keys = [k.strip() for k in os.environ.get(args.key_env, "").split(",") if k.strip()]
    if not keys:
        sys.exit(f"thieu bien moi truong {args.key_env}")
    with open(args.corpus, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    cache = read_cache(args.out)
    if args.retry_errors:
        cache = {k: v for k, v in cache.items()
                 if not str(v.get("axis", "")).endswith("_ERROR")}
    if args.relabel_axis:
        # Cache la JSONL ghi noi duoi va read_cache lay dong CUOI cho moi key,
        # nen chi can bo key khoi cache la no duoc goi lai va ghi de.
        drop = {a.strip() for a in args.relabel_axis.split(",") if a.strip()}
        cache = {k: v for k, v in cache.items() if v.get("axis") not in drop}
    todo = [d for d in docs if str(d.get("key", "")).upper() not in cache]
    if args.only_keys:
        # Han muc mien phi rat thap (20 luot/ngay moi cap khoa x model), nen
        # sau khi sua prompt hay goi lai DUNG bo da cham tay truoc: 5 luot goi
        # la biet prompt moi tot hon hay xau di, thay vi dot ca ngay han muc.
        want = {str(r["key"]).upper() for r in read_check(args.only_keys)}
        todo = [d for d in todo if str(d.get("key", "")).upper() in want]
    if args.limit:
        todo = todo[:args.limit]
    print(f"da co {len(cache)} | can goi {len(todo)}", flush=True)
    if not todo:
        return

    # Han muc tang mien phi tinh RIENG cho tung (khoa, model) va rat thap
    # (gemini-3.6-flash: 20 luot/ngay). Vi vay xoay vong tren tich khoa x model
    # va loai bo cap nao bao het han muc ngay, thay vi dung han.
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    slots = [(genai.Client(api_key=k, http_options=types.HttpOptions(timeout=60000)), m)
             for k in keys for m in models]
    print(f"{len(keys)} khoa x {len(models)} model = {len(slots)} cap", flush=True)
    dead: set[int] = set()
    turn = itertools.cycle(range(len(slots)))

    def next_slot() -> int | None:
        for _ in range(len(slots)):
            i = next(turn)
            if i not in dead:
                return i
        return None
    cfg = types.GenerateContentConfig(temperature=0, max_output_tokens=args.max_tokens)
    stats = Counter()
    lock = threading.Lock()
    done_count = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "a", encoding="utf-8")

    def work(batch: list[dict]) -> list[dict]:
        one = len(batch) == 1
        prompt = make_prompt(batch[0]) if one else make_batch_prompt(batch)
        results: list[tuple[str, str, bool]] | None = None
        err = ""
        used = args.model
        for attempt in range(args.retries + 1):
            with lock:
                slot = next_slot()
            if slot is None:
                err = "HET HAN MUC tat ca cap khoa x model"
                break
            client, used = slots[slot]
            try:
                resp = client.models.generate_content(
                    model=used, contents=prompt, config=cfg)
                text = resp.text or ""
                results = ([parse(text)] if one else parse_batch(text, len(batch)))
                if results is None:
                    err = "PARSE_ERROR"
                break
            except Exception as exc:                           # noqa: BLE001
                err = str(exc)[:300]
                if "PerDay" in err or "RequestsPerDay" in err:
                    with lock:
                        dead.add(slot)                 # cap nay het cho hom nay
                    continue
                if any(t in err for t in TRANSIENT) and attempt < args.retries:
                    time.sleep(args.backoff * (attempt + 1))
                    continue
                break
        if results is None:
            kind = "PARSE_ERROR" if err == "PARSE_ERROR" else "API_ERROR"
            results = [(kind, err, False)] * len(batch)
        return [{"key": str(d["key"]).upper(), "axis": a, "side": s,
                 "confident": c, "model": used}
                for d, (a, s, c) in zip(batch, results)]

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"{len(batches)} luot goi, moi luot {args.batch} issue", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rows in pool.map(work, batches):
                with lock:
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats[row["axis"]] += 1
                        done_count += 1
                    out.flush()
                    if done_count % 200 < args.batch or done_count == len(todo):
                        print(f"  {done_count}/{len(todo)}  {dict(stats)}", flush=True)
    finally:
        out.close()
    errors = stats["API_ERROR"]
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    if errors:
        print(f"{errors} dong loi - chay lai chinh lenh nay de goi lai rieng chung "
              "(cache bo qua dong da xong, nhung dong API_ERROR se duoc goi lai "
              "sau khi loc bang --retry-errors).")


def cmd_check(args) -> None:
    with open(args.corpus, encoding="utf-8") as f:
        docs = {str(json.loads(line)["key"]).upper(): json.loads(line)
                for line in f if line.strip()}
    llm = {k: v for k, v in read_cache(args.out).items()
           if v["axis"] not in {"API_ERROR", "PARSE_ERROR"}}
    rule = {k: set(classify(judging_text(d), d.get("summary", "")))
            for k, d in docs.items()}
    rule = {k: v for k, v in rule.items() if v}

    both = [k for k in llm if k in rule]
    agree_axis = sum(1 for k in both
                     if llm[k]["axis"] in {a for a, _ in rule[k]})
    agree_full = sum(1 for k in both if (llm[k]["axis"], llm[k]["side"]) in rule[k])
    labeled = [k for k, v in llm.items() if v["axis"] != "none"]
    print(f"LLM gan nhan   : {len(labeled)}/{len(llm)} = {len(labeled) / max(1, len(llm)):.1%}")
    print(f"luat tu vung   : {len(rule)}/{len(docs)} = {len(rule) / max(1, len(docs)):.1%}")
    print(f"ca hai cung gan: {len(both)}")
    if both:
        print(f"  trung truc   : {agree_axis}/{len(both)} = {agree_axis / len(both):.1%}")
        print(f"  trung ca ben : {agree_full}/{len(both)} = {agree_full / len(both):.1%}")
    print(Counter(v["axis"] for v in llm.values() if v["axis"] != "none"))

    # Mau de nguoi cham: chi lay issue LLM gan nhan ma LUAT KHONG gan, vi day
    # moi la phan them vao - phan trung nhau khong noi len dieu gi moi.
    new = [k for k in labeled if k not in rule]
    rng = random.Random(SEED)
    rng.shuffle(new)
    rows = [{"key": k, "axis": llm[k]["axis"], "side": llm[k]["side"],
             "confident": llm[k]["confident"], "ok": "",
             "text": " ".join(judging_text(docs[k]).split())[:400]}
            for k in new[:args.sample]]
    if rows:
        with open(args.review_out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"-> {args.review_out} ({len(rows)} issue LLM gan them, de cham tay)")


def load_corpus(path: str) -> dict[str, dict]:
    docs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs[str(d["key"]).upper()] = d
    return docs


def read_check(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cmd_sample(args) -> None:
    """Rut mau de nguoi cham TAY do chinh xac nhan cua MOT truc.

    Khac ``check``: o day lay ca issue ma luat tu vung cung gan, vi cau hoi
    khong phai "LLM them duoc gi" ma "trong so nhan truc nay, bao nhieu phan
    tram dung". Mau duoc chot bang --seed de lan chay sau van la dung mau do.
    """
    docs = load_corpus(args.corpus)
    llm = read_cache(args.out)
    exclude = set()
    for path in (args.exclude or "").split(","):
        path = path.strip()
        if path and os.path.exists(path):
            exclude |= {str(r["key"]).upper() for r in read_check(path)}
    rng = random.Random(args.seed)
    rows = []
    for axis in (a.strip() for a in args.axis.split(",") if a.strip()):
        keys = sorted(k for k, v in llm.items()
                      if v["axis"] == axis and k in docs and k not in exclude)
        rng.shuffle(keys)
        rows += [{"key": k, "axis": axis, "side": llm[k]["side"], "ok": "",
                  "note": "",
                  "text": " ".join(judging_text(docs[k]).split())[:400]}
                 for k in keys[:args.sample]]
    if not rows:
        sys.exit("khong co issue nao mang truc nay")
    with open(args.review_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"-> {args.review_out} ({len(rows)} issue, cot ok de trong: "
          "1 = nhan dung, 0 = nhan sai)")


def cmd_score(args) -> None:
    """So nhan HIEN TAI voi mot bo da cham tay.

    Bo cham tay ghi nhan CU (truoc khi siet). Vi vay moi dong duoc doc nhu sau:
        ok=1, nhan giu nguyen -> GIU DUNG   (tu so cua do chinh xac moi)
        ok=1, nhan doi        -> MAT        (siet qua tay, xoa mat nhan dung)
        ok=0, nhan thanh none -> DA SUA
        ok=0, nhan giu nguyen -> CON SAI
        ok=0, nhan sang truc khac -> CAN XEM LAI (may khong tu cham duoc)
    """
    llm = read_cache(args.out)
    rows = read_check(args.check)
    buckets = Counter()
    per_axis: dict[str, Counter] = {}
    moved: list[str] = []
    for r in rows:
        key = str(r["key"]).upper()
        ok = str(r.get("ok", "")).strip()
        if ok not in {"0", "1"}:
            buckets["chua_cham"] += 1
            continue
        axis = r["axis"].strip()
        cur = llm.get(key)
        same = bool(cur) and cur["axis"] == axis and cur["side"] == r["side"].strip()
        new_axis = cur["axis"] if cur else "(thieu)"
        if ok == "1":
            tag = "giu_dung" if same else "mat"
        elif same:
            tag = "con_sai"
        elif new_axis in {"none", "API_ERROR", "PARSE_ERROR", "(thieu)"}:
            tag = "da_sua"
        else:
            tag = "can_xem_lai"
            moved.append(f"{key} {axis}/{r['side']} -> {new_axis}/{cur['side']}")
        buckets[tag] += 1
        per_axis.setdefault(axis, Counter())[tag] += 1

    def report(name: str, c: Counter) -> None:
        old_ok = c["giu_dung"] + c["mat"]
        old_n = old_ok + c["con_sai"] + c["da_sua"]
        keep = c["giu_dung"] + c["con_sai"] + c["can_xem_lai"]
        print()
        print(f"{name}: {old_n} issue da cham")
        print(f"  truoc khi siet : {old_ok}/{old_n} dung = "
              f"{old_ok / max(1, old_n):.0%}")
        if keep:
            print(f"  sau khi siet   : {c['giu_dung']}/{keep} dung = "
                  f"{c['giu_dung'] / keep:.0%}  (con giu {keep}/{old_n} nhan)")
        else:
            print("  sau khi siet   : khong con nhan nao o truc nay")
        print(f"  da sua {c['da_sua']} | con sai {c['con_sai']} | "
              f"MAT {c['mat']} nhan dung | can xem lai {c['can_xem_lai']}")

    for axis in sorted(per_axis):
        report(axis, per_axis[axis])
    if len(per_axis) > 1:
        report("GOP", buckets)
    if buckets["chua_cham"]:
        print()
        print(f"{buckets['chua_cham']} dong chua cham (cot ok trong) - bo qua")
    if moved:
        print()
        print("chuyen sang truc khac, phai xem tay:")
        for line in moved[:20]:
            print("  " + line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "check", "sample", "score"])
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--out", default="data/processed/direction_labels_llm.jsonl")
    ap.add_argument("--review-out", default="data/processed/direction_labels_review.csv")
    ap.add_argument("--model", default=MODEL,
                    help="mot hoac nhieu model, ngan cach bang dau phay")
    ap.add_argument("--key-env", default="GEMINI_API_KEY")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--axis", default="send_count,permission",
                    help="sample: rut mau cac truc nay")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude", default="",
                    help="sample: cac file CSV co san, khong rut lai "
                         "nhung key da nam trong do")
    ap.add_argument("--check", default="data/processed/direction_axis_check_dev.csv",
                    help="score: file da cham tay (cot ok = 0/1)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10,
                    help="so issue gop vao mot luot goi")
    ap.add_argument("--relabel-axis", default="",
                    help="goi lai cac issue dang mang truc nay (vd result_status) "
                         "sau khi sua prompt; cac truc khac giu nguyen cache")
    ap.add_argument("--only-keys", default="",
                    help="chi goi lai nhung key nam trong file CSV nay (cot key); "
                         "dung chung voi --relabel-axis de thu prompt moi tren bo "
                         "da cham truoc khi chay ca truc")
    ap.add_argument("--retry-errors", action="store_true",
                    help="goi lai nhung dong API_ERROR/PARSE_ERROR da cache")
    args = ap.parse_args()
    {"label": cmd_label, "check": cmd_check, "sample": cmd_sample,
     "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
