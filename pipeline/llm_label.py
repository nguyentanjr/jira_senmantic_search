# -*- coding: utf-8 -*-
"""
Goi LLM gan nhan, hop nhat hai chieu, va lay mau cho nguoi audit.

    python pipeline/llm_label.py run --dry-run        # uoc luong chi phi
    python pipeline/llm_label.py run --tier A         # goi that, chay tang A truoc

Khong khoa vao mot nha cung cap -- xem `make_caller`:

    --provider openai      moi endpoint OpenAI-compatible (mac dinh)
    --provider gemini      google-genai
    --provider anthropic   anthropic

    # model noi bo, du lieu khong ra khoi mang don vi
    --base-url http://localhost:11434/v1 --model qwen2.5:14b-instruct \
        --api-key-env NONE
    python pipeline/llm_label.py ingest               # hop nhat + loc
    python pipeline/llm_label.py audit --n 100        # lay mau cho nguoi soat
    python pipeline/llm_label.py kappa                # sau khi nguoi dien nhan

Chay `build_llm_label_pool.py` truoc.

LOC THEO TINH NHAT QUAN, KHONG THEO CONFIDENCE
Moi cap duoc hoi hai lan, dao thu tu (A,B) va (B,A). Quan he doi xung nen hai
lan phai ra cung nhan. Bat dong khi dao la phep do hieu chuan truc tiep va
khong ton them nhan nao.

Confidence LLM tu khai KHONG duoc dung lam bo loc chinh: no hieu chuan kem, va
no tu tin nhat dung luc bi danh lua boi hai van ban giong nhau -- tuc dung ca
ta can no than trong. Confidence chi duoc ghi lai de doi chieu.

BA MUC TIN CAY DAU RA
    agree_high   hai chieu cung nhan, ca hai confidence >= --conf-floor
    agree_low    hai chieu cung nhan, confidence thap
    disagree     hai chieu khac nhau -> KHONG dung lam nhan, day sang nguoi

Chi `agree_high` duoc dung lam silver label. `disagree` va `agree_low` vao hang
doi audit -- do la nhung cap kho, va cung la nhung cap dang cho nguoi xem nhat.

NHAN TANG A VA TANG B KHONG DUOC TRON
Tang A tra loi "cung underlying defect"; tang B tra loi "cung bieu hien quan sat
duoc". Hai cau hoi khac nhau -> hai quan he khac nhau. Trong bao cao goi tang B
la "behavioral duplicate". Xem build_llm_label_pool.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

LABELS = ("DUPLICATE", "RELATED", "UNRELATED", "UNCERTAIN")
JSON_RE = re.compile(r"\{.*\}", re.S)


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dump_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_reply(text: str) -> dict:
    """LLM duoc yeu cau tra JSON thuan, nhung van phai chiu duoc rac quanh no."""
    m = JSON_RE.search(text or "")
    if not m:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "reason": text[:200]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"label": "PARSE_ERROR", "confidence": 0.0, "reason": text[:200]}
    lab = str(d.get("label", "")).strip().upper()
    return {
        "label": lab if lab in LABELS else "PARSE_ERROR",
        "confidence": float(d.get("confidence", 0.0) or 0.0),
        "reason": str(d.get("reason", ""))[:300],
    }


# -------------------------------------------------------------- provider
#
# Khong khoa vao mot nha cung cap. `openai` la duong OpenAI-compatible, va no
# phu gan het: Ollama / vLLM / LM Studio chay noi bo, cung nhu DeepSeek, Qwen,
# Together, Groq... -- chi doi --base-url.
#
# LUU Y VE DU LIEU: day la bug report noi bo. Van ban da qua `scrub()` (thay
# secret / PII / so tai khoan bang the), nhung van con ten service, ma loi,
# luong nghiep vu. Neu don vi khong cho phep gui ra ngoai thi chay model noi bo:
#     ollama serve
#     python pipeline/llm_label.py run --base-url http://localhost:11434/v1 \
#         --model qwen2.5:14b-instruct --api-key-env NONE
DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.0-flash",
}
DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def make_caller(args):
    """-> ham nhan prompt tra ve text. Nem ngoai le de vong retry bat."""
    prov = args.provider
    key_env = args.api_key_env or DEFAULT_KEY_ENV[prov]
    key = "local" if key_env.upper() == "NONE" else os.environ.get(key_env, "")
    if not key:
        sys.exit(f"chua dat bien moi truong {key_env} "
                 f"(dat --api-key-env NONE neu endpoint noi bo khong can key)")

    if prov == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("chua cai SDK:  pip install openai")
        client = OpenAI(api_key=key, base_url=args.base_url or None)

        def call(prompt: str) -> str:
            r = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=args.max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        return call

    if prov == "anthropic":
        try:
            import anthropic
        except ImportError:
            sys.exit("chua cai SDK:  pip install anthropic")
        client = anthropic.Anthropic(api_key=key)

        def call(prompt: str) -> str:
            m = client.messages.create(
                model=args.model, max_tokens=args.max_tokens, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in m.content
                           if getattr(b, "type", "") == "text")
        return call

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("chua cai SDK:  pip install google-genai")
    # Khong co timeout thi mot ket noi treo dung ca lan chay (da gap khi fill
    # pooling: 8 phut khong tien them). Tham so tuy chon -> script cu khong doi.
    timeout_s = getattr(args, "timeout", 0) or 0
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)) if timeout_s else None)

    def call(prompt: str) -> str:
        r = client.models.generate_content(
            model=args.model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=args.max_tokens))
        return r.text or ""
    return call


# ------------------------------------------------------------------- run
def cmd_run(args) -> None:
    reqs = load_jsonl(args.requests)
    if not reqs:
        sys.exit(f"khong co {args.requests} -- chay build_llm_label_pool.py truoc")

    # --resume bo qua luot DA THANH CONG. Dong loi (API_ERROR / PARSE_ERROR)
    # phai duoc goi lai -- neu tinh ca chung la "da xong" thi lan chay lai se
    # bo qua dung nhung luot can goi. Dong loi cu duoc loc bo khi ingest, va
    # dong moi ghi de o buoc do vi ingest lay theo request_id.
    done = {r["request_id"] for r in load_jsonl(args.out)
            if not str(r.get("label", "")).endswith("_ERROR")}         if args.resume else set()
    todo = [r for r in reqs if r["request_id"] not in done]
    if args.tier:
        todo = [r for r in todo if r["tier"] == args.tier]
    if args.limit:
        todo = todo[: args.limit]

    args.model = args.model or DEFAULT_MODEL[args.provider]
    chars = sum(len(r["prompt"]) for r in todo)
    print(f"can goi   : {len(todo)} luot  (da co {len(done)})")
    print(f"input     : ~{chars // 4:,} token (uoc tinh 4 ky tu/token)")
    print(f"output    : ~{len(todo) * 60:,} token")
    print(f"provider  : {args.provider}"
          + (f"  base_url={args.base_url}" if args.base_url else ""))
    print(f"model     : {args.model}")
    if args.dry_run:
        print("\n--dry-run: khong goi API")
        return

    call = make_caller(args)
    written = 0
    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    with open(args.out, mode, encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            reply, err = "", ""
            for attempt in range(args.retries):
                try:
                    reply = call(r["prompt"])
                    err = ""
                    break
                except Exception as e:                       # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:200]
                    # 429/RESOURCE_EXHAUSTED la han muc theo phut, khong phai
                    # loi tam thoi -- lui 2^n giay (toi da 30) khong du. Do
                    # tren lan chay tang A dau tien: 38/240 luot mat trang vi
                    # ly do nay. Gap han muc thi cho lau han han.
                    exhausted = "RESOURCE_EXHAUSTED" in err or "429" in err
                    time.sleep(args.quota_wait if exhausted
                               else min(2 ** attempt, 30))
            out = {k: r[k] for k in
                   ("request_id", "pair_id", "order", "tier", "a", "b")}
            # Ghi lai model sinh ra dong nay: neu phai chay lai bang model khac
            # (vd het han muc) thi tap nhan bi tron, va tron ma khong ghi lai
            # thi khong tai lap duoc.
            out["model"] = args.model
            out.update(parse_reply(reply) if reply else
                       {"label": "API_ERROR", "confidence": 0.0, "reason": err})
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if args.sleep:
                time.sleep(args.sleep)
            if i % 25 == 0:
                print(f"  ... {i}/{len(todo)}", flush=True)
    print(f"\nghi {written} phan hoi -> {args.out}")


# ---------------------------------------------------------------- ingest
def cmd_ingest(args) -> None:
    resp = load_jsonl(args.responses)
    if not resp:
        sys.exit(f"khong co {args.responses}")

    by_pair: dict[str, dict] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for r in resp:
        by_pair[r["pair_id"]][r["order"]] = r
        meta[r["pair_id"]] = r

    rows, stat = [], Counter()
    for pid, orders in sorted(by_pair.items()):
        ab, ba = orders.get("ab"), orders.get("ba")
        if not (ab and ba):
            stat["thieu_mot_chieu"] += 1
            continue
        la, lb = ab["label"], ba["label"]
        cmin = min(ab["confidence"], ba["confidence"])
        if "PARSE_ERROR" in (la, lb) or "API_ERROR" in (la, lb):
            tier_conf = "error"
        elif la != lb:
            tier_conf = "disagree"
        elif cmin >= args.conf_floor:
            tier_conf = "agree_high"
        else:
            tier_conf = "agree_low"
        stat[tier_conf] += 1
        rows.append({
            "pair_id": pid, "a": meta[pid]["a"], "b": meta[pid]["b"],
            "tier": meta[pid]["tier"],
            "label_ab": la, "label_ba": lb,
            "label": la if la == lb else "",
            "conf_min": round(cmin, 3), "agreement": tier_conf,
            "reason_ab": ab.get("reason", ""), "reason_ba": ba.get("reason", ""),
        })

    dump_jsonl(args.out, rows)
    silver = [r for r in rows if r["agreement"] == "agree_high"
              and r["label"] in ("DUPLICATE", "RELATED", "UNRELATED")]
    dump_jsonl(args.silver, silver)
    review = [r for r in rows if r["agreement"] in ("disagree", "agree_low")]
    dump_jsonl(args.review, review)

    n = len(rows) or 1
    print(f"cap co ca hai chieu : {len(rows)}")
    for k in ("agree_high", "agree_low", "disagree", "error", "thieu_mot_chieu"):
        if stat[k]:
            print(f"  {k:<18}{stat[k]:>6}  ({stat[k]/n:.1%})")
    print(f"\nnhan silver (agree_high) : {len(silver)}")
    for t in ("A", "B"):
        sub = [r for r in silver if r["tier"] == t]
        if sub:
            c = Counter(r["label"] for r in sub)
            print(f"  tang {t} ({len(sub)}): " +
                  "  ".join(f"{k}={v}" for k, v in c.most_common()))
    print(f"can nguoi soat           : {len(review)}")

    # doi chieu voi cue cu tren cap gold -- KHONG phai de tin cue, ma de biet
    # LLM co dang chi chep lai nhan cu khong.
    cue = {}
    for p in load_jsonl(args.gold):
        k = tuple(sorted((p.get("a", ""), p.get("b", ""))))
        cue[k] = (p.get("cue", "") or "").strip().lower()
    if cue:
        m = Counter()
        for r in rows:
            c = cue.get(tuple(sorted((r["a"], r["b"]))))
            if c and r["label"]:
                cls = ("duplicate" if c in ("trùng", "duplicate")
                       else "similar" if c in ("tương tự", "giống bug",
                                               "giống lỗi") else "other")
                m[(cls, r["label"])] += 1
        if m:
            print("\nDOI CHIEU voi cue cu (chi de tham khao):")
            for (c, l), v in sorted(m.items(), key=lambda x: -x[1]):
                print(f"  cue={c:<10} -> LLM={l:<10} {v}")
    print(f"\n-> {args.out}\n-> {args.silver}\n-> {args.review}")


# ----------------------------------------------------------------- audit
AUDIT_FIELDS = ["pair_id", "a", "b", "tier", "agreement", "llm_label",
                "human_label", "conf_min", "reason_ab", "summary_a", "summary_b"]


def cmd_audit(args) -> None:
    rows = load_jsonl(args.merged)
    if not rows:
        sys.exit(f"khong co {args.merged} -- chay ingest truoc")
    docs = {}
    for line in open(args.corpus, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            docs[d["key"]] = d.get("summary", "")

    # Lay mau PHAN TANG: neu chi lay ngau nhien thi mau se toan agree_high
    # (nhom de) va kappa do duoc se dep gia tao.
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["agreement"])].append(r)
    per = max(1, args.n // max(1, len(groups)))
    sample = []
    for k in sorted(groups):
        g = groups[k]
        sample += rng.sample(g, min(per, len(g)))
    rng.shuffle(sample)

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        for r in sample:
            w.writerow({
                "pair_id": r["pair_id"], "a": r["a"], "b": r["b"],
                "tier": r["tier"], "agreement": r["agreement"],
                # Voi hang `disagree` thi `label` rong. Dien nhan chieu thuan
                # vao day, neu khong kappa cua nhom do luon = 0 mot cach vo
                # nghia (so nhan that voi chuoi rong). Cot `agreement` van cho
                # thay day la cap khong on dinh.
                "llm_label": r["label"] or r["label_ab"], "human_label": "",
                "conf_min": r["conf_min"], "reason_ab": r["reason_ab"],
                "summary_a": docs.get(r["a"], ""),
                "summary_b": docs.get(r["b"], ""),
            })
    print(f"lay mau {len(sample)} cap (phan tang theo tier x agreement)")
    for k in sorted(groups):
        print(f"  {k}: co {len(groups[k])}, lay {min(per, len(groups[k]))}")
    print(f"\n-> {args.out}")
    print("Dien cot `human_label` bang mot trong: "
          "DUPLICATE / RELATED / UNRELATED / UNCERTAIN, roi chay `kappa`.")


#: Nhan dien tay nen phai chiu duoc go nham. Chi sua loi chinh ta ro rang;
#: gia tri khong nhan ra duoc thi bao loi chu khong doan bua.
_ALIAS = {"REALTED": "RELATED", "RELATE": "RELATED", "RELATED": "RELATED",
          "DUPLICATED": "DUPLICATE", "DUP": "DUPLICATE", "DUPLICATE": "DUPLICATE",
          "UNRELATED": "UNRELATED", "UNRELATE": "UNRELATED",
          "UNCERTAIN": "UNCERTAIN", "UNSURE": "UNCERTAIN"}


def norm_label(s: str) -> str:
    return _ALIAS.get((s or "").strip().upper(), (s or "").strip().upper())


def cmd_kappa(args) -> None:
    with open(args.audit, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if (r.get("human_label") or "").strip()]
    if not rows:
        sys.exit("chua co dong nao dien `human_label`")

    fixed = Counter()
    for r in rows:
        raw = (r.get("human_label") or "").strip().upper()
        if raw and norm_label(raw) != raw:
            fixed[raw] += 1
    if fixed:
        print("da chuan hoa nhan go nham: " +
              ", ".join(f"{k}->{norm_label(k)} ({v})" for k, v in fixed.items()))
    unknown = {norm_label(r["human_label"]) for r in rows} - set(LABELS)
    if unknown:
        sys.exit(f"gia tri human_label khong hop le: {sorted(unknown)}")

    def stats(sub: list[dict]) -> str:
        n = len(sub)
        a = [norm_label(r["llm_label"]) for r in sub]
        h = [norm_label(r["human_label"]) for r in sub]
        agree = sum(1 for x, y in zip(a, h) if x == y) / n
        ca, ch = Counter(a), Counter(h)
        pe = sum(ca[k] / n * ch[k] / n for k in set(a) | set(h))
        k = (agree - pe) / (1 - pe) if pe < 1 else 0.0
        return f"n={n:<5} dong thuan={agree:.1%}  kappa={k:.3f}"

    print("LLM vs NGUOI")
    print(f"  tong        {stats(rows)}")
    for key in ("tier", "agreement"):
        for v in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == v]
            if len(sub) >= 5:
                print(f"  {key}={v:<12}{stats(sub)}")
    print("\n  kappa: <0.20 rat kem | 0.21-0.40 kem | 0.41-0.60 vua |"
          " 0.61-0.80 tot | >0.80 rat tot")
    print("  Bao cao con nay trong luan van: khong co no thi tap nhan LLM"
          " khong bao ve duoc.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    D = "data/processed"

    r = sub.add_parser("run", help="goi LLM")
    r.add_argument("--requests", default=f"{D}/llm_label_requests.jsonl")
    r.add_argument("--out", default=f"{D}/llm_label_responses.jsonl")
    r.add_argument("--provider", default="openai",
                   choices=["openai", "anthropic", "gemini"],
                   help="openai = moi endpoint OpenAI-compatible "
                        "(ke ca Ollama/vLLM noi bo qua --base-url)")
    r.add_argument("--base-url", default="",
                   help="vd http://localhost:11434/v1 cho Ollama")
    r.add_argument("--api-key-env", default="",
                   help="ten bien moi truong chua key; NONE neu khong can")
    r.add_argument("--model", default="",
                   help="mac dinh theo provider, xem DEFAULT_MODEL")
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--tier", default="", choices=["", "A", "B"])
    r.add_argument("--retries", type=int, default=4)
    r.add_argument("--sleep", type=float, default=0.0,
                   help="nghi giua hai luot goi (giay) -- tranh cham han muc")
    r.add_argument("--quota-wait", type=float, default=65.0,
                   help="cho bao lau khi gap 429 (han muc thuong theo phut)")
    r.add_argument("--resume", action="store_true", default=True)
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("ingest", help="hop nhat hai chieu + loc")
    g.add_argument("--responses", default=f"{D}/llm_label_responses.jsonl")
    g.add_argument("--gold", default=f"{D}/dup_candidates_from_text.jsonl")
    g.add_argument("--out", default=f"{D}/llm_labels_merged.jsonl")
    g.add_argument("--silver", default=f"{D}/llm_labels_silver.jsonl")
    g.add_argument("--review", default=f"{D}/llm_labels_needs_review.jsonl")
    # MAC DINH 0.0 -- KHONG loc theo confidence.
    # Do tren 200 phan hoi that cua tang A: confidence p50=0.95, min=0.80,
    # max=1.00. Model KHONG BAO GIO bao duoi 0.8, ke ca tren nhung cap ma
    # chinh no lat nhan khi dao thu tu. Nguong 0.8 hay 0.9 deu cho qua 100%,
    # nen no chi tao cam giac an toan gia. Bo loc that la tinh nhat quan hai
    # chieu. Confidence van duoc ghi lai de doi chieu.
    g.add_argument("--conf-floor", type=float, default=0.0)
    g.set_defaults(func=cmd_ingest)

    a = sub.add_parser("audit", help="lay mau cho nguoi soat")
    a.add_argument("--merged", default=f"{D}/llm_labels_merged.jsonl")
    a.add_argument("--corpus", default=f"{D}/corpus.jsonl")
    a.add_argument("--out", default=f"{D}/llm_label_audit.csv")
    a.add_argument("--n", type=int, default=100)
    a.add_argument("--seed", type=int, default=13)
    a.set_defaults(func=cmd_audit)

    k = sub.add_parser("kappa", help="do dong thuan LLM vs nguoi")
    k.add_argument("--audit", default=f"{D}/llm_label_audit.csv")
    k.set_defaults(func=cmd_kappa)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
