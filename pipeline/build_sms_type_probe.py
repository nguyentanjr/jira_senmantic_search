"""Build a candidate list for the SMS defect-type ordering probe.

Question (user, 2026-09-23): when the query is an "extra SMS" issue, "wrong SMS
content" issues may appear but must rank below other "extra SMS" issues.
This script only proposes issues per group; the user confirms the group in
`candidates.csv` before anything is measured.

Groups:
  thua      - SMS sent too many times / duplicated / to extra recipients
  sai       - SMS sent, but content wrong (missing param, wrong amount, template)
  khong_gui - SMS that should be sent is not sent (control group)

    python pipeline/build_sms_type_probe.py
"""
import argparse
import collections
import csv
import json
import re

SMS = re.compile(r"\bsms\b|tin nhắn|bản tin|notification|notify", re.I)
THUA = re.compile(r"(gửi|nhận|bắn|sinh)\s*(ra\s*)?(thừa|lặp|trùng|double|dup"
                  r"|2 lần|hai lần|nhiều lần|2 sms|2 tin|thêm \d* ?(sms|tin))"
                  r"|(sms|tin nhắn|bản tin)\s*(bị\s*)?(thừa|lặp|trùng|double)"
                  r"|thừa\s*(sms|tin nhắn|bản tin)", re.I)
SAI = re.compile(r"(sms|tin nhắn|bản tin|nội dung)[^.\n]{0,40}"
                 r"(sai|thiếu|thừa param|lỗi font|chưa đúng|không đúng|bị lỗi hiển thị"
                 r"|hiển thị (sai|lỗi)|null|undefined|\{\w+\})"
                 r"|sai nội dung|sai template|thiếu param|thừa\s*(kí|ký)\s*tự", re.I)
KHONG = re.compile(r"(không|chưa)\s*(gửi|nhận được|nhận|bắn|sinh)\s*(sms|tin nhắn|bản tin|thông báo)"
                   r"|(sms|tin nhắn|bản tin)[^.\n]{0,20}(không|chưa)\s*(được\s*)?(gửi|nhận)", re.I)
# "SMS thừa ký tự" is wrong content; "thiếu SMS" / "bổ sung gửi thêm" is a missing SMS.
NOT_THUA = re.compile(r"thừa\s*(kí|ký)\s*tự|thiếu\s*(sms|tin nhắn|bản tin)|gửi thiếu|bổ sung", re.I)
WRONG_RECIPIENT = re.compile(r"(gửi|nhận)[^.\n]{0,15}(nhầm|sai)\s*(người|đối tượng|số|sđt)", re.I)
ACTUAL = re.compile(r"actual result\s*:?\s*(.*?)(?:\n\s*[A-Z]\s*\.\s*expected|expected result|$)",
                    re.I | re.S)
EXPECTED = re.compile(r"expected result\s*:?\s*(.*?)$", re.I | re.S)
FLOW = re.compile(r"^\s*\[([^\]]+)\]")


def clip(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def flow_of(summary):
    m = FLOW.match(summary or "")
    return m.group(1).strip().lower() if m else ""


def load_direction(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["key"]] = d
    return out


def classify(doc, direction):
    summary = doc.get("summary", "")
    body = doc.get("text_business", "")
    m = ACTUAL.search(body)
    actual = m.group(1) if m else ""
    focus = summary + "\n" + actual
    if not SMS.search(summary + "\n" + body):
        return None
    lab = direction.get(doc["key"], {})
    hits = {
        "thua": (bool(THUA.search(focus)) or (lab.get("axis") == "send_count" and lab.get("side") == "extra"))
                and not NOT_THUA.search(summary),
        "sai": bool(SAI.search(focus)),
        "khong_gui": bool(KHONG.search(summary)) or (
            lab.get("axis") == "send_count" and lab.get("side") == "missing" and bool(KHONG.search(focus))),
    }
    groups = [g for g, v in hits.items() if v]
    if len(groups) != 1:
        return None  # ambiguous or none: leave out of the proposal
    g = groups[0]
    # strength: evidence in the title is stronger than in the body only
    title_hit = {"thua": THUA, "sai": SAI, "khong_gui": KHONG}[g].search(summary)
    return {
        "group": g,
        "strength": 2 if title_hit else 1,
        "wrong_recipient": bool(WRONG_RECIPIENT.search(focus)),
        "direction_label": f"{lab.get('axis', '')}:{lab.get('side', '')}".strip(":"),
        "actual": actual,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl")
    ap.add_argument("--direction", default="data/processed/direction_labels_llm.jsonl")
    ap.add_argument("--out", default="data/processed/sms_type_probe_20260923/candidates.csv")
    ap.add_argument("--quota", default="thua=15,sai=15,khong_gui=10")
    args = ap.parse_args()

    quota = {k: int(v) for k, v in (p.split("=") for p in args.quota.split(","))}
    direction = load_direction(args.direction)
    pool = collections.defaultdict(list)
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            c = classify(doc, direction)
            if c:
                pool[c["group"]].append((doc, c))

    # Prefer flows where more than one group exists: that is where the
    # extra-vs-wrong confusion can actually happen.
    flows_by_group = {g: {flow_of(d["summary"]) for d, _ in items} for g, items in pool.items()}
    def shared(doc):
        fl = flow_of(doc["summary"])
        return sum(fl in s for s in flows_by_group.values()) if fl else 0

    rows = []
    for g, n in quota.items():
        items = sorted(pool[g], key=lambda x: (-x[1]["strength"], -shared(x[0]), x[0]["key"]))
        seen_flows = collections.Counter()
        picked = []
        for doc, c in items:  # at most 2 per flow so one screen cannot dominate
            fl = flow_of(doc["summary"])
            if fl and seen_flows[fl] >= 2:
                continue
            seen_flows[fl] += 1
            picked.append((doc, c))
            if len(picked) == n:
                break
        for doc, c in picked:
            m = EXPECTED.search(doc.get("text_business", ""))
            rows.append({
                "key": doc["key"],
                "suggested_group": g,
                "confirm_group": "",
                "note": "",
                "boundary_wrong_recipient": "yes" if c["wrong_recipient"] else "",
                "flow": flow_of(doc["summary"]),
                "summary": doc["summary"],
                "actual": clip(c["actual"], 500),
                "expected": clip(m.group(1) if m else "", 400),
                "direction_label": c["direction_label"],
                "url": doc.get("url", ""),
            })
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("pool sizes:", {g: len(v) for g, v in pool.items()})
    print("written:", collections.Counter(r["suggested_group"] for r in rows), "->", args.out)


if __name__ == "__main__":
    main()
