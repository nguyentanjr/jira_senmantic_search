# -*- coding: utf-8 -*-
"""
Xay corpus chuan hoa tu cac file Jira XML export.

    python pipeline/build_corpus.py [--data-dir data] [--out-dir data/processed]

Cac buoc:
    1. Parse XML (full-field export), dedup theo issue key
    2. Scrub credential / PII -> mask GIA TRI, giu TEN FIELD (giu tin hieu lexical)
    3. Tach mo ta nghiep vu khoi stacktrace / log / payload ky thuat
    4. Trich signal lexical (errorCode, traceId, API path, exception, SQL)
    5. Chuan hoa facet (team, root cause, severity, defect type, moi truong...)
    6. Chia corpus / query_seed, danh dau trung lap
"""
import argparse
import glob
import html
import json
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

CODE_MARK = "\x00CODE\x00"


# ------------------------------------------------------------------ 1. PARSE

def strip_html(s, keep_block_breaks=True):
    """HTML (da unescape 1 lan boi ElementTree) -> text, giu ranh gioi block."""
    if not s:
        return ""
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    # danh dau block ky thuat de buoc 3 nhan ra
    s = re.sub(r"(?is)<(pre|code)[^>]*>(.*?)</\1>",
               lambda m: "\n" + CODE_MARK + m.group(2) + "\n", s)
    if keep_block_breaks:
        s = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|ul|ol|blockquote)>", "\n", s)
        s = re.sub(r"(?i)<br\s*/?>", "\n", s)
        s = re.sub(r"(?i)<li[^>]*>", "\n- ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = s.replace("\xa0", " ").replace("​", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n[ \n]*", "\n", s)
    return s.strip()


def parse_file(path):
    root = ET.parse(path).getroot()
    for it in root.iter("item"):
        g = lambda t: (it.findtext(t) or "").strip()
        key = g("key")
        if not key:
            continue

        comments = []
        cs = it.find("comments")
        if cs is not None:
            for c in cs.findall("comment"):
                body = strip_html(c.text or "")
                if body:
                    comments.append({"author": c.get("author", ""),
                                     "created": c.get("created", ""),
                                     "body": body})

        links = []
        il = it.find("issuelinks")
        if il is not None:
            for lt in il.findall("issuelinktype"):
                name = (lt.findtext("name") or "").strip()
                for d in ("outwardlinks", "inwardlinks"):
                    dd = lt.find(d)
                    if dd is None:
                        continue
                    for k in dd.findall("issuelink"):
                        tk = (k.findtext("issuekey") or "").strip()
                        if tk:
                            links.append({"type": name, "dir": d[:-5],
                                          "desc": dd.get("description", ""),
                                          "target": tk})

        atts = []
        att = it.find("attachments")
        if att is not None:
            for a in att.findall("attachment"):
                atts.append({"name": a.get("name", ""), "size": a.get("size", "")})

        cf = {}
        cfs = it.find("customfields")
        if cfs is not None:
            for c in cfs.findall("customfield"):
                nm = (c.findtext("customfieldname") or c.get("id", "")).strip()
                vs = [strip_html(v.text or "") for v in c.iter("customfieldvalue")]
                vs = [v for v in vs if v]
                if vs:
                    cf[nm] = " | ".join(vs)

        yield {
            "key": key,
            "project_key": key.rsplit("-", 1)[0],
            "project": g("project"),
            "type": g("type"),
            "status": g("status"),
            "resolution": g("resolution"),
            "priority": g("priority"),
            "summary": strip_html(g("summary")),
            "description_raw": strip_html(g("description")),
            "environment": strip_html(g("environment")),
            "created": g("created"),
            "updated": g("updated"),
            "reporter": g("reporter"),
            "assignee": g("assignee"),
            "labels": [l.text for l in it.iter("label") if l.text],
            "components": [c.text for c in it.iter("component") if c.text],
            "fix_versions": [c.text for c in it.iter("fixVersion") if c.text],
            "parent": it.findtext("parent") or "",
            "subtasks": [s.text for s in it.iter("subtask") if s.text],
            "comments": comments,
            "links": links,
            "attachments": atts,
            "customfields": cf,
            "source_file": os.path.basename(path),
            "url": g("link"),
        }


# ------------------------------------------------------------------ 2. SCRUB
# Nguyen tac: mask GIA TRI, giu TEN FIELD.
# traceId / requestId / errorCode / responseCode / API path -> GIU NGUYEN.

# Key DAI, khong the la substring vo tinh -> match thoai mai.
SECRET_KEYS = (r"password|passwd|secret|token|api[_-]?key|apikey|"
               r"authorization|credential|checksum|signature|"
               r"private[_-]?key|client[_-]?secret|access[_-]?key")

# Key NGAN -> phai co ranh gioi tu, khong duoc match kieu substring.
# ("errorMaping" chua "pin", "Mapping" chua "pin" -> tung bi che oan.)
SECRET_KEYS_SHORT = (r"(?<![A-Za-z])(?:pin|pwd|otp|cvv|cvc|passcode)(?![A-Za-z])"
                     r"|(?<=[a-z0-9])(?:Pin|Pwd|Otp|Cvv|Cvc|Passcode)(?![A-Za-z])"
                     r"|(?<![A-Za-z])(?:PIN|PWD|OTP|CVV|CVC)(?![A-Za-z])")

# Ma dinh danh ky thuat: chuoi so dai sau cac key nay KHONG phai so the/tai khoan,
# va la tin hieu lexical co gia tri -> bao ve khoi rule long_digits.
TECH_ID_RE = re.compile(
    r"(?i)\b(trans(?:action)?_?id|txn_?id|trace_?id|request_?id|order_?id|"
    r"ref(?:erence)?_?(?:id|no)?|batch_?id|session_?id)"
    r"(\W{1,4})(\d{10,25})\b")

PII_KEYS = (r"msisdn|isdn|phone(?:number)?|mobile|email|id[_-]?no|idnumber|nationalid|nida|"
            r"cust[_-]?name|customer[_-]?name|full[_-]?name|birth[_-]?date|dob|"
            r"card[_-]?(?:no|number)|account[_-]?(?:no|number)|acct[_-]?no|address")


def _mask_kv(text, keys, tag):
    """Mask gia tri sau key: JSON / XML / query string / form / key=value."""
    n = [0]

    def sub_json(m):
        n[0] += 1
        return m.group(1) + '"' + tag + '"'

    text = re.sub(r'(?i)("(?:[\w.\-]*(?:' + keys + r')[\w.\-]*)"\s*:\s*)"[^"]*"',
                  sub_json, text)
    text = re.sub(r'(?i)("(?:[\w.\-]*(?:' + keys + r')[\w.\-]*)"\s*:\s*)(?!")[\w.@+\-]+',
                  sub_json, text)

    def sub_xml(m):
        n[0] += 1
        return m.group(1) + tag + m.group(3)

    text = re.sub(r"(?i)(<([\w:.\-]*(?:" + keys + r")[\w:.\-]*)>)[^<]*(</\2>)",
                  sub_xml, text)

    def sub_kv(m):
        n[0] += 1
        return m.group(1) + tag

    text = re.sub(r"(?i)\b([\w.\-]*(?:" + keys + r")[\w.\-]*\s*[=:]\s*)(?!//)"
                  r"[^\s,;&\"'<>|]{2,}", sub_kv, text)
    return text, n[0]


def scrub(text):
    """-> (text da lam sach, {loai: so lan})"""
    if not text:
        return text, {}
    hits = Counter()

    text, c = _mask_kv(text, SECRET_KEYS, "[REDACTED_SECRET]")
    hits["secret_kv"] += c
    text, c = _mask_kv(text, SECRET_KEYS_SHORT, "[REDACTED_SECRET]")
    hits["secret_kv"] += c
    text, c = _mask_kv(text, PII_KEYS, "[REDACTED_PII]")
    hits["pii_kv"] += c

    # Cat cac ma dinh danh ky thuat ra ngoai truoc khi chay rule long_digits,
    # roi tra lai nguyen ven.
    keep = []

    def _stash(m):
        keep.append(m.group(0))
        return "\x01%d\x01" % (len(keep) - 1)

    text = TECH_ID_RE.sub(_stash, text)

    rules = [
        (r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}(?:\.[A-Za-z0-9_\-]+)?",
         "[REDACTED_JWT]", "jwt"),
        (r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._\-=+/]{16,}",
         r"\1 [REDACTED_TOKEN]", "bearer"),
        (r"(?i)\b(X-[\w\-]*(?:Token|Key|Secret)[\w\-]*)\s*:\s*[^\s\"'<>,;]{8,}",
         r"\1: [REDACTED_TOKEN]", "header_token"),
        (r"\b[A-Za-z0-9+/]{120,}={0,2}\b", "[REDACTED_BLOB]", "blob"),
        (r"(?<![\w.])(?:\d[ \-]?){12,18}\d(?![\w.])",
         "[REDACTED_CARD_OR_ACCT]", "long_digits"),
        (r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]{2,}\b", "[REDACTED_EMAIL]", "email"),
    ]
    for pat, repl, label in rules:
        text, k = re.subn(pat, repl, text)
        if k:
            hits[label] += k

    if keep:
        text = re.sub(r"\x01(\d+)\x01", lambda m: keep[int(m.group(1))], text)
        hits["tech_id_kept"] += len(keep)

    return text, {k: v for k, v in hits.items() if v}


# --------------------------------------- 3. TACH NGHIEP VU / KY THUAT

TECH_PATTERNS = [
    (re.escape(CODE_MARK), 4),
    (r"(?i)\bat\s+[\w$]+(?:\.[\w$]+){2,}\([^)]*\)", 4),
    (r"(?i)\b\w*(?:Exception|Throwable)\b\s*[:(]", 3),
    (r"(?i)\bcaused by\b", 3),
    (r"(?im)^\s*curl\b|--location|--header|-X\s+(?:GET|POST|PUT|DELETE)", 4),
    (r"<[a-z]+:\w+|</[a-z]+:\w+>|soapenv|xmlns:", 4),
    (r'"\w+"\s*:', 2),
    (r"(?i)\b(traceId|transId|requestId|@timestamp|timestamp_log4j|"
     r"Response Body|Request Body|responseCode|errorType)\b", 3),
    (r"(?i)\b(select\s+.+\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from)\b", 4),
    (r"(?i)\b(?:GET|POST|PUT|DELETE|PATCH)\s+/\S+", 3),
    (r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", 2),
    (r"(?i)\b(INFO|DEBUG|WARN|ERROR|TRACE)\b\s*[\[\]|:]", 3),
]

NARRATIVE_PATTERNS = [
    (r"(?i)\b(steps?|các\s*bước|thực hiện|mong muốn|kỳ vọng|expect|actual|kết quả|"
     r"hiện tại|precondition|điều kiện|mô tả|hiển thị|không thể|bị lỗi|yêu cầu|"
     r"kiểm tra|KQTT|KQMM)\b", 3),
]


def _score(block):
    t = sum(w for p, w in TECH_PATTERNS if re.search(p, block, re.M))
    n = sum(w for p, w in NARRATIVE_PATTERNS if re.search(p, block, re.M))
    if len(block) > 40:
        punct = len(re.findall(r"[{}\[\]<>:;\"/\\=|]", block)) / len(block)
        if punct > 0.08:
            t += 2
        if punct > 0.15:
            t += 2
    if len(block) > 300 and (len(block) / (block.count(" ") + 1)) > 12:
        t += 2   # "tu" rat dai -> blob / log
    return t, n


def split_business_technical(text):
    """-> (mo ta nghiep vu, khoi ky thuat)"""
    if not text:
        return "", ""
    blocks = [b for b in re.split(r"\n+", text) if b.strip()]
    biz, tech, buf, buf_tech = [], [], [], None
    for b in blocks:
        t, n = _score(b)
        is_tech = t > n
        if buf and is_tech == buf_tech:
            buf.append(b)
            continue
        if buf:
            (tech if buf_tech else biz).append("\n".join(buf))
        buf, buf_tech = [b], is_tech
    if buf:
        (tech if buf_tech else biz).append("\n".join(buf))

    def clean(xs):
        s = "\n".join(xs).replace(CODE_MARK, "")
        return re.sub(r"\n{3,}", "\n\n", s).strip()

    return clean(biz), clean(tech)


# ------------------------------------------- 4. SIGNAL LEXICAL

SIGNAL_RES = {
    "error_code":  r"\b(?:[A-Z]{2,6}[-_]?\d{3,5}|ER\d{3,6}|PS\d{4}|TP\d{4})\b",
    "http_status": r"(?i)\bHTTP[\s/]*[\d.]*\s*([45]\d{2})\b",
    "exception":   r"\b[\w$]*Exception\b",
    "api_path":    r"(?<![\w.])/(?:api|v\d)[\w./\-{}]{2,60}",
    "url_host":    r"https?://([\w.\-]+)",
    "class_id":    r"\b(?:[a-z]\w*\.){2,}[A-Z]\w*\b",
    "trace_id":    r"(?i)\b(?:traceId|transId|requestId)\b[\"'\s:=]+([\w\-]{6,})",
    "func_code":   r"(?i)\b(?:funcCode|serviceCode|partnerCode|roleTypeCode)\b[\"'\s:=]+\"?(\w{2,})",
}

# Signal dung de DINH NGHIA nhom lexical-advantage.
# Loai url_host (boilerplate: 1 host staging chiem 25% corpus -> khong phan biet duoc)
# va trace_id (gan nhu duy nhat moi issue -> truy van kieu tra cuu 1 ban ghi,
# khong phai tim issue tuong tu).
DISCRIMINATIVE_SIGNALS = ("error_code", "func_code", "exception",
                          "class_id", "api_path", "http_status")

# Issue key ("RO-11923") khop pattern error_code -> phai loai.
JIRA_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")


def extract_signals(text, project_keys=()):
    pk = {p.upper() for p in project_keys}
    out = {}
    for name, pat in SIGNAL_RES.items():
        found = re.findall(pat, text or "")
        found = [f if isinstance(f, str) else f[0] for f in found]
        found = [f.strip() for f in found if f and not f.startswith("[REDACTED")]
        if name == "error_code":
            found = [f for f in found
                     if not (JIRA_KEY_RE.match(f) and f.split("-")[0].upper() in pk)]
        if found:
            seen, uniq = set(), []
            for f in found:
                if f.lower() not in seen:
                    seen.add(f.lower())
                    uniq.append(f)
            out[name] = uniq[:20]
    return out


# ------------------------------------------- 5. SECTION + FACET

# Cho phep enumerator dan dau ("A.", "B)", "1.") va nhan ket thuc bang
# dau ":" HOAC het dong -- template pho bien trong du lieu la:
#     A.Precondition:
#     B. Step
#     C. Actual Result
#     D. Expected Result
SECTION_RE = re.compile(
    r"(?im)^[\s\-*>]*(?:[A-Z]\s*[.)]\s*|\d+\s*[.)]\s*)?"
    r"(steps?|các\s*bước|precondition|pre[-\s]?condition|điều\s*kiện|data|input|"
    r"actual(?:\s*result)?|kqtt|kết\s*quả\s*thực\s*tế|thực\s*tế|hiện\s*tại|"
    r"expect(?:ed)?(?:\s*result)?|kqmm|kết\s*quả\s*mong\s*muốn|mong\s*muốn|kỳ\s*vọng|"
    r"note|ghi\s*chú|nguyên\s*nhân|root\s*cause)"
    r"\s*(?:[:\-–]|$)\s*")

SECTION_CANON = {
    "steps": "steps", "step": "steps", "các bước": "steps",
    "data": "steps", "input": "steps",
    "precondition": "precondition", "pre-condition": "precondition",
    "pre condition": "precondition", "điều kiện": "precondition",
    "actual": "actual", "actual result": "actual", "kqtt": "actual",
    "kết quả thực tế": "actual", "thực tế": "actual", "hiện tại": "actual",
    "expect": "expected", "expected": "expected",
    "expect result": "expected", "expected result": "expected", "kqmm": "expected",
    "kết quả mong muốn": "expected", "mong muốn": "expected", "kỳ vọng": "expected",
    "note": "note", "ghi chú": "note",
    "nguyên nhân": "root_cause_text", "root cause": "root_cause_text",
}


def parse_sections(text):
    if not text:
        return {}
    labels = [(m.start(), m.end(), re.sub(r"\s+", " ", m.group(1).strip().lower()))
              for m in SECTION_RE.finditer(text)]
    if not labels:
        return {}
    out = {}
    for i, (s, e, lab) in enumerate(labels):
        end = labels[i + 1][0] if i + 1 < len(labels) else len(text)
        canon = SECTION_CANON.get(lab)
        if not canon:
            continue
        val = text[e:end].strip()
        if val:
            out[canon] = (out[canon] + "\n" + val).strip() if canon in out else val
    return out


FACET_MAP = {
    "team":        "Team phụ trách",
    "root_cause":  "Root Cause",
    "defect_type": "Defect Type",
    "severity":    "Severity",
    "test_env":    "Môi trường kiểm thử",
    "uat_env":     "Môi trường nghiệm thu (BA)",
    "sprint":      "Sprint",
    "epic":        "Epic Link",
    "impact":      "Phạm vi ảnh hưởng",
    "work_class":  "Phân loại công việc",
    "partner":     "Đối tác",
    "cause_note":  "Nguyên nhân",
}

JIRA_DATE = re.compile(r"(\d{1,2})\s+(\w{3})\s+(\d{4})")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def iso_date(s):
    m = JIRA_DATE.search(s or "")
    if not m:
        return ""
    d, mo, y = m.groups()
    return "%s-%02d-%02d" % (y, MONTHS[mo], int(d)) if mo in MONTHS else ""


# ------------------------------------------- 6. DEDUP

def norm_key_text(s):
    s = unicodedata.normalize("NFKC", (s or "").lower())
    s = re.sub(r"\[[^\]]{0,40}\]", " ", s)          # bo tien to [Module]
    s = re.sub(r"[^\w\sà-ỹ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


TOKEN_RE = re.compile(r"[0-9a-zà-ỹ_]{2,}")


def toks(s):
    return set(TOKEN_RE.findall(unicodedata.normalize("NFKC", (s or "").lower())))


def find_near_dups(recs, thr=0.72):
    """Blocking bang inverted index tren token it pho bien -> jaccard."""
    tk = {r["key"]: toks(norm_key_text(r["summary"])) for r in recs}
    df = Counter()
    for s in tk.values():
        df.update(s)
    inv = defaultdict(list)
    for k, s in tk.items():
        for t in s:
            if 1 < df[t] <= 300:
                inv[t].append(k)
    cand = Counter()
    for t, ks in inv.items():
        if len(ks) > 120:
            continue
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = (ks[i], ks[j]) if ks[i] < ks[j] else (ks[j], ks[i])
                cand[(a, b)] += 1
    pairs = []
    for (a, b), c in cand.items():
        if c < 2:
            continue
        A, B = tk[a], tk[b]
        if not A or not B:
            continue
        j = len(A & B) / len(A | B)
        if j >= thr:
            pairs.append({"a": a, "b": b, "jaccard": round(j, 3)})
    pairs.sort(key=lambda x: -x["jaccard"])
    return pairs


# ------------------------- 7. KHAI THAC CAP TRUNG LAP TU VAN BAN
# Jira link khong co quan he "Duplicates", nhung ky su ghi tay trong comment:
#   "tuong tu bug RO-10049", "trung giong bug RO-10025", "van de tuong tu 11730"
# -> suy ra cap ground-truth mien phi.

DUP_CUE_RE = re.compile(
    r"(?i)(trùng|duplicate|tương tự|giống\s+(?:bug|issue|case|lỗi)|"
    r"đã\s+(?:có|report|log)\s+(?:bug|issue)|same as)")

# Bat "RO-123" / "browse/RO-123", hoac so tran 3-6 chu so nhung PHAI dung rieng.
# Lookaround dung \w (khong chi \d): neu khong, "849"/"283" bi boc ra tu trong
# UUID "f849bde5-283d-4b1a" va tu trans_id=1374572842562015232.
DUP_REF_RE = re.compile(r"(?i)(?:\bRO-(\d{1,6})\b|(?<![\w.])(\d{3,6})(?![\w.]))")


def mine_text_dup_candidates(recs, window=140):
    keys = {r["key"] for r in recs}
    out, seen = [], set()
    for r in recs:
        text = (r["text_comments"] or "") + "\n" + (r["text_business"] or "")
        for m in DUP_CUE_RE.finditer(text):
            seg = text[m.start():m.start() + window]
            for g1, g2 in DUP_REF_RE.findall(seg):
                tgt = "RO-" + (g1 or g2)
                if tgt not in keys or tgt == r["key"]:
                    continue
                pair = tuple(sorted((r["key"], tgt)))
                if pair in seen:
                    continue
                seen.add(pair)
                out.append({"a": pair[0], "b": pair[1],
                            "cue": m.group(1).lower(),
                            "found_in": r["key"],
                            "context": re.sub(r"\s+", " ", seg)[:160]})
    return out


# ------------------------------------------------------------------ BUILD

BUG_TYPES_RE = re.compile(r"(?i)\bbug\b|problem")


def build(rec, project_keys=()):
    hits = Counter()
    desc, h = scrub(rec["description_raw"]); hits.update(h)
    summ, h = scrub(rec["summary"]);         hits.update(h)
    cmts = []
    for c in rec["comments"]:
        b, h = scrub(c["body"]); hits.update(h)
        cmts.append({**c, "body": b})

    biz, tech = split_business_technical(desc)
    cbiz, ctech = [], []
    for c in cmts:
        b, t = split_business_technical(c["body"])
        if b:
            cbiz.append(b)
        if t:
            ctech.append(t)

    all_tech = "\n".join([tech] + ctech).strip()
    signals = extract_signals(desc + "\n" + "\n".join(c["body"] for c in cmts),
                              project_keys)
    sections = parse_sections(desc)
    lexical_group = any(k in signals for k in DISCRIMINATIVE_SIGNALS)

    cf = rec["customfields"]
    facets = {k: cf.get(v, "") for k, v in FACET_MAP.items()}
    facets["team_group"] = facets["team"].split("|")[-1].strip() if facets["team"] else ""

    text_dense = "\n".join(x for x in [
        summ,
        sections.get("actual") or biz,
        sections.get("expected", ""),
    ] if x).strip() or summ

    text_bm25 = "\n".join(x for x in [
        summ, biz, all_tech,
        " ".join(v for vs in signals.values() for v in vs),
        " ".join(rec["labels"]),
    ] if x).strip()

    return {
        "key": rec["key"], "url": rec["url"],
        "type": rec["type"], "status": rec["status"],
        "resolution": rec["resolution"], "priority": rec["priority"],
        "summary": summ,
        "text_business": biz,
        "text_technical": all_tech,
        "text_comments": "\n".join(cbiz),
        "sections": sections,
        "signals": signals,
        "facets": facets,
        "labels": rec["labels"], "components": rec["components"],
        "fix_versions": rec["fix_versions"],
        "created": iso_date(rec["created"]), "updated": iso_date(rec["updated"]),
        "reporter": rec["reporter"], "assignee": rec["assignee"],
        "parent": rec["parent"], "subtasks": rec["subtasks"],
        "n_comments": len(cmts), "n_attachments": len(rec["attachments"]),
        "attachments": [a["name"] for a in rec["attachments"]],
        "links": rec["links"],
        "is_bug": bool(BUG_TYPES_RE.search(rec["type"])),
        "lexical_group": lexical_group,
        "scrubbed": dict(hits),
        "text_dense": text_dense,
        "text_bm25": text_bm25,
        "source_file": rec["source_file"],
    }


# ------------------------------------------------------------------ MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-desc", type=int, default=100)
    ap.add_argument("--seed-min-desc", type=int, default=250)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.data_dir, "*.xml")))
    if not files:
        sys.exit("khong thay file .xml nao trong " + a.data_dir)

    raw, dup_files = {}, Counter()
    for f in files:
        n_new = 0
        for rec in parse_file(f):
            if rec["key"] in raw:
                dup_files[(raw[rec["key"]]["source_file"], rec["source_file"])] += 1
                continue
            raw[rec["key"]] = rec
            n_new += 1
        print("  %-24s +%d" % (os.path.basename(f), n_new))
    print("\n%d issue unique tu %d file" % (len(raw), len(files)))
    if dup_files:
        print("  file trung dai:")
        for (x, y), c in dup_files.most_common():
            print("    %s & %s: %d key trung" % (x, y, c))

    project_keys = sorted({r["project_key"] for r in raw.values()})
    print("  project key: " + ", ".join(project_keys))
    recs = [build(r, project_keys) for r in raw.values()]

    # Dedup noi dung. Chi so sanh khi CO noi dung -- neu khong, moi ban ghi
    # rong se dung chung khoa ("", "") va bi gan trung lan nhau.
    sig, exact = {}, []
    for r in recs:
        ns, nb = norm_key_text(r["summary"]), norm_key_text(r["text_business"])
        r["dup_of"] = ""
        if not ns and not nb:
            continue
        s = (ns, nb[:400])
        if s in sig:
            exact.append({"dup": r["key"], "of": sig[s]})
            r["dup_of"] = sig[s]
        else:
            sig[s] = r["key"]

    corpus = [r for r in recs
              if r["is_bug"] and len(r["text_business"]) + len(r["summary"]) >= a.min_desc]
    seed = [r for r in corpus
            if not r["dup_of"] and len(r["text_business"]) >= a.seed_min_desc
            and {"actual", "expected"} & set(r["sections"])]
    near = find_near_dups(corpus)
    dup_text = mine_text_dup_candidates(recs)

    def dump(name, rows):
        p = os.path.join(a.out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  -> %s  (%d)" % (p, len(rows)))

    print("\nghi file:")
    dump("all_issues.jsonl", recs)
    dump("corpus.jsonl", corpus)
    dump("query_seed.jsonl", seed)
    dump("near_duplicates.jsonl", near)
    dump("dup_candidates_from_text.jsonl", dup_text)
    dump("scrub_audit.jsonl",
         [{"key": r["key"], **r["scrubbed"]} for r in recs if r["scrubbed"]])

    R = []
    P = R.append
    pct = lambda n, d: 100.0 * n / max(1, d)
    N = len(corpus)

    P("\n" + "=" * 74)
    P("BAO CAO")
    P("=" * 74)
    P("tong issue                    %d" % len(recs))
    P("  la bug                      %d" % sum(1 for r in recs if r["is_bug"]))
    P("corpus (bug & desc>=%d)      %d" % (a.min_desc, N))
    P("query_seed                    %d" % len(seed))
    P("trung noi dung y het          %d (toan bo) / %d (trong corpus)"
      % (len(exact), sum(1 for r in corpus if r["dup_of"])))
    P("cap gan-trung (jaccard>=0.72) %d" % len(near))
    P("cap trung-lap suy tu van ban  %d  (can nguoi soat lai)" % len(dup_text))
    if dup_text:
        cc = Counter(d["cue"] for d in dup_text)
        P("   cue: " + ", ".join("%s=%d" % kv for kv in cc.most_common(6)))
        nb = sum(1 for d in dup_text
                 if d["a"] in {r["key"] for r in corpus}
                 and d["b"] in {r["key"] for r in corpus})
        P("   ca 2 phia deu nam trong corpus: %d" % nb)

    nsc = sum(1 for r in recs if r["scrubbed"])
    P("\n-- SCRUB -- issue bi che: %d (%.1f%%)" % (nsc, pct(nsc, len(recs))))
    tot = Counter()
    for r in recs:
        tot.update(r["scrubbed"])
    for k, v in tot.most_common():
        P("   %7d  %s" % (v, k))

    P("\n-- TACH NGHIEP VU / KY THUAT (corpus) --")
    nb = sum(1 for r in corpus if r["text_business"])
    nt = sum(1 for r in corpus if r["text_technical"])
    P("   co phan nghiep vu          %d (%.0f%%)" % (nb, pct(nb, N)))
    P("   co phan ky thuat           %d (%.0f%%)" % (nt, pct(nt, N)))
    for nm, f in (("nghiep vu", lambda r: r["text_business"]),
                  ("ky thuat", lambda r: r["text_technical"])):
        L = sorted(len(f(r)) for r in corpus if f(r))
        if L:
            P("   do dai %-10s median=%d p90=%d" % (nm, L[len(L) // 2], L[9 * len(L) // 10]))

    P("\n-- SECTION nhan dien duoc (corpus) --")
    for k, v in Counter(k for r in corpus for k in r["sections"]).most_common():
        P("   %6d (%4.0f%%)  %s" % (v, pct(v, N), k))

    P("\n-- SIGNAL LEXICAL (corpus) --")
    for k, v in Counter(k for r in corpus for k in r["signals"]).most_common():
        P("   %6d (%4.0f%%)  %s" % (v, pct(v, N), k))
    P("   (url_host + trace_id KHONG tinh vao nhom lexical -- xem"
      " DISCRIMINATIVE_SIGNALS)")
    nlex = sum(1 for r in corpus if r["lexical_group"])
    P("   %6d (%4.0f%%)  nhom LEXICAL-advantage  (co ma loi/func/exception/class/API)"
      % (nlex, pct(nlex, N)))
    P("   %6d (%4.0f%%)  nhom SEMANTIC-advantage (thuan ngon ngu nghiep vu)"
      % (N - nlex, pct(N - nlex, N)))

    P("\n-- FACET (corpus, do phu) --")
    for k in list(FACET_MAP) + ["team_group"]:
        v = [r["facets"].get(k, "") for r in corpus]
        nn = sum(1 for x in v if x)
        P("   %6d (%4.0f%%)  %-12s %4d gia tri"
          % (nn, pct(nn, N), k, len(set(x for x in v if x))))

    txt = "\n".join(R)
    print(txt)
    with open(os.path.join(a.out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
