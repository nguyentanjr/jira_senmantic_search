# -*- coding: utf-8 -*-
"""
Kien thuc domain cua du lieu RO: tien to [Tag] trong summary KHONG phai nhieu.

Phat hien tu review cua ky su:
    [USSD EU][Show balance]  vs  [USSD supper agent][check balance]
    -> KHONG phai trung lap. EU / Agent / Super Agent la ba TAC NHAN khac nhau,
       cung mot trieu chung tren ba tac nhan la ba bug khac nhau.

    [APP EU][Bundle]  vs  [EU_APP][Buy Bundle]
    -> LA trung lap. "APP EU" = "EU_APP" (chi khac thu tu/dau noi),
       chuc nang trung nhau.

Nen quy tac dung la:
    trung lap  <=> CUNG kenh  VA  CUNG tac nhan  VA  chuc nang trung nhieu
    hard neg   <=> KHAC kenh hoac tac nhan  NHUNG  van ban rat giong

Ban dau code strip het [Tag] -> mat dung tin hieu phan biet, sinh ra
hang tram cap "nghi trung" sai.
"""
from __future__ import annotations

import re
import unicodedata

TAG_RE = re.compile(r"\[([^\]]{1,60})\]")
_SEP_RE = re.compile(r"[\s_\-/,]+")

# Kenh giao tiep -- rut ra tu tan suat thuc te trong corpus
CHANNEL = {"ussd", "app", "web", "api", "cms", "webview", "portal", "sms",
           "backoffice", "bo", "mobile"}

# Tac nhan / vai tro -- day la truc phan biet quan trong nhat
ACTOR = {"eu", "agent", "merchant", "customer", "enterprise", "client",
         "master", "admin", "isv", "staff", "user", "consumer", "partner",
         "super", "sub", "cc"}

# Chuan hoa bien the viet
ALIAS = {
    "sa": ("super", "agent"),
    "supper": ("super",),
    "supperagent": ("super", "agent"),
    "superagent": ("super", "agent"),
    "submerchant": ("sub", "merchant"),
    "subagent": ("sub", "agent"),
    "eneterprise": ("enterprise",),
    "enteprirse": ("enterprise",),
    "custommer": ("customer",),
}

# Tu chi moi truong / giai doan -- KHONG phan biet bug, bo di khi so sanh
ENV_WORDS = {"staging", "production", "prod", "beta", "release", "uat",
             "test", "testing", "dev", "ba", "perf", "clio", "cr", "retest",
             "hotfix", "bug", "issue", "new", "old"}


def _norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for w in _SEP_RE.split(text):
        w = re.sub(r"[^a-z0-9]", "", w)
        if not w:
            continue
        out.extend(ALIAS.get(w, (w,)))
    return out


def parse_tags(summary: str) -> dict:
    """-> {channel:set, actor:set, function:set, has_tag:bool}"""
    tags = TAG_RE.findall(summary or "")
    words: list[str] = []
    for t in tags:
        words.extend(_norm_words(t))
    channel = {w for w in words if w in CHANNEL}
    actor = {w for w in words if w in ACTOR}
    function = {w for w in words
                if w not in CHANNEL and w not in ACTOR and w not in ENV_WORDS
                and len(w) > 1}
    return {"channel": channel, "actor": actor, "function": function,
            "has_tag": bool(tags)}


def same_context(a: dict, b: dict) -> bool:
    """Cung kenh VA cung tac nhan?

    Chi ket luan KHAC khi ca hai ben deu co thong tin do -- neu mot ben
    thieu tag thi khong du can cu de loai, tra ve True (de buoc so sanh
    van ban quyet dinh).
    """
    for field in ("channel", "actor"):
        x, y = a[field], b[field]
        if x and y and x != y:
            return False
    return True


def context_label(t: dict) -> str:
    c = "+".join(sorted(t["channel"])) or "?"
    a = "+".join(sorted(t["actor"])) or "?"
    return f"{c}/{a}"


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Dong nghia dong tu thao tac -- gom lai de "show balance" == "check balance".
# KHONG gom approve/reject hay bat/tat: do la hai thao tac NGUOC nhau.
VERB_SYNONYM = {
    "show": "view", "check": "view", "get": "view", "display": "view",
    "xem": "view", "hien": "view", "hienthi": "view",
    "add": "create", "new": "create", "them": "create", "tao": "create",
    "edit": "update", "modify": "update", "sua": "update",
    "remove": "delete", "xoa": "delete",
    "list": "search", "find": "search", "query": "search",
}

# Tu chung chung: xuat hien mot ben khong du de ket luan la hai bug khac nhau.
# Giu danh sach NGAN. Ban dau co "info"/"detail"/"screen" -> lam
# "[Customer Info]" va "[Query Balance]" bi coi la trung lap, vi "info" bi bo
# qua nen chenh lech thanh mot chieu. Tu chi doi tuong/man hinh la CO phan biet.
GENERIC = {"function", "chuc", "nang", "case", "tc",
           "for", "of", "the", "and", "with", "va"}


def _canon_function(fn: set[str]) -> set[str]:
    return {VERB_SYNONYM.get(w, w) for w in fn}


# --- Chu ky "vi mo" cua tieu de test case -------------------------------
# Trong tieu de QA, thong tin phan biet thuong nam o TOAN TU, CON SO va
# TU PHU DINH -- dung nhung token ma chuan hoa van ban thong thuong vut di:
#     "From Date = To Date"  vs  "From Date < To Date"   -> hai test case
#     "nhap 10 ky tu"        vs  "nhap 11 ky tu"         -> hai test case
#     "hien thi"             vs  "khong hien thi"        -> nguoc nhau
OPERATOR_RE = re.compile(r"<=|>=|!=|<>|[<>=≠≥≤]")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?!\w)")
NEGATION = {"khong", "ko", "kg", "chua", "chang", "not", "no", "without",
            "never", "unable", "fail", "failed"}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Tu chuc nang: xuat hien lech mot ben KHONG du de ket luan hai bug khac nhau.
STOPWORDS = {
    # tieng Viet (da bo dau)
    "khi", "va", "cua", "cho", "voi", "tai", "trong", "de", "la", "co", "bi",
    "duoc", "thi", "ma", "nhung", "van", "con", "cac", "nay", "do", "den",
    "tu", "ra", "vao", "len", "xuong", "sau", "truoc", "hay", "hoac", "neu",
    "nhu", "tren", "duoi", "ve", "boi", "theo", "gi", "khac", "lai", "roi",
    "thuc", "hien", "kiem", "tra", "loi", "bug", "case", "tc", "test",
    # tieng Anh
    "the", "and", "or", "of", "for", "with", "to", "in", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "when", "then", "that", "this",
    "not", "no", "a", "an", "it", "as", "from", "error", "issue",
}


def body_tokens(summary: str) -> set[str]:
    """Token phan than summary (ngoai [tag]), da bo dau va stopword.

    Dung de bat token phan biet nam NGOAI tag:
        "Bam Save"      vs "Bam Submit"          -> nut khac nhau
        "role = LOCKED" vs "role = TERMINATED"   -> gia tri enum khac nhau
        "role = LOCKED" vs "role = Locked"       -> chi khac hoa thuong => GIONG
    """
    body = TAG_RE.sub(" ", summary or "")
    words = set()
    for w in _strip_accents(body.lower()).replace("_", " ").split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) >= 2 and w not in STOPWORDS and w not in GENERIC:
            words.add(w)
    return words


def _body_bigrams(summary: str) -> set[str]:
    """Bigram phan than, giu thu tu tu -- bat truong hop tui-tu bo sot."""
    body = TAG_RE.sub(" ", summary or "")
    ws = []
    for w in _strip_accents(body.lower()).replace("_", " ").split():
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) >= 2 and w not in STOPWORDS and w not in GENERIC:
            ws.append(VERB_SYNONYM.get(w, w))
    return {a + "_" + b for a, b in zip(ws, ws[1:])}


def micro_signature(summary: str) -> tuple:
    """(toan tu, con so, so luong tu phu dinh) -- chu ky phan biet test case."""
    s = summary or ""
    ops = frozenset(OPERATOR_RE.findall(s))
    nums = frozenset(NUMBER_RE.findall(s))
    words = _strip_accents(s.lower()).split()
    neg = sum(1 for w in words if re.sub(r"[^a-z]", "", w) in NEGATION)
    return ops, nums, neg


def pair_signals(summary_a: str, summary_b: str, text_sim: float) -> dict:
    """Rut cac TIN HIEU doc lap cua mot cap, khong ket luan.

    Jaccard chi tra loi "hai text co nhieu token chung khong" -- no KHONG
    tra loi "hai bug co cung behavior/root cause khong". Nen no la mot tin
    hieu, khong phai tieu chi quyet dinh.
    """
    ta, tb = parse_tags(summary_a), parse_tags(summary_b)
    oa, na, ga = micro_signature(summary_a)
    ob, nb, gb = micro_signature(summary_b)

    fa, fb = _canon_function(ta["function"]), _canon_function(tb["function"])
    f_only_a = {w for w in fa - fb if w not in GENERIC}
    f_only_b = {w for w in fb - fa if w not in GENERIC}

    ba = _canon_function(body_tokens(summary_a))
    bb = _canon_function(body_tokens(summary_b))
    b_only_a, b_only_b = ba - bb, bb - ba

    # So sanh tui-tu mat thong tin vi tri:
    #     "Bam Save   ... chuc nang Save"
    #     "Bam Submit ... chuc nang Save"
    # -> A-B rong (vi "save" co o ca hai) nen chenh chi mot chieu, lot luoi.
    # Bigram giu duoc vi tri: {bam_save} vs {bam_submit} -> chenh hai chieu.
    ga2, gb2 = _body_bigrams(summary_a), _body_bigrams(summary_b)
    g_only_a, g_only_b = ga2 - gb2, gb2 - ga2

    return {
        "text_sim": round(text_sim, 3),
        "operator_diff": sorted(oa ^ ob),
        "number_diff": sorted(na ^ nb),
        "negation_diff": ga != gb,
        "channel_diff": sorted(ta["channel"] ^ tb["channel"])
        if (ta["channel"] and tb["channel"]) else [],
        "actor_diff": sorted(ta["actor"] ^ tb["actor"])
        if (ta["actor"] and tb["actor"]) else [],
        # chenh HAI chieu moi tinh: mot chieu chi la cach viet day du hon
        "function_diff": sorted(f_only_a | f_only_b) if (f_only_a and f_only_b) else [],
        "entity_diff": sorted(b_only_a | b_only_b) if (b_only_a and b_only_b) else [],
        "order_diff": sorted(g_only_a | g_only_b)[:6]
        if (g_only_a and g_only_b) else [],
    }


# Tin hieu du manh de tu dong ket luan "hai bug khac nhau".
# EU / Agent / Super Agent, hay USSD / APP, la cac truc phan biet ro rang --
# da duoc ky su xac nhan tren nhieu vi du.
STRONG_DIFF = ("actor_diff", "channel_diff")

# Tin hieu chi du de NGHI NGO -> dua nguoi soat, khong tu dong gan nhan.
WEAK_DIFF = ("operator_diff", "number_diff", "negation_diff",
             "function_diff", "entity_diff", "order_diff")


def classify_pair(summary_a: str, summary_b: str, text_sim: float,
                  dup_min_sim: float = 0.72) -> tuple[str, dict]:
    """-> (nhan, tin hieu)

    Nhan:
      'unrelated'  van ban khong du giong
      'different'  co tin hieu MANH (khac tac nhan / kenh) -> hard negative
      'review'     giong nhau nhung co tin hieu yeu -> NGUOI quyet dinh
      'duplicate'  giong nhau va khong tin hieu phan biet nao

    Bat doi xung chi phi (theo yeu cau nghiep vu): false positive trong tap
    fine-tune nguy hiem hon bo sot -- no day model keo hai bug KHAC NHAU lai
    gan nhau. Nen moi tin hieu yeu deu day sang 'review', khong tu gan nhan.
    """
    sig = pair_signals(summary_a, summary_b, text_sim)
    if text_sim < dup_min_sim:
        return "unrelated", sig
    if any(sig[k] for k in STRONG_DIFF):
        return "different", sig
    if any(sig[k] for k in WEAK_DIFF):
        return "review", sig
    return "duplicate", sig
