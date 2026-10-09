# -*- coding: utf-8 -*-
"""Tu dien domain -- TAT CA deu rut ra tu thong ke corpus that, khong doan.

Moi nhom o day tra loi mot cau hoi khac nhau cho scoring:

    PROVIDERS   khac nha cung cap  -> penalty NHE (co the cung root cause)
    STATUSES    khac trang thai    -> contradiction MANH
    ACTION_*    thao tac loai tru  -> contradiction MANH
    NEGATIONS   phu dinh lech nhau -> contradiction MANH

Doi chieu tan suat (corpus 4388 bug, quet summary + text_business):
    halopesa 646, luku 220, halotel 196, axieva 144, bima 128, selcom 93,
    tips 87, tukuza 85, nmb 68, crdb 55, umoja 44, zctv 31, ttcl 22 ...
    approve 882, confirm 657, lock 372, reject 303, block 271, save 140,
    submit 80, unlock 180, unblock 110 ...
    active 299, locked 218, failed 217, processing 194, pending 163,
    blocked 154, dormant 87, terminated 73, cancelled 81 ...
"""
from __future__ import annotations

# ------------------------------------------------------------- nha cung cap
# Ngan hang / vi / doi tac tien ich xuat hien trong corpus (thi truong Tanzania).
BANKS = {
    "crdb", "nmb", "nbc", "azania", "dtb", "exim", "kcb", "stanbic", "absa",
    "equity", "akiba", "mkombozi", "tpb", "amana", "maendeleo", "uba",
    "ecobank", "letshego",
}

WALLETS = {
    "halopesa", "mpesa", "tigopesa", "azampesa", "mixx",
}

PROVIDERS = BANKS | WALLETS | {
    "halotel", "airtel", "tigo", "vodacom", "zantel", "ttcl",
    "selcom", "tips", "umoja", "tukuza", "luku", "zctv", "dstv", "startimes",
    "azam", "axieva", "bima", "visa", "mastercard", "jumia",
}

# ------------------------------------------------------------------ kenh
CHANNELS = {
    "ussd", "app", "web", "api", "cms", "webview", "portal", "sms",
    "backoffice", "bo", "mobile", "ivr",
}

# ------------------------------------------------------------- trang thai
# Enum trang thai: hai gia tri khac nhau => hai test case khac nhau.
# Day la truc hard-negative duoc neu dich danh trong yeu cau (TERMINATED/LOCKED).
STATUSES = {
    "active", "inactive", "locked", "unlocked", "terminated", "dormant",
    "blocked", "unblocked", "suspended", "pending", "processing", "failed",
    "success", "successful", "cancelled", "canceled", "expired", "approved",
    "rejected", "closed", "draft", "transformed", "verified",
    "unverified", "registered", "unregistered",
}

# Gia tri trang thai NAM CUNG MOT TRUC -- lech nhau la mau thuan that.
# ("new" khong xep vao truc nao: no con dung nhu tinh tu thong thuong.)
STATUS_AXES = [
    {"active", "inactive", "locked", "unlocked", "terminated", "dormant",
     "blocked", "unblocked", "suspended", "closed", "transformed"},
    {"pending", "processing", "failed", "success", "successful", "cancelled",
     "canceled", "expired", "approved", "rejected"},
    {"verified", "unverified"},
    {"registered", "unregistered"},
]

# ------------------------------------------------------------------ thao tac
# Dong nghia -> gom ve mot dai dien. CHI gom khi that su cung hanh vi.
# KHONG gom approve/reject, lock/unlock: do la thao tac NGUOC nhau.
ACTION_CANON = {
    "save": "save", "luu": "save",
    "submit": "submit",
    "send": "send", "gui": "send",
    "resend": "resend",
    "approve": "approve", "duyet": "approve",
    "reject": "reject",
    "cancel": "cancel", "huy": "cancel",
    "confirm": "confirm", "xacnhan": "confirm",
    "create": "create", "add": "create", "tao": "create",
    "them": "create", "register": "create", "dangky": "create",
    "update": "update", "edit": "update", "modify": "update", "sua": "update",
    "capnhat": "update",
    "delete": "delete", "remove": "delete", "xoa": "delete",
    "view": "view", "show": "view", "check": "view", "display": "view",
    "query": "view", "inquiry": "view", "xem": "view", "hienthi": "view",
    "search": "search", "filter": "search", "find": "search",
    "timkiem": "search", "list": "search",
    "login": "login", "signin": "login", "dangnhap": "login",
    "logout": "logout", "signout": "logout", "dangxuat": "logout",
    "lock": "lock", "khoa": "lock",
    "unlock": "unlock", "mokhoa": "unlock",
    "block": "block", "chan": "block",
    "unblock": "unblock",
    "activate": "activate", "kichhoat": "activate",
    "deactivate": "deactivate",
    "transfer": "transfer", "chuyen": "transfer",
    "withdraw": "withdraw", "rut": "withdraw", "cashout": "withdraw",
    "deposit": "deposit", "nap": "deposit", "cashin": "deposit",
    "topup": "topup", "recharge": "topup",
    "refund": "refund", "hoantien": "refund",
    "reverse": "reverse", "reversal": "reverse",
    "retry": "retry",
    "upload": "upload",
    "download": "download",
    "export": "export", "xuat": "export",
    "import": "import",
    "verify": "verify", "xacthuc": "verify",
    "buy": "buy", "mua": "buy", "purchase": "buy",
    "sell": "sell", "ban": "sell",
    "upgrade": "upgrade", "nangcap": "upgrade",
}

# Nhom thao tac LOAI TRU LAN NHAU: hai thao tac khac nhau trong cung nhom
# => hai bug khac nhau, du van ban gan nhu trung ("Bam Save" vs "Bam Submit").
ACTION_EXCLUSIVE_GROUPS = [
    {"save", "submit", "send", "confirm"},
    {"approve", "reject", "cancel"},
    {"lock", "unlock"},
    {"block", "unblock"},
    {"activate", "deactivate"},
    {"login", "logout"},
    {"create", "delete", "update"},
    {"deposit", "withdraw", "transfer", "topup"},
    {"buy", "sell"},
    {"upload", "download"},
    {"export", "import"},
]

# ------------------------------------------------------------- trieu chung
# HIEN TUONG quan sat duoc, doc tu phan duoi [tag] cua summary.
#
# Vi sao khong dung expected/actual cho viec nay: do lai tren corpus, ba bug
# false-positive dien hinh deu co `actual` RONG va `expected` gan nhu trung
# nhau ("He thong tra dung gia tri {token} ve cho khach hang"). Thong tin
# phan biet nam o duoi tag cua summary: "trung voi referenceNumber" /
# "sms gui sai" / "tra ve null".
#
# Danh sach giu NGAN va chi gom tu KHONG da nghia sau khi bo dau. Da loai:
#   trang  -- "man hinh trang" lan voi "phan trang" / "trang chu" (8.3%)
#   hang   -- "treo" lan voi "hang hoa" / "hang san xuat"
#   rong   -- "rong" lan voi "rong rai"
# Tan suat trong summary: sai 22.1%, thieu 4.8%, timeout 1.3%, null 1.2%,
# trung 0.8%.
SYMPTOM_CANON = {
    "sai": "wrong", "wrong": "wrong", "incorrect": "wrong", "lech": "wrong",
    "trung": "duplicated", "duplicate": "duplicated", "duplicated": "duplicated",
    "null": "null",
    "thieu": "missing", "missing": "missing",
    "timeout": "timeout",
    "busy": "busy",
}


def symptom_conflict(a: set[str], b: set[str]) -> list[str]:
    """Hai hien tuong khac han nhau -> nhieu kha nang la hai bug khac nhau.

    Chi tinh khi CA HAI ben deu doc duoc trieu chung va khong giao nhau --
    mot ben khong ta hien tuong thi la thieu thong tin, khong phai mau thuan.
    """
    if not a or not b or (a & b):
        return []
    return [f"{u}!={v}" for u in sorted(a) for v in sorted(b)]


# ------------------------------------------------------------------ phu dinh
# CHI tu phu dinh that. "fail/failed" tung nam o day va do la sai: chung la
# TU CHI KET QUA, khong phai phu dinh -- "transaction failed" hai ben deu co
# thi khong co gi mau thuan, va chung da duoc xu ly o truc trang thai.
NEGATIONS = {
    "khong", "ko", "kg", "chua", "chang",
    "not", "no", "without", "never", "unable", "cannot", "cant", "missing",
}

# Tu chuc nang thuan tuy -- khong dung lam "vi ngu bi phu dinh"
NEG_SCOPE_IGNORE = {
    "la", "co", "bi", "duoc", "cac", "mot", "the", "a", "an", "is", "are",
    "to", "of", "and", "or", "in", "on", "at", "it", "that", "this",
}

# ------------------------------------------------------------------ toan tu
# Giu nguyen phan biet: "=" va "<" phai la HAI token khac nhau.
OPERATOR_TOKENS = {
    "<=": "op_le", ">=": "op_ge", "!=": "op_ne", "<>": "op_ne",
    "==": "op_eq", "=": "op_eq", "<": "op_lt", ">": "op_gt",
    "≤": "op_le", "≥": "op_ge", "≠": "op_ne",
}

# Toan tu doi nghich -- lech nhau la mau thuan, khong chi "khac chu".
OPERATOR_CONFLICT_PAIRS = {
    frozenset(("op_eq", "op_lt")), frozenset(("op_eq", "op_gt")),
    frozenset(("op_eq", "op_ne")), frozenset(("op_lt", "op_gt")),
    frozenset(("op_lt", "op_ge")), frozenset(("op_gt", "op_le")),
    frozenset(("op_le", "op_ge")), frozenset(("op_eq", "op_le")),
    frozenset(("op_eq", "op_ge")),
}

# ------------------------------------------------------------------ vai tro
ACTORS = {
    "eu", "agent", "merchant", "customer", "enterprise", "client", "master",
    "admin", "isv", "staff", "user", "consumer", "partner", "super", "sub",
    "cc", "cashier", "operator", "supervisor",
}

ACTOR_ALIAS = {
    "sa": ("super", "agent"), "supper": ("super",),
    "supperagent": ("super", "agent"), "superagent": ("super", "agent"),
    "submerchant": ("sub", "merchant"), "subagent": ("sub", "agent"),
    "eneterprise": ("enterprise",), "enteprirse": ("enterprise",),
    "custommer": ("customer",),
}

# Tu chi moi truong / giai doan -- KHONG phan biet bug
ENV_WORDS = {
    "staging", "production", "prod", "beta", "release", "uat", "test",
    "testing", "dev", "ba", "perf", "clio", "cr", "retest", "hotfix",
    "bug", "issue", "new", "old", "sit", "sanity", "regression",
}

# Tu qua chung -- xuat hien lech mot ben khong ket luan duoc gi
GENERIC = {
    "function", "chuc", "nang", "case", "tc", "for", "of", "the", "and",
    "with", "va", "cua", "khi", "la", "co", "bi", "duoc", "thi", "ma",
    "system", "screen", "man", "hinh",
}

# Stopword cho so sanh tui-tu (da bo dau).
# LUU Y: KHONG dua tu phu dinh vao day -- phu dinh la tin hieu, khong phai nhieu.
STOPWORDS = {
    "khi", "va", "cua", "cho", "voi", "tai", "trong", "de", "la", "co", "bi",
    "duoc", "thi", "ma", "nhung", "van", "con", "cac", "nay", "do", "den",
    "tu", "ra", "vao", "len", "xuong", "sau", "truoc", "hay", "hoac", "neu",
    "nhu", "tren", "duoi", "ve", "boi", "theo", "gi", "khac", "lai", "roi",
    "thuc", "hien", "kiem", "tra", "bug", "case", "tc", "test",
    "the", "and", "or", "of", "for", "with", "to", "in", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "when", "then", "that", "this",
    "a", "an", "it", "as", "from", "issue",
}


def canon_action(word: str) -> str | None:
    return ACTION_CANON.get(word)


def _cross(xa: set[str], xb: set[str]) -> list[str]:
    return [f"{u}!={v}" for u in sorted(xa) for v in sorted(xb)]


def action_conflict(a: set[str], b: set[str]) -> list[str]:
    """Cac cap thao tac loai tru nhau giua hai tap."""
    out: list[str] = []
    for grp in ACTION_EXCLUSIVE_GROUPS:
        xa, xb = a & grp, b & grp
        if xa and xb and xa != xb:
            out.extend(_cross(xa - xb, xb - xa))
    return out


def status_conflict(a: set[str], b: set[str]) -> list[str]:
    """Hai trang thai cung truc nhung khac gia tri -> mau thuan."""
    out: list[str] = []
    for axis in STATUS_AXES:
        xa, xb = a & axis, b & axis
        if xa and xb and xa != xb:
            out.extend(_cross(xa - xb, xb - xa))
    return out


def operator_conflict(a: set[str], b: set[str]) -> list[str]:
    out: list[str] = []
    for u in sorted(a - b):
        for v in sorted(b - a):
            if frozenset((u, v)) in OPERATOR_CONFLICT_PAIRS:
                out.append(f"{u}!={v}")
    return out
