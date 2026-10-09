# -*- coding: utf-8 -*-
"""Phase 1 -- chuan hoa GIU NGUYEN ngu nghia.

Loi cua tokenizer cu (pipeline/retrieval.py::tokenize):

    _TOKEN_RE = r"[A-Za-z0-9A-y][A-Za-z0-9A-y_.\\-/]*"

khong khop "=", "<", ">" -> ba dong duoi day cho ra CUNG mot tui token:

    From Date = To Date      -> {from, date, to, date}
    From Date < To Date      -> {from, date, to, date}
    From Date >= To Date     -> {from, date, to, date}

Toan tu bien mat hoan toan. Do la mat mat ngu nghia nghiem trong voi tieu de
test case QA. Module nay sinh THEM token toan tu ("op_eq", "op_lt") va giu
nguyen so ("num_0" != "num_1"), phu dinh ("neg_khong").

KHONG stem, KHONG gop dong nghia o day: "TERMINATED" va "LOCKED" phai ra hai
token khac nhau. Viec gom dong nghia (neu can) nam o phase 2, va chi ap dung
cho DONG TU thao tac, khong ap dung cho gia tri enum.

Con giu duoc gi sau normalize:
    operator  = < > != <= >=        -> op_eq / op_lt / op_gt / op_ne / ...
    number    0 1 10 11             -> num_0 / num_1 / num_10 / num_11
    negation  khong / not / chua    -> neg_<tu>
    enum      TERMINATED / LOCKED   -> terminated / locked  (van phan biet)
    dinh danh ER80011, trans_id     -> giu nguyen ca cum + mieng con
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .lexicons import NEGATIONS, OPERATOR_TOKENS, STOPWORDS

# Dinh danh ky thuat giu nguyen mot token: ER80011, trans_id, /api/v1/mini-app
_TOKEN_RE = re.compile(r"[A-Za-z0-9À-ỹ][A-Za-z0-9À-ỹ_.\-/]*")
_SPLIT_RE = re.compile(r"[_.\-/]+")
_VN_WORD_RE = re.compile(r"^[a-zà-ỹ]+$")
# Toan tu 2 ky tu phai duoc thu TRUOC toan tu 1 ky tu.
_OP_RE = re.compile(r"<=|>=|!=|<>|==|[<>=≠≥≤]")

# Hai nguon "toan tu gia" trong du lieu that -- do lai tren corpus 4388 bug:
#   MUI TEN      "thuc hien X -> he thong bao loi"  (rat pho bien trong mo ta QA)
#   PLACEHOLDER  "transaction ID <transactionId> is created"  (va the XML con sot)
# Ca hai deu sinh ra "<" / ">" khong he mang nghia SO SANH. De nguyen thi
# op_gt dem duoc 962 lan tren corpus, va mot ben co "->" con ben kia co "="
# se bi bao mau thuan toan tu -- mot mau thuan hoan toan bia ra.
_ARROW_RE = re.compile(r"<-+>|<=+>|-+>|=+>|<-+|<=+(?=\s)")
_PLACEHOLDER_RE = re.compile(r"<\s*/?\s*[A-Za-z_][\w.:\-]*\s*/?\s*>")


def _strip_fake_operators(text: str) -> str:
    """Bo mui ten va placeholder truoc khi doc toan tu so sanh."""
    return _ARROW_RE.sub(" ", _PLACEHOLDER_RE.sub(" ", text or ""))


_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?!\w)")
_PURE_NUM_RE = re.compile(r"^\d+(?:[.,]\d+)?$")

# Khoa dinh danh: con so di sau chung la DUY NHAT moi issue (traceId,
# transId, timestamp...). So sanh chung luon cho ra "khac nhau" nen chung
# khong mang thong tin gi ve viec hai bug co cung ban chat hay khong.
_ID_CONTEXT = {
    "traceid", "transid", "transactionid", "requestid", "orderid", "refid",
    "reference", "batchid", "sessionid", "id", "msisdn", "isdn", "phone",
    "timestamp", "time", "date", "created", "updated", "ver", "version",
}

_ISSUE_REF_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,6}\b")
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b|\b[a-z]+[A-Z]{2,}\b")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn")


@dataclass
class Normalized:
    """Ket qua chuan hoa mot doan van ban."""
    raw: str
    text: str                                  # da NFKC + gon khoang trang
    tokens: list[str] = field(default_factory=list)       # token cho BM25
    words: list[str] = field(default_factory=list)        # tu thuong, bo dau
    operators: list[str] = field(default_factory=list)    # op_eq / op_lt ...
    numbers: list[str] = field(default_factory=list)      # "0" "1" "10"
    negations: list[str] = field(default_factory=list)    # khong / not ...

    @property
    def token_set(self) -> set[str]:
        return set(self.tokens)


class Preprocessor:
    """Chuan hoa van ban ma khong lam mat tin hieu phan biet bug."""

    def __init__(self, cfg=None):
        from .config import PreprocessConfig
        self.cfg = cfg or PreprocessConfig()

    # -------------------------------------------------------------- text
    def clean(self, text: str) -> str:
        """NFKC + gon khoang trang. KHONG bo dau cau mang ngu nghia."""
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", text)
        t = t.replace("\xa0", " ").replace("​", "")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" *\n[ \n]*", "\n", t)
        return t.strip()

    def mask_issue_refs(self, text: str) -> str:
        """Che tham chieu "RO-10049" trong van ban duoc INDEX.

        Ly do (chong ro ri nhan): tap gold duoc dao tu chinh cau
        "trung bug RO-10049" trong comment. Neu de nguyen ma issue trong
        van ban index, mot phan nhan chay nguoc vao feature. Trong van hanh
        that co the bat lai -- day la lua chon cua evaluation, khong phai
        cua san pham.
        """
        return _ISSUE_REF_RE.sub("[ISSUE_REF]", text or "")

    # ------------------------------------------------------------ tokens
    def tokenize(self, text: str) -> list[str]:
        """Token cho lexical retrieval, CO toan tu / so / phu dinh."""
        cfg = self.cfg
        if not text:
            return []
        t = unicodedata.normalize("NFKC", text)
        if cfg.lowercase:
            t = t.lower()

        out: list[str] = []
        plain: list[str] = []
        for m in _TOKEN_RE.finditer(t):
            tok = m.group(0).strip("._-/")
            if not tok or len(tok) > cfg.max_token_len:
                continue
            out.append(tok)
            if _SPLIT_RE.search(tok):
                out.extend(p for p in _SPLIT_RE.split(tok) if len(p) > 1)
            if _VN_WORD_RE.match(tok):
                plain.append(tok)
        if cfg.bigrams:
            out.extend(a + "_" + b for a, b in zip(plain, plain[1:]))

        if cfg.keep_operators:
            out.extend(OPERATOR_TOKENS[o]
                       for o in _OP_RE.findall(_strip_fake_operators(t))
                       if o in OPERATOR_TOKENS)
        if cfg.keep_numbers:
            out.extend("num_" + n.replace(",", ".") for n in _NUM_RE.findall(t))
        if cfg.keep_negation:
            flat = strip_accents(t)
            out.extend("neg_" + w for w in re.findall(r"[a-z]+", flat)
                       if w in NEGATIONS)
        return out

    def operators(self, text: str) -> list[str]:
        return [OPERATOR_TOKENS[o]
                for o in _OP_RE.findall(_strip_fake_operators(text))
                if o in OPERATOR_TOKENS]

    def numbers(self, text: str) -> list[str]:
        return [n.replace(",", ".") for n in _NUM_RE.findall(text or "")]

    def negations(self, text: str) -> list[str]:
        flat = strip_accents((text or "").lower())
        return [w for w in re.findall(r"[a-z]+", flat) if w in NEGATIONS]

    def raw_words(self, text: str) -> list[str]:
        """Tu tho: bo dau, thuong hoa, GIU CA stopword va thu tu.

        Dung cho phan tich pham vi phu dinh -- o do "hien thi" la vi ngu
        quan trong, nhung no lai nam trong STOPWORDS neu loc kieu tui-tu.
        """
        flat = strip_accents((text or "").lower()).replace("_", " ")
        return re.findall(r"[a-z0-9]+", flat)

    @staticmethod
    def bigrams(words: list[str]) -> set[str]:
        return {a + "_" + b for a, b in zip(words, words[1:])}

    def negation_scopes(self, text: str, window: int = 3) -> set[str]:
        """VI NGU BI PHU DINH, duoi dang BIGRAM.

        Vi sao khong dem so tu phu dinh trong ca doan: dem toan van cho ket
        qua sai o dung nhom quan trong nhat cua de tai -- cap dien dat khac
        nhau. Vi du hai cau duoi la MOT loi, nhung mot ben co "khong":

            "nguoi dung KHONG dang nhap duoc sau khi doi mat khau"
            "tai khoan bi tu choi xac thuc sau khi cap nhat mat khau"

        Vi sao BIGRAM chu khong phai tung tu: tieng Viet da am tiet, mot am
        tiet roi khong phai mot vi ngu. Ban dau ham nay tra ve tung tu va no
        sinh ra mau thuan bia hang loat -- do lai tren tap eval:

            A: "... KHONG hien thi du lieu"   -> pham vi {hien, thi, du}
            B: "Loi phan trang du lieu cua field"
            -> khop tren "du" (mot nua cua "du lieu") -> bao LECH PHU DINH

        Hai bug do khong he mau thuan. Voi bigram: pham vi cua A la
        {hien_thi, thi_du}, bigram cua B la {phan_trang, trang_du, du_lieu,
        ...} -> giao rong -> khong con mau thuan gia.

        Truong hop that van bat duoc:
            A "man hinh hien thi thong bao"          bigram co hien_thi
            B "man hinh KHONG hien thi thong bao"    pham vi co hien_thi
            -> mot ben khang dinh, mot ben phu dinh cung vi ngu -> mau thuan
        """
        ws = self.raw_words(text)
        out: set[str] = set()
        for i, w in enumerate(ws):
            if w not in NEGATIONS:
                continue
            # CHI dau ngu ngay sau tu phu dinh -- tuc DONG TU bi phu dinh.
            # Khong lay ca tan ngu: "khong hien thi DU LIEU" ma tinh ca
            # "du_lieu" thi mot bug khac chi can nhac "phan trang du lieu"
            # da bi bao mau thuan, du no khong he khang dinh "hien thi".
            seg = [x for x in ws[i + 1: i + 1 + window] if x not in NEGATIONS]
            if len(seg) >= 2:
                out.add(seg[0] + "_" + seg[1])
        return out

    def numeric_contexts(self, text: str) -> dict[str, set[str]]:
        """{tu dung truoc con so: {cac gia tri}} -- de so sanh so CUNG CHO.

        So sanh thang hai tap con so cua hai bug la qua long. Do lai tren
        tap eval: mot bug co "1" (danh so buoc), bug kia co "5" (so ngay)
        -> bao KHAC GIA TRI SO, hoan toan bia.

        Gan con so vao tu dung truoc no (bo qua toan tu) thi chi con mau
        thuan khi CUNG mot truong mang hai gia tri:

            "Nhap so tien = 0"  ->  {tien: {0}}
            "Nhap so tien = 1"  ->  {tien: {1}}   -> mau thuan that
        """
        out: dict[str, set[str]] = {}
        prev = ""
        # Bat CA CUM chu-so lam mot token roi moi hoi "co phai so thuan
        # khong". Neu tach rieng chu va so thi traceId "5bc6fe8424984f3d"
        # bi bam thanh 5 / 6 / 84 / 24984 ... va sinh ra "mau thuan gia tri
        # so" giua hai bug chi vi hai traceId khac nhau -- do la loi that
        # da bat duoc khi soat tap eval.
        for m in re.finditer(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9_.,]*|[<>=!]+",
                             text or ""):
            tok = m.group(0).strip(".,")
            if not tok:
                continue
            if _PURE_NUM_RE.match(tok):
                if prev and prev not in _ID_CONTEXT:
                    out.setdefault(prev, set()).add(tok.replace(",", "."))
            elif tok[0].isalpha():
                prev = strip_accents(tok.lower())
            # toan tu: giu nguyen `prev` de "so tien = 0" van gan vao "tien"
        return out

    def content_words(self, text: str) -> list[str]:
        """Tu noi dung: bo dau, bo stopword. GIU phu dinh va so."""
        flat = strip_accents((text or "").lower()).replace("_", " ")
        out = []
        for w in re.findall(r"[a-z0-9]+", flat):
            if len(w) < 2 and not w.isdigit():
                continue
            if w in STOPWORDS and w not in NEGATIONS:
                continue
            out.append(w)
        return out

    def camel_identifiers(self, text: str) -> list[str]:
        """requestTime, accessChannelID -> ten field/entity ky thuat."""
        return [m.group(0) for m in _CAMEL_RE.finditer(text or "")]

    # ------------------------------------------------------------ full
    def run(self, text: str, mask_refs: bool | None = None) -> Normalized:
        raw = text or ""
        t = self.clean(raw)
        if self.cfg.mask_issue_refs if mask_refs is None else mask_refs:
            t = self.mask_issue_refs(t)
        return Normalized(
            raw=raw, text=t,
            tokens=self.tokenize(t),
            words=self.content_words(t),
            operators=self.operators(t),
            numbers=self.numbers(t),
            negations=self.negations(t),
        )
