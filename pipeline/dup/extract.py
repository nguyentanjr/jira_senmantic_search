# -*- coding: utf-8 -*-
"""Phase 2 -- bieu dien co cau truc cua mot Jira bug.

Nguyen tac:
  * rule-based + pattern matching ro rang, KHONG doan.
  * khong chac -> de rong / None. Mot truong rong nghia la "khong biet",
    va phase 4 se coi "khong biet" la KHONG co bang chung, chu khong phai
    bang chung phu dinh.
  * extractor la module rieng -> thay bang model khac ma khong dong pipeline.

Nguon du lieu cho tung truong (theo do tin cay giam dan):
    tag [..] trong summary   -> channel, actor, function, module
    customfields (facets)    -> team, root_cause, defect_type, partner
    signals da trich san     -> error_code, api_path, exception, class_id
    section actual/expected  -> observed_behavior / expected_behavior
    quet van ban             -> provider, status, action, entity, field
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import lexicons as LX
from .preprocess import Preprocessor, strip_accents

TAG_RE = re.compile(r"\[([^\]]{1,60})\]")
_SEP_RE = re.compile(r"[\s_\-/,]+")
_FIELD_QUOTED_RE = re.compile(r"[\"'“”‘’]([A-Za-z][\w \-]{1,30})[\"'“”‘’]")
_ENUM_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,20}\b")


@dataclass
class BugRepresentation:
    """Bieu dien co cau truc. Truong rong = khong biet, khong phai "khac"."""
    key: str
    summary: str = ""
    text: str = ""                       # van ban dung cho embedding
    tokens: list[str] = field(default_factory=list)
    token_set: set[str] = field(default_factory=set)
    words: set[str] = field(default_factory=set)

    # -- ngu canh
    channel: set[str] = field(default_factory=set)
    actor: set[str] = field(default_factory=set)
    function: set[str] = field(default_factory=set)
    module: set[str] = field(default_factory=set)
    domain: set[str] = field(default_factory=set)
    provider: set[str] = field(default_factory=set)
    bank: set[str] = field(default_factory=set)
    api: set[str] = field(default_factory=set)

    # -- thanh phan nghiep vu
    entities: set[str] = field(default_factory=set)
    fields_: set[str] = field(default_factory=set)
    actions: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)
    error_codes: set[str] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)
    trace_ids: set[str] = field(default_factory=set)

    # -- chu ky vi mo
    operators: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    negations: int = 0
    neg_scope: set[str] = field(default_factory=set)    # vi ngu bi phu dinh (bigram)
    raw_words: set[str] = field(default_factory=set)    # tu tho, ke ca stopword
    raw_bigrams: set[str] = field(default_factory=set)  # bigram tho, giu thu tu
    number_ctx: dict = field(default_factory=dict)      # {tu truoc: {gia tri}}
    symptoms: set[str] = field(default_factory=set)     # hien tuong quan sat duoc

    # -- hanh vi
    expected_behavior: str | None = None
    observed_behavior: str | None = None
    expected_words: set[str] = field(default_factory=set)
    observed_words: set[str] = field(default_factory=set)

    # -- meta tho (de debug / hien thi)
    facets: dict = field(default_factory=dict)
    status_field: str = ""
    lexical_group: bool = False

    def context_label(self) -> str:
        c = "+".join(sorted(self.channel)) or "?"
        a = "+".join(sorted(self.actor)) or "?"
        return f"{c}/{a}"

    def brief(self) -> dict:
        """Rut gon de in ra log / bao cao."""
        return {
            "key": self.key,
            "context": self.context_label(),
            "function": sorted(self.function)[:6],
            "provider": sorted(self.provider),
            "actions": sorted(self.actions),
            "statuses": sorted(self.statuses),
            "error_codes": sorted(self.error_codes)[:4],
            "operators": sorted(self.operators),
            "numbers": sorted(self.numbers)[:6],
            "negations": self.negations,
        }


class StructuredExtractor:
    """doc (dict tu corpus.jsonl) -> BugRepresentation."""

    def __init__(self, cfg=None, pre: Preprocessor | None = None):
        from .config import ExtractConfig
        self.cfg = cfg or ExtractConfig()
        self.pre = pre or Preprocessor()

    # ------------------------------------------------------------- tags
    def _tag_words(self, summary: str) -> list[list[str]]:
        out = []
        for t in TAG_RE.findall(summary or ""):
            ws = []
            for w in _SEP_RE.split(strip_accents(t.lower())):
                w = re.sub(r"[^a-z0-9]", "", w)
                if w:
                    ws.extend(LX.ACTOR_ALIAS.get(w, (w,)))
            if ws:
                out.append(ws)
        return out

    # ------------------------------------------------------------- main
    def extract(self, doc: dict) -> BugRepresentation:
        cfg = self.cfg
        key = doc.get("key", "")
        summary = doc.get("summary", "") or ""
        sections = doc.get("sections", {}) or {}
        signals = doc.get("signals", {}) or {}
        facets = doc.get("facets", {}) or {}

        dense_text = doc.get("text_dense") or summary
        body = "\n".join(x for x in (summary, doc.get("text_business", "")) if x)

        rep = BugRepresentation(key=key, summary=summary, text=dense_text,
                                facets=facets,
                                status_field=doc.get("status", ""),
                                lexical_group=bool(doc.get("lexical_group")))

        if not cfg.enabled:
            # Tat phase 2: chi con van ban tho (dung cho ablation).
            n = self.pre.run(body)
            rep.tokens, rep.token_set = n.tokens, n.token_set
            rep.words = set(n.words)
            return rep

        n_body = self.pre.run(body)
        rep.tokens, rep.token_set = n_body.tokens, n_body.token_set
        rep.words = set(n_body.words)

        # -- chu ky vi mo: toan tu / so / phu dinh (chi tren summary +
        #    expected/actual -- day la noi mang y dinh test case)
        micro_src = "\n".join(x for x in (
            summary, sections.get("expected", ""), sections.get("actual", "")) if x)
        micro = self.pre.run(micro_src)
        rep.operators = set(micro.operators)
        rep.numbers = set(micro.numbers)
        rep.negations = len(micro.negations)
        rep.neg_scope = self.pre.negation_scopes(micro_src)
        _rw = self.pre.raw_words(micro_src)
        rep.raw_words = set(_rw)
        rep.raw_bigrams = self.pre.bigrams(_rw)
        rep.number_ctx = self.pre.numeric_contexts(micro_src)

        # -- trieu chung: doc o phan duoi [tag] cua summary va o `actual`.
        #    KHONG doc `expected` -- do la trang thai MONG MUON, khong phai
        #    hien tuong loi, va no gan nhu la cau mau ("He thong hien thi
        #    dung...") nen khong phan biet duoc gi.
        sym_src = TAG_RE.sub(" ", summary) + " " + (sections.get("actual") or "")
        for w in re.findall(r"[a-z0-9]+", strip_accents(sym_src.lower())):
            c = LX.SYMPTOM_CANON.get(w)
            if c:
                rep.symptoms.add(c)

        # -- tag: kenh / vai tro / chuc nang / module
        if cfg.use_tags:
            for ws in self._tag_words(summary):
                rep.channel |= {w for w in ws if w in LX.CHANNELS}
                rep.actor |= {w for w in ws if w in LX.ACTORS}
                fn = {w for w in ws
                      if w not in LX.CHANNELS and w not in LX.ACTORS
                      and w not in LX.ENV_WORDS and len(w) > 1}
                rep.function |= fn
                if fn:
                    rep.module.add("_".join(ws))

        # -- domain: component + epic + team
        if cfg.use_facets:
            for c in doc.get("components", []) or []:
                rep.domain.add(strip_accents(str(c).lower()).strip())
            for fkey in ("team_group", "epic"):
                v = facets.get(fkey) or ""
                if v:
                    rep.domain.add(strip_accents(v.lower()).strip()[:60])

        # -- provider / bank: quet ca summary + body (facet partner gan nhu rong)
        flat = strip_accents(body.lower())
        flat_words = set(re.findall(r"[a-z0-9]+", flat))
        rep.provider = flat_words & LX.PROVIDERS
        rep.bank = flat_words & LX.BANKS
        p = facets.get("partner") or ""
        if p:
            rep.provider |= set(re.findall(r"[a-z0-9]+", strip_accents(p.lower()))) \
                & LX.PROVIDERS

        # -- api / error code / identifier tu signals da trich san
        if cfg.use_signals:
            rep.api = {a.lower() for a in signals.get("api_path", [])[:10]}
            rep.error_codes = {e.upper() for e in signals.get("error_code", [])[:10]}
            rep.error_codes |= {"HTTP" + s for s in signals.get("http_status", [])[:5]}
            # CHI funcCode/serviceCode/partnerCode moi la dinh danh nghiep vu
            # dung de so mau thuan. class_id (ten class trong stacktrace) da
            # tung nam o day va do la sai: hai bug cua cung mot he thong bao
            # gio cung dump ra bo class khac nhau tuy log, nen moi cap deu bi
            # bao "khac dinh danh". Class name chuyen sang lam bang chung DUONG.
            rep.identifiers = {f.lower() for f in signals.get("func_code", [])[:5]}
            rep.entities |= {c for c in signals.get("class_id", [])[:5]}
            # traceId gan nhu duy nhat moi issue -> KHONG bao gio dung lam
            # bang chung mau thuan, neu khong moi cap deu "khac dinh danh".
            rep.trace_ids = {i for i in signals.get("trace_id", [])[:5]}
            for e in signals.get("exception", [])[:5]:
                rep.entities.add(e)

        # -- action: dong tu thao tac (chi lay tu co trong tu dien)
        for w in flat_words:
            a = LX.canon_action(w)
            if a:
                rep.actions.add(a)
        for ws in self._tag_words(summary):
            for w in ws:
                a = LX.canon_action(w)
                if a:
                    rep.actions.add(a)

        # -- status: chi nhan gia tri enum, uu tien dang VIET HOA trong van ban
        for m in _ENUM_RE.finditer(body):
            w = m.group(0).lower()
            if w in LX.STATUSES:
                rep.statuses.add(w)
        # dang thuong: chi nhan khi di sau dau hieu gan gia tri (= : la/is/status)
        for m in re.finditer(
                r"(?i)\b(?:status|state|trang\s*thai|role|type)\b[^\n]{0,20}?"
                r"([A-Za-z_]{3,20})", body):
            w = strip_accents(m.group(1).lower())
            if w in LX.STATUSES:
                rep.statuses.add(w)

        # -- entity / field ky thuat
        cam = self.pre.camel_identifiers(body)
        rep.fields_ |= {c for c in cam[:cfg.max_entities]}
        rep.fields_ |= {q.strip() for q in _FIELD_QUOTED_RE.findall(summary)[:10]}
        rep.entities |= {w for w in rep.function if len(w) > 2}
        rep.entities |= {f.lower() for f in rep.fields_}

        # -- hanh vi expected / observed
        if cfg.use_sections:
            exp = (sections.get("expected") or "").strip()
            obs = (sections.get("actual") or "").strip()
            rep.expected_behavior = exp or None
            rep.observed_behavior = obs or None
            rep.expected_words = set(self.pre.content_words(exp)) if exp else set()
            rep.observed_words = set(self.pre.content_words(obs)) if obs else set()

        return rep

    def extract_query(self, text: str, key: str = "__query__") -> BugRepresentation:
        """Truy van tu do (ky su go mo ta) -- coi nhu mot bug khong co metadata."""
        return self.extract({"key": key, "summary": text, "text_dense": text,
                             "text_business": "", "sections": {}, "signals": {},
                             "facets": {}})
