# -*- coding: utf-8 -*-
"""Lop dich vu dung lai dung hai cau hinh retrieval da duoc chot."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import resolution
from pipeline.build_corpus import scrub
from pipeline.build_pooling_set import describe
from pipeline.retrieval import (CrossEncoderReranker, DenseIndex, SearchConfig,
                                SearchEngine, Timing, load_jsonl, tokenize)
from pipeline.run_ablation import CONFIGS
from pipeline.search import FLOWS, plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
DEFAULT_FEEDBACK = ROOT / "data" / "pilot" / "feedback.jsonl"
DEFAULT_SEARCH_LOG = ROOT / "data" / "pilot" / "searches.jsonl"
_SPACE_RE = re.compile(r"\s+")
FILTER_FIELDS = ("project", "component", "status", "priority",
                 "created_from", "created_to")
FAILURE_REASONS = {"wrong_module", "wrong_defect_type",
                   "opposite_direction", "other"}


class PilotError(ValueError):
    """Loi du lieu dau vao co the hien an toan tren UI."""


def _project(doc: dict) -> str:
    key = str(doc.get("key", ""))
    return key.rsplit("-", 1)[0] if "-" in key else ""


def _service_components(doc: dict) -> list[str]:
    values = list(doc.get("components") or [])
    facets = doc.get("facets") or {}
    # team_group is derived from team (the last segment after "|").
    # Show it only when the full team name is unavailable.
    values.append(facets.get("team") or facets.get("team_group", ""))
    return sorted({str(x).strip() for x in values if str(x).strip()}, key=str.casefold)


def _short(text: str, limit: int) -> str:
    text = _SPACE_RE.sub(" ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def matched_snippet(doc: dict, query: str, limit: int = 360) -> str:
    """Chon dong mo ta co nhieu token truy van trung nhat, fallback ve dau mo ta."""
    body = describe(doc, 2400)
    if not body:
        return _short(doc.get("summary", ""), limit)
    chunks = [_short(x, 700) for x in re.split(r"[\r\n]+|(?<=[.!?])\s+", body)
              if x.strip()]
    q_tokens = set(tokenize(query, bigrams=False))
    if not chunks or not q_tokens:
        return _short(body, limit)
    best = max(enumerate(chunks),
               key=lambda p: (len(q_tokens & set(tokenize(p[1], bigrams=False))),
                              -p[0]))[1]
    return _short(best if set(tokenize(best, bigrams=False)) & q_tokens else body, limit)


def filter_doc_ids(docs: list[dict], filters: dict[str, Any] | None) -> set[int] | None:
    """Tra tap doc id de pre-filter; None nghia la khong co dieu kien loc."""
    raw = filters or {}
    project = str(raw.get("project", "")).strip().casefold()
    component = str(raw.get("component", "")).strip().casefold()
    status = str(raw.get("status", "")).strip().casefold()
    priority = str(raw.get("priority", "")).strip().casefold()
    created_from = str(raw.get("created_from", "")).strip()
    created_to = str(raw.get("created_to", "")).strip()
    if created_from and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", created_from):
        raise PilotError("Ngày bắt đầu phải có dạng YYYY-MM-DD.")
    if created_to and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", created_to):
        raise PilotError("Ngày kết thúc phải có dạng YYYY-MM-DD.")
    if created_from and created_to and created_from > created_to:
        raise PilotError("Ngày bắt đầu không thể sau ngày kết thúc.")
    active = any((project, component, status, priority, created_from, created_to))
    if not active:
        return None

    out: set[int] = set()
    for i, doc in enumerate(docs):
        created = str(doc.get("created", ""))[:10]
        if project and _project(doc).casefold() != project:
            continue
        if component and component not in {x.casefold() for x in _service_components(doc)}:
            continue
        if status and str(doc.get("status", "")).strip().casefold() != status:
            continue
        if priority and str(doc.get("priority", "")).strip().casefold() != priority:
            continue
        if created_from and (not created or created < created_from):
            continue
        if created_to and (not created or created > created_to):
            continue
        out.add(i)
    return out


def _facet(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"value": k, "count": v}
            for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0].casefold()))]


class PilotSearchService:
    """Tai corpus mot lan, lazy-load model, ghi telemetry va feedback local."""

    def __init__(self, corpus_path: str | os.PathLike[str] = DEFAULT_CORPUS,
                 feedback_path: str | os.PathLike[str] = DEFAULT_FEEDBACK,
                 device: str = "cpu",
                 search_log_path: str | os.PathLike[str] = DEFAULT_SEARCH_LOG):
        self.corpus_path = Path(corpus_path)
        self.feedback_path = Path(feedback_path)
        self.search_log_path = Path(search_log_path)
        self.device = device
        self.docs = load_jsonl(str(self.corpus_path))
        if not self.docs:
            raise RuntimeError(f"Corpus rỗng: {self.corpus_path}")
        self.by_key = {str(d["key"]).upper(): i for i, d in enumerate(self.docs)}
        self._engines: dict[str, SearchEngine] = {}
        self._encoder_model = None
        self._search_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._search_log_lock = threading.Lock()
        self._search_records: dict[str, dict[str, Any]] = {}
        self._search_count = 0
        if self.search_log_path.exists():
            with self.search_log_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    search_id = str(row.get("search_id", ""))
                    if search_id:
                        self._search_records[search_id] = row
                        self._search_count += 1
        self.timings = {"text": Timing(), "issue": Timing()}
        self.total_ms: dict[str, list[float]] = {"text": [], "issue": []}
        self.started_at = datetime.now(timezone.utc).isoformat()

    def metadata(self) -> dict[str, Any]:
        projects, components, statuses, priorities = Counter(), Counter(), Counter(), Counter()
        dates = []
        for d in self.docs:
            if _project(d):
                projects[_project(d)] += 1
            for c in _service_components(d):
                components[c] += 1
            if d.get("status"):
                statuses[str(d["status"])] += 1
            if d.get("priority"):
                priorities[str(d["priority"])] += 1
            if d.get("created"):
                dates.append(str(d["created"])[:10])
        return {
            "corpus_size": len(self.docs),
            "filters": {
                "projects": _facet(projects),
                "components": _facet(components),
                "statuses": _facet(statuses),
                "priorities": _facet(priorities),
                "created_min": min(dates) if dates else "",
                "created_max": max(dates) if dates else "",
            },
            "flows": {
                "text": {"label": "Mô tả tự do", "model": FLOWS["text"]["model"],
                         "config": FLOWS["text"]["config"],
                         "rerank_k": FLOWS["text"]["rerank_k"]},
                "issue": {"label": "Từ một issue", "model": FLOWS["issue"]["model"],
                          "config": FLOWS["issue"]["config"], "rerank_k": 0},
            },
        }

    def issue_suggestions(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        q = query.strip().casefold()
        if not q:
            return []
        prefix, contains = [], []
        for d in self.docs:
            key, summary = str(d.get("key", "")), str(d.get("summary", ""))
            target = f"{key} {summary}".casefold()
            if key.casefold().startswith(q):
                prefix.append({"key": key, "summary": _short(summary, 140)})
            elif q in target:
                contains.append({"key": key, "summary": _short(summary, 140)})
        return (prefix + contains)[:max(1, min(int(limit), 20))]

    def _engine(self, flow: str) -> SearchEngine:
        if flow in self._engines:
            return self._engines[flow]
        f = FLOWS[flow]
        model_path = f["model"]
        reranker_path = f["reranker"]
        for path in (model_path, reranker_path if flow == "text" else ""):
            if path.startswith("models/") and not (ROOT / path / "config.json").exists():
                raise RuntimeError(f"Thiếu trọng số model: {path}")
        dense = DenseIndex(model_path, cache_dir=str(ROOT / "data" / "cache"),
                           device=self.device, batch_size=32,
                           query_prefix=f["query_prefix"], passage_prefix=f["passage_prefix"])
        if self._encoder_model is not None:
            dense._model = self._encoder_model
        reranker = (CrossEncoderReranker(reranker_path, device=self.device)
                    if flow == "text" else None)
        engine = SearchEngine(self.docs, dense=dense, reranker=reranker,
                              bm25_fold=f["bm25_fold"])
        engine.build_bm25()
        engine.build_dense(show_progress=True)
        # Corpus cache khong can model; nap mot lan khi query dau va chia se cho luong con lai.
        self._engines[flow] = engine
        return engine

    def warm_up(self) -> None:
        """Nap ca hai index va model; query ngan de chi phi khoi dong khong roi vao user dau."""
        with self._search_lock:
            first = next(iter(self.by_key))
            for flow in ("issue", "text"):
                engine = self._engine(flow)
                engine.dense.encode(["khởi động"], is_query=True)
                if self._encoder_model is None:
                    self._encoder_model = engine.dense._model
                # Luong text nap cross-encoder bang mot cap nho.
                if flow == "text" and engine.reranker is not None:
                    engine.reranker.score("khởi động", ["khởi động"])
            # Bao dam key sample co that; bien chi giup warm-up de doc de hieu.
            assert first in self.by_key

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        flow = str(payload.get("flow", "text")).strip().lower()
        if flow not in ("text", "issue"):
            raise PilotError("Luồng tìm kiếm không hợp lệ.")
        top_k = int(payload.get("top_k", 10))
        review_enabled = payload.get("feedback_enabled", True) is not False
        if not 1 <= top_k <= 20:
            raise PilotError("Số kết quả phải nằm trong khoảng 1–20.")

        query = str(payload.get("query", "")).strip()
        key = str(payload.get("key", "")).strip().upper()
        if flow == "text":
            if len(query) < 3:
                raise PilotError("Hãy nhập mô tả có ít nhất 3 ký tự.")
            if len(query) > 5000:
                raise PilotError("Mô tả dài tối đa 5.000 ký tự.")
            route = plan(query=query)
            qtext, exclude = query, None
            query_label = query
        else:
            if key not in self.by_key:
                raise PilotError(f"Không tìm thấy issue {key or '(trống)'} trong corpus.")
            route = plan(key=key)
            source_id = self.by_key[key]
            source = self.docs[source_id]
            qtext = source.get("text_dense") or source.get("summary", "")
            exclude = {source_id}
            query_label = key

        filters = {k: str((payload.get("filters") or {}).get(k, "")).strip()
                   for k in FILTER_FIELDS}
        allowed = filter_doc_ids(self.docs, filters)
        eligible = len(self.docs) if allowed is None else len(allowed - (exclude or set()))
        if eligible == 0:
            response = self._empty_response(flow, query_label, route, eligible)
            return self._log_search(response, flow, query, key, filters, top_k,
                                    review_enabled)

        config = SearchConfig(name=route["config"], top_k=top_k,
                              rerank_k=route["rerank_k"],
                              rerank_fusion=route["rerank_fusion"],
                              rerank_weight=route["rerank_weight"],
                              **CONFIGS[route["config"]])
        timing = self.timings[flow]
        started = time.perf_counter()
        with self._search_lock:
            engine = self._engine(flow)
            ranked, raw_scores = engine.search(qtext, config, exclude=exclude,
                                                allowed=allowed, timing=timing,
                                                return_scores=True)
            if self._encoder_model is None:
                self._encoder_model = engine.dense._model
        total = (time.perf_counter() - started) * 1000
        self.total_ms[flow].append(total)

        results = self._build_results(ranked, raw_scores, qtext)
        latest = {
            "embed_ms": round(timing.embed_ms[-1], 1) if timing.embed_ms else 0.0,
            "search_ms": round(timing.search_ms[-1], 1) if timing.search_ms else 0.0,
            "rerank_ms": round(timing.rerank_ms[-1], 1) if config.rerank and timing.rerank_ms else 0.0,
            "total_ms": round(total, 1),
        }
        response = {
            "flow": flow, "query_label": query_label, "eligible_docs": eligible,
            "config": route["config"], "model": route["model"], "measured": True,
            "rerank_k": config.rerank_k if config.rerank else 0,
            "latency": latest, "results": results,
            "score_note": "Điểm tương đối trong danh sách hiện tại; không phải xác suất.",
        }
        return self._log_search(response, flow, query, key, filters, top_k,
                                review_enabled)

    def _build_results(self, ranked: Any, raw_scores: Any, qtext: str) -> list[dict[str, Any]]:
        """Chuyen ket qua thô cua engine thanh schema chung cho cac giao dien."""
        scores = np.asarray(raw_scores, dtype=float)
        if len(scores) <= 1 or float(scores.max() - scores.min()) < 1e-12:
            relative = np.full(len(scores), 100.0)
        else:
            relative = 100.0 * (scores - scores.min()) / (scores.max() - scores.min())
        results = []
        for rank, (doc_id, raw, rel) in enumerate(zip(ranked, scores, relative), 1):
            d = self.docs[int(doc_id)]
            res = resolution.extract(d.get("text_comments", ""))
            results.append({
                "rank": rank,
                "key": d.get("key", ""),
                "summary": d.get("summary", ""),
                "snippet": matched_snippet(d, qtext),
                "relative_score": round(float(rel), 1),
                "ranking_score": round(float(raw), 6),
                "status": d.get("status", ""),
                "resolution_status": d.get("resolution", ""),
                "resolution": _short(res, 600),
                "priority": d.get("priority", ""),
                "project": _project(d),
                "components": _service_components(d),
                "created": d.get("created", ""),
                "updated": d.get("updated", ""),
                "url": d.get("url", ""),
            })
        return results

    def compare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """So sanh ba che do retrieval cua luong mo ta, khong ghi telemetry pilot."""
        query = str(payload.get("query", "")).strip()
        if len(query) < 3:
            raise PilotError("Hãy nhập mô tả có ít nhất 3 ký tự.")
        if len(query) > 5000:
            raise PilotError("Mô tả dài tối đa 5.000 ký tự.")
        top_k = int(payload.get("top_k", 10))
        if not 1 <= top_k <= 20:
            raise PilotError("Số kết quả phải nằm trong khoảng 1–20.")

        results: dict[str, list[dict[str, Any]]] = {}
        latency: dict[str, dict[str, float]] = {}
        with self._search_lock:
            engine = self._engine("text")
            for name in ("keyword", "semantic", "hybrid"):
                config = SearchConfig(name=name, top_k=top_k, **CONFIGS[name])
                timing = Timing()
                started = time.perf_counter()
                ranked, scores = engine.search(query, config, timing=timing,
                                               return_scores=True)
                total = (time.perf_counter() - started) * 1000
                results[name] = self._build_results(ranked, scores, query)
                latency[name] = {
                    "embed_ms": round(timing.embed_ms[-1], 1) if timing.embed_ms else 0.0,
                    "search_ms": round(timing.search_ms[-1], 1) if timing.search_ms else 0.0,
                    "rerank_ms": 0.0,
                    "total_ms": round(total, 1),
                }
            if self._encoder_model is None:
                self._encoder_model = engine.dense._model
        return {"query": query, "results": results, "latency": latency}

    def _log_search(self, response: dict[str, Any], flow: str, query: str,
                    source_key: str, filters: dict[str, str], top_k: int,
                    review_enabled: bool = True) -> dict[str, Any]:
        """Ghi snapshot da hien thi; query duoc scrub nhung ranking giu nguyen."""
        search_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._search_log_lock:
            self._search_count += 1
            # 1/3 luot duoc chi dinh cham day du top-5, ke ca qua restart.
            if not review_enabled:
                # Nguoi dung tat danh gia: van log luot tim, khong giao cham top-5.
                review_mode = "disabled"
            elif self._search_count % 3 == 1:
                review_mode = "full_top5"
            else:
                review_mode = "standard"
            event = {
                "schema_version": 2, "created_at": now, "search_id": search_id,
                "sequence": self._search_count, "flow": flow,
                "query": scrub(query)[0][:5000], "source_key": source_key,
                "filters": filters, "requested_top_k": top_k,
                "eligible_docs": response["eligible_docs"],
                "config": response["config"], "model": response["model"],
                "rerank_k": response.get("rerank_k", 0),
                "review_mode": review_mode,
                "results": [{"rank": r["rank"], "key": r["key"],
                             "ranking_score": r.get("ranking_score")}
                            for r in response.get("results", [])],
                "latency": response["latency"],
            }
            self.search_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.search_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._search_records[search_id] = event
        response.update({"search_id": search_id, "review_mode": review_mode,
                         "review_required_ranks": ([1, 2, 3, 4, 5]
                                                   if review_mode == "full_top5" else [])})
        return response

    @staticmethod
    def _empty_response(flow: str, query_label: str, route: dict, eligible: int) -> dict:
        return {
            "flow": flow, "query_label": query_label, "eligible_docs": eligible,
            "config": route["config"], "model": route["model"], "measured": True,
            "latency": {"embed_ms": 0, "search_ms": 0, "rerank_ms": 0, "total_ms": 0},
            "results": [],
            "score_note": "Không có issue nào thỏa toàn bộ bộ lọc.",
        }

    def telemetry(self) -> dict[str, Any]:
        out = {"started_at": self.started_at, "flows": {}}
        for flow, timing in self.timings.items():
            totals = self.total_ms[flow]
            summary = timing.summary()
            summary.update({
                "total_p50": round(float(np.percentile(totals, 50)), 1) if totals else 0.0,
                "total_p95": round(float(np.percentile(totals, 95)), 1) if totals else 0.0,
                "requests": len(totals),
            })
            out["flows"][flow] = summary
        return out

    def record_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        search_id = str(payload.get("search_id", "")).strip()
        search = self._search_records.get(search_id)
        if not search:
            raise PilotError("search_id không tồn tại hoặc đã hết phiên.")
        result_key = str(payload.get("result_key", "")).strip().upper()
        verdict = str(payload.get("verdict", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip().lower()
        if verdict not in ("related", "not_related", "no_answer", "cleared"):
            raise PilotError("Phản hồi không hợp lệ.")
        shown = {str(r["key"]).upper(): int(r["rank"]) for r in search["results"]}
        if verdict == "no_answer" and result_key:
            raise PilotError("Phản hồi không có đáp án không nhận result_key.")
        if verdict == "no_answer" or (verdict == "cleared" and not result_key):
            # "cleared" khong kem result_key = bo danh dau "khong co dap an".
            rank = 0
        else:
            if result_key not in shown:
                raise PilotError("Issue không thuộc kết quả của lượt tìm kiếm này.")
            rank = shown[result_key]
        if verdict == "not_related" and reason not in FAILURE_REASONS:
            raise PilotError("Hãy chọn nguyên nhân kết quả không liên quan.")
        if verdict != "not_related":
            reason = ""
        event = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "search_id": search_id,
            "flow": search["flow"], "query": search["query"],
            "source_key": search["source_key"], "filters": search["filters"],
            "review_mode": search["review_mode"],
            "result_key": result_key,
            "rank": rank, "verdict": verdict, "reason": reason,
            # Ghi chu la van ban tu do: scrub nhu query de khong luu PII/secret.
            "note": scrub(str(payload.get("note", ""))[:500])[0],
            "jira_writeback": "pending",
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._feedback_lock:
            with self.feedback_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"saved": True, "search_id": search_id,
                "jira_writeback": "pending"}
