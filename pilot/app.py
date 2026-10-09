# -*- coding: utf-8 -*-
"""HTTP server nhe cho pilot, khong can them web framework.

Chay tu thu muc repo:
    python -m pilot.app
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pilot.service import (DEFAULT_CORPUS, DEFAULT_SEARCH_LOG, PilotError,
                           PilotSearchService, ROOT)


STATIC = Path(__file__).resolve().parent / "static"
STATIC_ROUTES = {
    "/": STATIC / "index.html",
    "/index.html": STATIC / "index.html",
    "/app.css": STATIC / "app.css",
    "/app.js": STATIC / "app.js",
    "/compare": STATIC / "compare.html",
    "/compare.html": STATIC / "compare.html",
    "/compare.js": STATIC / "compare.js",
}


class PilotHandler(BaseHTTPRequestHandler):
    server_version = "SemanticIssuePilot/1.0"

    @property
    def service(self) -> PilotSearchService:
        return self.server.service  # type: ignore[attr-defined]

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self'; script-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
                         "frame-ancestors 'none'")
        self.end_headers()

    def _json(self, data: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": "Không tìm thấy tài nguyên."}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self._headers(HTTPStatus.OK, content_type, len(body))
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise PilotError("Content-Length không hợp lệ.") from exc
        if length <= 0 or length > 1_000_000:
            raise PilotError("Request body phải từ 1 byte đến 1 MB.")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PilotError("Request body phải là JSON UTF-8 hợp lệ.") from exc
        if not isinstance(value, dict):
            raise PilotError("Request body phải là một JSON object.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path in STATIC_ROUTES:
                self._file(STATIC_ROUTES[parsed.path])
            elif parsed.path == "/api/health":
                self._json({"ok": True, "corpus_size": len(self.service.docs),
                            "loaded_flows": sorted(self.service._engines)})
            elif parsed.path == "/api/meta":
                self._json(self.service.metadata())
            elif parsed.path == "/api/stats":
                self._json(self.service.telemetry())
            elif parsed.path == "/api/issues":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["8"])[0])
                self._json({"items": self.service.issue_suggestions(query, limit)})
            else:
                self._json({"error": "Không tìm thấy endpoint."}, HTTPStatus.NOT_FOUND)
        except (PilotError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self._json({"error": "Lỗi nội bộ. Xem log server để biết chi tiết."},
                       HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/search":
                self._json(self.service.search(self._body()))
            elif parsed.path == "/api/compare":
                self._json(self.service.compare(self._body()))
            elif parsed.path == "/api/feedback":
                self._json(self.service.record_feedback(self._body()), HTTPStatus.CREATED)
            else:
                self._json({"error": "Không tìm thấy endpoint."}, HTTPStatus.NOT_FOUND)
        except (PilotError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self._json({"error": "Lỗi nội bộ. Xem log server để biết chi tiết."},
                       HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: object) -> None:
        # Chi log method/path/status cua BaseHTTPRequestHandler, khong log request body.
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


class PilotServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: PilotSearchService):
        super().__init__(address, PilotHandler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Issue Search pilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--feedback", default=str(ROOT / "data" / "pilot" / "feedback.jsonl"))
    parser.add_argument("--search-log", default=str(DEFAULT_SEARCH_LOG))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lazy", action="store_true",
                        help="khong warm-up model; request dau se cham")
    args = parser.parse_args()

    missing = [name for name in ("numpy", "torch", "sentence_transformers")
               if importlib.util.find_spec(name) is None]
    if missing:
        packages = ", ".join(name.replace("_", "-") for name in missing)
        raise SystemExit(
            f"Môi trường Python đang thiếu: {packages}. "
            "Hãy dùng môi trường đã chạy pipeline hoặc cài pilot/requirements.txt.")

    # Giu model path tu search.py o dang relative de trung khoa cache da co.
    os.chdir(ROOT)
    service = PilotSearchService(args.corpus, args.feedback, args.device, args.search_log)
    if not args.lazy:
        print(f"Đang nạp 2 index và model trên {args.device}…", flush=True)
        service.warm_up()
    server = PilotServer((args.host, args.port), service)
    print(f"Pilot sẵn sàng: http://{args.host}:{args.port} ({len(service.docs):,} issues)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng pilot.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
