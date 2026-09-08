"""Minimal unauthenticated readiness surface used while a pod boots.

The endpoint deliberately exposes only bounded phase information.  It starts
before model provisioning so an operator can distinguish a slow first boot
from an exited container without giving the bootstrap access to the job API.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[^\s]+"),
    re.compile(r"\bhf_[A-Za-z0-9]+\b"),
    re.compile(r"\b[A-Fa-f0-9]{64,}\b"),
    re.compile(r"(?i)([?&](?:token|access_token|signature|x-amz-signature|key)=)[^&\s]+"),
    re.compile(r"(?i)(https://)[^/@\s]+:[^/@\s]+@"),
)


def _safe_detail(value: str) -> str:
    detail = value.replace("\r", " ").replace("\n", " ").strip()
    for pattern in _SECRET_PATTERNS:
        detail = pattern.sub(
            lambda match: f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]", detail
        )
    return detail[-500:]


@dataclass
class BootstrapStatus:
    stage: str = "starting"
    error_type: str | None = None
    error_detail: str | None = None
    _started_at: float = field(default_factory=monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
            self.error_type = None
            self.error_detail = None

    def fail(self, error: Exception) -> None:
        with self._lock:
            self.stage = "failed"
            self.error_type = type(error).__name__
            self.error_detail = _safe_detail(str(error))

    def payload(self) -> dict[str, Any]:
        with self._lock:
            value: dict[str, Any] = {
                "status": "failed" if self.stage == "failed" else "provisioning",
                "stage": self.stage,
                "ready": False,
                "elapsed_ms": round((monotonic() - self._started_at) * 1_000),
            }
            if self.error_type is not None:
                value["error_type"] = self.error_type
            if self.error_detail:
                value["error_detail"] = self.error_detail
            return value


class _BootstrapRequestHandler(BaseHTTPRequestHandler):
    server: _BootstrapHttpServer

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if self.path not in {"/", "/healthz", "/readyz"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = json.dumps(
            self.server.bootstrap_status.payload(), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        # RunPod probes the root path while the image is provisioning. Root is
        # a liveness endpoint; /healthz and /readyz are strict readiness
        # endpoints and stay unavailable until the worker starts.
        self.send_response(HTTPStatus.OK if self.path == "/" else HTTPStatus.SERVICE_UNAVAILABLE)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"bootstrap-http {self.address_string()} {format % args}", flush=True)


class _BootstrapHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, bootstrap_status: BootstrapStatus) -> None:
        self.bootstrap_status = bootstrap_status
        super().__init__(address, _BootstrapRequestHandler)


@dataclass
class BootstrapStatusServer:
    server: _BootstrapHttpServer
    thread: threading.Thread

    @property
    def status(self) -> BootstrapStatus:
        return self.server.bootstrap_status

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_bootstrap_status_server(port: int) -> BootstrapStatusServer:
    status = BootstrapStatus()
    server = _BootstrapHttpServer(("0.0.0.0", port), bootstrap_status=status)  # noqa: S104
    thread = threading.Thread(
        target=server.serve_forever,
        name="firewarning-bootstrap-health",
        daemon=True,
    )
    thread.start()
    return BootstrapStatusServer(server=server, thread=thread)
