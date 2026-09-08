"""Authenticated direct HTTP queue for the persistent RunPod staging pod."""

from __future__ import annotations

import hmac
import json
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic
from typing import Any

from fireviewer_orchestrator.handler import handle_job

MAX_REQUEST_BYTES = 8 * 1024 * 1024


@dataclass
class PodJob:
    job_id: str
    request: dict[str, Any]
    status: str = "IN_QUEUE"
    output: dict[str, Any] | None = None
    execution_time_ms: int | None = None
    delay_time_ms: int | None = None
    queued_at: float = field(default_factory=monotonic)
    started_at: float | None = None

    def response(self) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.job_id, "status": self.status}
        if self.output is not None:
            value["output"] = self.output
        if self.execution_time_ms is not None:
            value["executionTime"] = self.execution_time_ms
        if self.delay_time_ms is not None:
            value["delayTime"] = self.delay_time_ms
        return value


class PodJobQueue:
    """One FIFO consumer keeps GPU work sequential on a persistent validation pod."""

    def __init__(self) -> None:
        self._jobs: dict[str, PodJob] = {}
        self._pending: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

    def submit(self, request: dict[str, Any]) -> PodJob:
        job = PodJob(job_id=f"pod-{uuid.uuid4().hex}", request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self._pending.put(job.job_id)
        return job

    def get(self, job_id: str) -> PodJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> PodJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status == "IN_QUEUE":
                job.status = "CANCELLED"
            return job

    def _consume(self) -> None:
        while True:
            job_id = self._pending.get()
            try:
                with self._lock:
                    job = self._jobs[job_id]
                    if job.status == "CANCELLED":
                        continue
                    job.status = "IN_PROGRESS"
                    job.started_at = monotonic()
                    job.delay_time_ms = round((job.started_at - job.queued_at) * 1_000)
                    request = job.request
                started_at = monotonic()
                output = handle_job(request)
                execution_time_ms = round((monotonic() - started_at) * 1_000)
                with self._lock:
                    job.output = output
                    job.execution_time_ms = execution_time_ms
                    job.status = "COMPLETED"
            except Exception as exc:  # pragma: no cover - last-resort runtime boundary
                with self._lock:
                    job.output = {
                        "schema_version": "1.0",
                        "batch_id": "INVALID",
                        "status": "failed",
                        "retryable": False,
                        "model_runs": [],
                        "items": [],
                        "validation_errors": [f"pod:{type(exc).__name__}:{exc}"],
                        "boot_ms": 0,
                    }
                    job.status = "FAILED"
            finally:
                self._pending.task_done()


class PodRequestHandler(BaseHTTPRequestHandler):
    server: PodHttpServer

    def _write_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.auth_token}"
        return hmac.compare_digest(supplied, expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def _read_object(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdecimal():
            self._write_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return None
        length = int(raw_length)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "payload_too_large"})
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return None
        if not isinstance(value, dict):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "object_required"})
            return None
        return value

    def do_GET(self) -> None:
        if self.path in {"/", "/healthz", "/readyz"}:
            self._write_json(HTTPStatus.OK, {"status": "ready", "mode": "runpod-pod"})
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if not self._require_auth():
            return
        prefix = "/v1/jobs/"
        if not self.path.startswith(prefix):
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        job = self.server.jobs.get(self.path.removeprefix(prefix))
        if job is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
            return
        self._write_json(HTTPStatus.OK, job.response())

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        if self.path == "/v1/jobs":
            request = self._read_object()
            if request is None:
                return
            if not isinstance(request.get("input"), dict):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "input_object_required"})
                return
            submitted_job = self.server.jobs.submit(request)
            self._write_json(HTTPStatus.ACCEPTED, submitted_job.response())
            return
        suffix = "/cancel"
        prefix = "/v1/jobs/"
        if self.path.startswith(prefix) and self.path.endswith(suffix):
            job_id = self.path[len(prefix) : -len(suffix)]
            cancelled_job = self.server.jobs.cancel(job_id)
            if cancelled_job is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
                return
            self._write_json(HTTPStatus.OK, cancelled_job.response())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"pod-http {self.address_string()} {format % args}", flush=True)


class PodHttpServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], *, auth_token: str) -> None:
        self.auth_token = auth_token
        self.jobs = PodJobQueue()
        super().__init__(address, PodRequestHandler)


def main() -> None:
    auth_token = os.getenv("FW_POD_AUTH_TOKEN", "")
    if len(auth_token) < 32:
        raise SystemExit("FW_POD_AUTH_TOKEN must contain at least 32 characters")
    port = int(os.getenv("FW_POD_PORT", "8000"))
    server = PodHttpServer(("0.0.0.0", port), auth_token=auth_token)  # noqa: S104
    print(f"FireWarning persistent pod queue listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
