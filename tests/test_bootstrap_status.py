from __future__ import annotations

import json
from http.client import HTTPConnection

from fireviewer_orchestrator.bootstrap_status import start_bootstrap_status_server


def _get_json(port: int, path: str = "/healthz") -> tuple[int, dict[str, object]]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_bootstrap_health_reports_progress_before_worker_is_ready() -> None:
    status_server = start_bootstrap_status_server(0)
    try:
        status_server.status.update("provisioning_models")

        status, payload = _get_json(status_server.port)

        assert status == 503
        assert payload["status"] == "provisioning"
        assert payload["stage"] == "provisioning_models"
        assert payload["ready"] is False
        assert isinstance(payload["elapsed_ms"], int)
    finally:
        status_server.close()


def test_bootstrap_health_redacts_failure_secrets() -> None:
    status_server = start_bootstrap_status_server(0)
    try:
        status_server.status.fail(
            RuntimeError(
                "x" * 600
                + " Bearer private-token hf_private "
                + "a" * 96
                + " https://user:password@example.test/file?token=query-token download failed"
            )
        )

        status, payload = _get_json(status_server.port)

        assert status == 503
        assert payload["status"] == "failed"
        assert payload["stage"] == "failed"
        assert payload["error_type"] == "RuntimeError"
        detail = str(payload["error_detail"])
        assert "private-token" not in detail
        assert "hf_private" not in detail
        assert "a" * 96 not in detail
        assert "password" not in detail
        assert "query-token" not in detail
        assert detail.count("[redacted]") == 5
    finally:
        status_server.close()


def test_bootstrap_root_is_liveness_but_not_readiness() -> None:
    status_server = start_bootstrap_status_server(0)
    try:
        status_server.status.update("provisioning_models")

        root_status, root_payload = _get_json(status_server.port, "/")
        health_status, health_payload = _get_json(status_server.port, "/healthz")
        ready_status, ready_payload = _get_json(status_server.port, "/readyz")

        assert root_status == 200
        assert root_payload["ready"] is False
        assert root_payload["stage"] == "provisioning_models"
        assert health_status == 503
        assert health_payload["ready"] is False
        assert ready_status == 503
        assert ready_payload["ready"] is False
    finally:
        status_server.close()
