from __future__ import annotations

import base64
import gzip
import hashlib
import http.client
import json
import threading
from datetime import UTC, datetime

import pytest

from fireviewer_contracts.client.media_fetcher import MediaFetchError
from fireviewer_orchestrator.pod_validation import (
    BundleMediaFetcher,
    PodValidationError,
    ValidationHTTPServer,
    ValidationRequestHandler,
    ValidationState,
    build_validation_report,
    decode_asset_bundle,
    decode_payload,
    run_validation,
    validation_asset_names,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "batch_id": "operational-reference-a-2026-07-08-evaluation",
        "batch_type": "satellite_media",
        "priority": "scheduled",
        "items": [
            {
                "input_id": "nasa-1",
                "media_type": "satellite_image",
                "working_file_url": "https://wvs.earthdata.nasa.gov/image.jpg",
            }
        ],
    }


def test_validation_payload_round_trip() -> None:
    raw = json.dumps(_payload()).encode()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()

    decoded = decode_payload(encoded)

    assert decoded["batch_id"] == "operational-reference-a-2026-07-08-evaluation"
    assert len(decoded["items"]) == 1


def _asset_bundle(value: bytes, *, digest: str | None = None) -> str:
    bundle = {
        "schema_version": 1,
        "assets": {
            "photo.jpg": {
                "sha256": digest or hashlib.sha256(value).hexdigest(),
                "data_b64": base64.b64encode(value).decode(),
            }
        },
    }
    return base64.b64encode(gzip.compress(json.dumps(bundle).encode(), mtime=0)).decode()


def test_validation_asset_bundle_round_trip() -> None:
    assert decode_asset_bundle(_asset_bundle(b"private-photo")) == {"photo.jpg": b"private-photo"}


def test_validation_asset_bundle_rejects_digest_mismatch() -> None:
    with pytest.raises(PodValidationError, match="digest mismatch"):
        decode_asset_bundle(_asset_bundle(b"private-photo", digest="0" * 64))


def test_bundle_fetcher_serves_only_exact_validation_https_url() -> None:
    fetcher = BundleMediaFetcher({"photo.jpg": b"private-photo"}, max_bytes=1024)

    with fetcher.download("https://validation-assets.internal/photo.jpg") as path:
        assert path.read_bytes() == b"private-photo"
    assert not path.exists()

    with (
        pytest.raises(MediaFetchError, match="bundled HTTPS boundary"),
        fetcher.download("https://example.org/photo.jpg"),
    ):
        pass


def test_authenticated_asset_upload_is_exact_and_one_shot(monkeypatch) -> None:
    import fireviewer_orchestrator.pod_validation as pod_validation

    payload = {
        "items": [
            {
                "working_file_url": "https://validation-assets.internal/photo.jpg",
            }
        ]
    }
    assert validation_asset_names(payload) == {"photo.jpg"}
    completed = threading.Event()

    def fake_worker(received_payload, state, assets):
        assert received_payload == payload
        assert assets == {"photo.jpg": b"private-photo"}
        state.update("completed", report={"pipeline_passed": True})
        completed.set()

    monkeypatch.setattr(pod_validation, "_validation_worker", fake_worker)
    server = ValidationHTTPServer(("127.0.0.1", 0), ValidationRequestHandler)
    server.validation_token = "t" * 32
    server.validation_state = ValidationState(initial_status="awaiting_assets")
    server.validation_payload = payload
    server.validation_worker_lock = threading.Lock()
    server.validation_worker_started = False
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request("POST", "/assets", body=_asset_bundle(b"private-photo"))
        assert connection.getresponse().status == 401
        connection.close()

        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request(
            "POST",
            "/assets",
            body=_asset_bundle(b"private-photo"),
            headers={"Authorization": f"Bearer {server.validation_token}"},
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["asset_count"] == 1
        connection.close()
        assert completed.wait(timeout=5)

        connection = http.client.HTTPConnection(*server.server_address, timeout=5)
        connection.request(
            "POST",
            "/assets",
            body=_asset_bundle(b"private-photo"),
            headers={"Authorization": f"Bearer {server.validation_token}"},
        )
        assert connection.getresponse().status == 409
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def test_image_pipeline_report_accepts_expected_detector_and_asr_skips() -> None:
    moment = datetime.now(UTC)
    output = {
        "items": [{"input_id": "nasa-1"}],
        "model_runs": [
            {"model_role": "asr", "status": "skipped"},
            {"model_role": "fire_detection", "status": "skipped"},
            {"model_role": "visual_grounding", "status": "succeeded"},
            {"model_role": "multimodal_extraction", "status": "succeeded"},
        ],
    }

    report = build_validation_report(
        _payload(),
        output,
        {"gpu_name": "NVIDIA RTX A4000"},
        started_at=moment,
        finished_at=moment,
    )

    assert report["pipeline_passed"] is True
    assert report["deployment_ready"] is False
    assert report["training_membership"] is False


def test_validation_runs_handler_boundary_with_injected_runtime() -> None:
    received = []

    def handler(job):
        received.append(job)
        return {
            "items": [{"input_id": "nasa-1"}],
            "model_runs": [
                {"model_role": "visual_grounding", "status": "succeeded"},
                {"model_role": "multimodal_extraction", "status": "succeeded"},
            ],
        }

    report = run_validation(
        _payload(),
        job_handler=handler,
        probe=lambda: {"gpu_name": "NVIDIA RTX A4000"},
    )

    assert received[0]["input"]["batch_id"] == _payload()["batch_id"]
    assert report["pipeline_passed"] is True
