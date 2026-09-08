from __future__ import annotations

import base64
import threading
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from fireviewer_contracts.mvp.contracts import EventEvidenceV1, EvidenceMedia, EvidenceSource
from fireviewer_evidence_supervisor.mvp.supervision import (
    BackendEvidenceMediaLocation,
    BackendVisualEvidenceReceipt,
    DurableEventEvidence,
)
from fireviewer_orchestrator.mvp.vision.yolo_cpu_service import (
    EvidenceAssetLocation,
    EvidenceDownloadError,
    HttpEvidenceImageLoader,
    YoloBackendEventRequest,
    YoloCpuServiceSettings,
    YoloEventRequest,
    YoloEventService,
    YoloHttpServer,
    YoloTransientImageRequest,
)


def _image_bytes() -> bytes:
    target = BytesIO()
    Image.new("RGB", (8, 6), color=(220, 40, 20)).save(target, format="PNG")
    return target.getvalue()


def _evidence(payload: bytes) -> EventEvidenceV1:
    return EventEvidenceV1(
        event_id="EVENT-YOLO-SERVICE",
        sources=(
            EvidenceSource(
                source_id="SOURCE-1",
                origin_id="ORIGIN-1",
                publisher="FireViewer test",
                retrieved_at=datetime.now(UTC),
                source_type="witness",
                independence_weight=1,
            ),
        ),
        media=(
            EvidenceMedia(
                media_id="MEDIA-1",
                source_id="SOURCE-1",
                media_group_id="GROUP-1",
                origin_id="ORIGIN-1",
                kind="photo",
                sha256=sha256(payload).hexdigest(),
            ),
        ),
    )


def test_request_requires_a_working_url_for_each_eligible_media() -> None:
    payload = _image_bytes()
    evidence = _evidence(payload)
    evidence = EventEvidenceV1.model_validate(
        evidence.model_copy(
            update={
                "media": (
                    *evidence.media,
                    EvidenceMedia(
                        media_id="MEDIA-2",
                        source_id="SOURCE-1",
                        media_group_id="GROUP-2",
                        origin_id="ORIGIN-2",
                        kind="photo",
                        sha256=sha256(payload).hexdigest(),
                    ),
                )
            }
        )
    )
    with pytest.raises(ValidationError, match="every photo and keyframe"):
        YoloEventRequest(
            evidence=evidence,
            assets=(
                EvidenceAssetLocation(
                    media_id="MEDIA-1",
                    working_file_url="https://api.example.test/evidence",
                ),
            ),
        )


def test_http_loader_enforces_host_and_sha256() -> None:
    payload = _image_bytes()
    media = _evidence(payload).media[0]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"].startswith("FireViewer-YOLO/")
        assert request.headers["Authorization"] == "Bearer " + ("b" * 40)
        return httpx.Response(200, content=payload, headers={"Content-Type": "image/png"})

    loader = HttpEvidenceImageLoader(
        locations={media.media_id: "https://api.example.test/evidence"},
        allowed_hosts=frozenset({"api.example.test"}),
        bearer_tokens_by_origin={"https://api.example.test": "b" * 40},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    image = loader.load(media)
    assert isinstance(image, Image.Image)
    assert image.size == (8, 6)
    rejected = HttpEvidenceImageLoader(
        locations={media.media_id: "https://untrusted.example/evidence"},
        allowed_hosts=frozenset({"api.example.test"}),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(EvidenceDownloadError, match="allowlist"):
        rejected.load(media)


class _ListTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def float(self) -> _ListTensor:
        return self

    def cpu(self) -> _ListTensor:
        return self

    def tolist(self) -> list[object]:
        return self.values


class _SmokeModel:
    def predict(self, **kwargs: object) -> list[object]:
        assert kwargs["device"] == "cpu"
        boxes = type(
            "Boxes",
            (),
            {
                "xyxy": _ListTensor([[2.0, 1.0, 6.0, 5.0]]),
                "conf": _ListTensor([0.88]),
                "cls": _ListTensor([0]),
            },
        )()
        return [type("Result", (), {"names": {0: "smoke"}, "boxes": boxes})()]


class _BackendAdapter:
    def __init__(self, durable: DurableEventEvidence) -> None:
        self.durable = durable

    def read(self, event_id: str) -> DurableEventEvidence:
        assert event_id == self.durable.event.event_id
        return self.durable


class _VisualPublisher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def publish(self, **kwargs: object) -> BackendVisualEvidenceReceipt:
        self.calls.append(kwargs)
        return BackendVisualEvidenceReceipt(
            candidate_id=str(kwargs["candidate_id"]),
            observation_count=len(kwargs["observations"]),  # type: ignore[arg-type]
            replayed=False,
            source_revision_sha256="f" * 64,
        )


class _GeographicService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def locate_payload(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {
            "schema": "fireviewer.geographic-hypotheses.v1",
            "event_id": str(payload["event_id"]),
            "status": "abstained",
            "hypotheses": [],
            "geometry_mutation_allowed": False,
        }


class _PointBundlePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime]] = []

    def build_payload(
        self,
        event_id: str,
        *,
        generated_at: datetime,
    ) -> dict[str, object]:
        self.calls.append((event_id, generated_at))
        return {
            "schema": "fireviewer.point-evidence-bundle-batch.v1",
            "event_id": event_id,
            "bundles": [],
            "geometry_mutation_allowed": False,
        }


def test_backend_yolo_persists_visual_observation_without_geographic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _image_bytes()
    evidence = _evidence(payload)
    durable = DurableEventEvidence(
        event=evidence,
        media_locations=(
            BackendEvidenceMediaLocation(
                media_id="MEDIA-1",
                working_file_url=(
                    "https://api.example.test/api/v1/internal/event-evidence/"
                    "EVENT-YOLO-SERVICE/assets/MEDIA-1/content"
                ),
            ),
        ),
        vision_artifacts=(),
        upload_locations=(),
        prior_fire_states=(),
        geospatial_checks=(),
        geographic_references=(),
        source_revision_sha256="e" * 64,
    )
    publisher = _VisualPublisher()
    settings = YoloCpuServiceSettings(
        auth_token="a" * 40,
        allowed_hosts=frozenset({"api.example.test"}),
        model_cache=Path("model-cache"),
        backend_base_url="https://api.example.test",
        backend_auth_token="b" * 40,
    )
    monkeypatch.setattr(
        HttpEvidenceImageLoader,
        "load",
        lambda _self, _media: Image.new("RGB", (8, 6)),
    )
    service = YoloEventService(
        settings=settings,
        model_loader=_SmokeModel,
        backend_adapter=_BackendAdapter(durable),  # type: ignore[arg-type]
        visual_publisher=publisher,  # type: ignore[arg-type]
    )

    result = service.analyze_backend(
        YoloBackendEventRequest(candidate_id="EVENT-YOLO-SERVICE")
    )

    assert result["candidate_id"] == "EVENT-YOLO-SERVICE"
    assert result["evidence"]["visual_observations"][0]["observation_type"] == "detection"
    assert result["artifacts"][0]["result"]["status"] == "smoke"
    assert result["persistence"]["observation_count"] == 1
    assert result["geographic_output"] == {
        "location_candidates_before": 0,
        "location_candidates_after": 0,
        "localization_attempts_created": 0,
    }
    assert len(publisher.calls) == 1
    assert publisher.calls[0]["source_revision_sha256"] == "e" * 64


def test_transient_public_frame_is_detected_without_storage_or_geographic_output() -> None:
    payload = _image_bytes()
    service = YoloEventService(
        settings=YoloCpuServiceSettings(
            auth_token="a" * 40,
            allowed_hosts=frozenset({"api.example.test"}),
            model_cache=Path("model-cache"),
        ),
        model_loader=_SmokeModel,
    )

    result = service.analyze_transient(
        YoloTransientImageRequest(
            media_id="KEYFRAME-PUBLIC-0001",
            content_type="image/png",
            content_sha256=sha256(payload).hexdigest(),
            content_base64=base64.b64encode(payload).decode("ascii"),
        )
    )

    assert result["result"]["media_id"] == "KEYFRAME-PUBLIC-0001"
    assert result["result"]["status"] == "smoke"
    assert result["result"]["provider_run"]["cost_usd"] == 0
    assert result["input_binary_stored"] is False
    assert result["geographic_output_created"] is False


def test_transient_yolo_http_route_requires_auth_and_never_returns_input_bytes() -> None:
    payload = _image_bytes()
    token = "a" * 40
    service = YoloEventService(
        settings=YoloCpuServiceSettings(
            auth_token=token,
            allowed_hosts=frozenset({"api.example.test"}),
            model_cache=Path("model-cache"),
        ),
        model_loader=_SmokeModel,
    )
    server = YoloHttpServer(("127.0.0.1", 0), auth_token=token, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = {
        "media_id": "KEYFRAME-PUBLIC-0002",
        "content_type": "image/png",
        "content_sha256": sha256(payload).hexdigest(),
        "content_base64": base64.b64encode(payload).decode("ascii"),
    }
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/transient-images/detect"
        unauthorized = httpx.post(endpoint, json=request, timeout=5)
        response = httpx.post(
            endpoint,
            json=request,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "smoke"
    assert "content_base64" not in response.text


def test_service_settings_are_cpu_only_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FW_YOLO_AUTH_TOKEN", "x" * 32)
    monkeypatch.setenv("FW_YOLO_ALLOWED_HOSTS", "API.EXAMPLE.TEST")
    monkeypatch.setenv("FW_YOLO_MODEL_CACHE", str(Path("model-cache")))
    settings = YoloCpuServiceSettings.from_environment()
    assert settings.torch_threads == 4
    assert settings.allowed_hosts == frozenset({"api.example.test"})
    monkeypatch.delenv("FW_YOLO_ALLOWED_HOSTS")
    with pytest.raises(ValueError, match="FW_YOLO_ALLOWED_HOSTS"):
        YoloCpuServiceSettings.from_environment()


def test_backend_geographic_stage_is_a_separate_route_without_yolo_inference() -> None:
    settings = YoloCpuServiceSettings(
        auth_token="a" * 40,
        allowed_hosts=frozenset({"api.example.test"}),
        model_cache=Path("model-cache"),
    )
    geographic_service = _GeographicService()
    service = YoloEventService(
        settings=settings,
        model_loader=lambda: (_ for _ in ()).throw(AssertionError("YOLO must not run")),
        geographic_service=geographic_service,  # type: ignore[arg-type]
    )

    result = service.locate_backend(
        YoloBackendEventRequest(candidate_id="EVENT-YOLO-SERVICE")
    )

    assert result["status"] == "abstained"
    assert result["geometry_mutation_allowed"] is False
    assert geographic_service.calls == [{"event_id": "EVENT-YOLO-SERVICE"}]


def test_point_bundle_stage_is_separate_and_does_not_run_yolo() -> None:
    settings = YoloCpuServiceSettings(
        auth_token="a" * 40,
        allowed_hosts=frozenset({"api.example.test"}),
        model_cache=Path("model-cache"),
    )
    pipeline = _PointBundlePipeline()
    service = YoloEventService(
        settings=settings,
        model_loader=lambda: (_ for _ in ()).throw(AssertionError("YOLO must not run")),
        point_bundle_pipeline=pipeline,  # type: ignore[arg-type]
    )

    result = service.build_point_bundles(
        YoloBackendEventRequest(candidate_id="EVENT-YOLO-SERVICE")
    )

    assert result["schema"] == "fireviewer.point-evidence-bundle-batch.v1"
    assert result["geometry_mutation_allowed"] is False
    assert pipeline.calls[0][0] == "EVENT-YOLO-SERVICE"
    assert pipeline.calls[0][1].tzinfo is UTC
