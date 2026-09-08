from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import AnyHttpUrl, Field, SecretStr, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.contracts import EventEvidenceV1, EvidenceMedia
from fireviewer_geolocation.mvp.localization.durable_terrain import AzureBackendTerrainResolver
from fireviewer_geolocation.mvp.localization.geographic_endpoint import (
    DurableGeographicHypothesisService,
)
from fireviewer_orchestrator.mvp.orchestration.point_bundle_pipeline import (
    GeographicPointBundlePipeline,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    AzureBackendEventEvidenceAdapter,
    AzureBackendEventEvidenceConfig,
    BackendVisualEvidencePublisher,
)
from fireviewer_vision_runtime.mvp.vision.event_vision import EventVisionRun, EventVisionRunner
from fireviewer_vision_runtime.mvp.vision.yolo import (
    MODEL_ID,
    MODEL_REVISION,
    HuggingFaceYoloModelLoader,
    YoloCpuConfig,
    YoloCpuVisionProvider,
)

MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class EvidenceAssetLocation(StrictModel):
    media_id: SafeIdentifierV2
    working_file_url: AnyHttpUrl


class YoloEventRequest(StrictModel):
    evidence: EventEvidenceV1
    assets: tuple[EvidenceAssetLocation, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_assets(self) -> YoloEventRequest:
        asset_ids = tuple(item.media_id for item in self.assets)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset locations must be uniquely keyed by media_id")
        media_by_id = {item.media_id: item for item in self.evidence.media}
        if any(media_id not in media_by_id for media_id in asset_ids):
            raise ValueError("asset location references unknown EventEvidence media")
        eligible_ids = {
            item.media_id for item in self.evidence.media if item.kind in {"photo", "keyframe"}
        }
        if eligible_ids - set(asset_ids):
            raise ValueError("every photo and keyframe requires a working_file_url")
        return self


class YoloBackendEventRequest(StrictModel):
    candidate_id: SafeIdentifierV2


class YoloTransientImageRequest(StrictModel):
    media_id: SafeIdentifierV2
    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(min_length=4, max_length=7 * 1_024 * 1_024, repr=False)

    def image(self) -> Image.Image:
        try:
            content = base64.b64decode(self.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("transient YOLO image base64 is invalid") from exc
        if not content or len(content) > 5 * 1_024 * 1_024:
            raise ValueError("transient YOLO image exceeds its byte limit")
        if sha256(content).hexdigest() != self.content_sha256:
            raise ValueError("transient YOLO image digest mismatch")
        try:
            image = Image.open(BytesIO(content))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("transient YOLO payload is not an image") from exc
        return image


class _StaticImageLoader:
    def __init__(self, *, media_id: str, image: Image.Image) -> None:
        self._media_id = media_id
        self._image = image

    def load(self, media: EvidenceMedia) -> object:
        if media.media_id != self._media_id:
            raise ValueError("transient YOLO media identity mismatch")
        return self._image


class EvidenceDownloadError(RuntimeError):
    pass


class HttpEvidenceImageLoader:
    def __init__(
        self,
        *,
        locations: Mapping[str, str],
        allowed_hosts: frozenset[str],
        bearer_tokens_by_origin: Mapping[str, str] | None = None,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        client: httpx.Client | None = None,
    ) -> None:
        if not allowed_hosts:
            raise ValueError("at least one evidence host must be allowlisted")
        self.locations = dict(locations)
        self.allowed_hosts = allowed_hosts
        self.bearer_tokens_by_origin = {
            key.rstrip("/").casefold(): value
            for key, value in (bearer_tokens_by_origin or {}).items()
        }
        self.max_download_bytes = max_download_bytes
        self._client = client

    def load(self, media: EvidenceMedia) -> object:
        raw_url = self.locations.get(media.media_id)
        if raw_url is None:
            raise EvidenceDownloadError("missing working URL for evidence media")
        parsed = urlsplit(raw_url)
        hostname = (parsed.hostname or "").casefold()
        if (
            parsed.scheme != "https"
            or hostname not in self.allowed_hosts
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise EvidenceDownloadError("evidence URL is outside the HTTPS host allowlist")
        origin = f"https://{parsed.netloc.casefold()}"
        request_headers = {
            "Accept": "image/*",
            "User-Agent": "FireViewer-YOLO/1.0 (+https://fireviewer.org)",
        }
        bearer_token = self.bearer_tokens_by_origin.get(origin)
        if bearer_token is not None:
            request_headers["Authorization"] = f"Bearer {bearer_token}"
        owned_client = self._client is None
        client = self._client or httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, connect=10.0),
            trust_env=False,
        )
        try:
            with client.stream(
                "GET",
                raw_url,
                headers=request_headers,
            ) as response:
                if response.is_redirect:
                    raise EvidenceDownloadError("evidence URL redirects are not allowed")
                response.raise_for_status()
                declared_length = response.headers.get("Content-Length")
                if (
                    declared_length
                    and declared_length.isdecimal()
                    and int(declared_length) > self.max_download_bytes
                ):
                    raise EvidenceDownloadError("evidence image exceeds the download limit")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > self.max_download_bytes:
                        raise EvidenceDownloadError("evidence image exceeds the download limit")
        except httpx.HTTPError as exc:
            raise EvidenceDownloadError("evidence image download failed") from exc
        finally:
            if owned_client:
                client.close()
        if sha256(payload).hexdigest() != media.sha256:
            raise EvidenceDownloadError("evidence image SHA-256 differs from EventEvidence")
        try:
            image = Image.open(BytesIO(payload))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise EvidenceDownloadError("evidence payload is not a supported image") from exc
        return image


@dataclass(frozen=True, slots=True)
class YoloCpuServiceSettings:
    auth_token: str
    allowed_hosts: frozenset[str]
    model_cache: Path
    backend_base_url: str | None = None
    backend_auth_token: str | None = None
    port: int = 8000
    torch_threads: int = 4
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES

    @classmethod
    def from_environment(cls) -> YoloCpuServiceSettings:
        auth_token = os.getenv("FW_YOLO_AUTH_TOKEN", "")
        if len(auth_token) < 32:
            raise ValueError("FW_YOLO_AUTH_TOKEN must contain at least 32 characters")
        allowed_hosts = frozenset(
            item.strip().casefold()
            for item in os.getenv("FW_YOLO_ALLOWED_HOSTS", "").split(",")
            if item.strip()
        )
        if not allowed_hosts:
            raise ValueError("FW_YOLO_ALLOWED_HOSTS must contain at least one hostname")
        backend_base_url = os.getenv("FIREVIEWER_BACKEND_BASE_URL", "").rstrip("/") or None
        backend_auth_token = os.getenv("FIREVIEWER_BACKEND_TOKEN", "") or None
        if (backend_base_url is None) != (backend_auth_token is None):
            raise ValueError(
                "FIREVIEWER_BACKEND_BASE_URL and FIREVIEWER_BACKEND_TOKEN "
                "must be configured together"
            )
        if backend_base_url is not None:
            parsed_backend = urlsplit(backend_base_url)
            backend_host = (parsed_backend.hostname or "").casefold()
            if (
                parsed_backend.scheme != "https"
                or parsed_backend.username is not None
                or parsed_backend.password is not None
                or parsed_backend.query
                or parsed_backend.fragment
                or parsed_backend.path not in {"", "/"}
                or backend_host not in allowed_hosts
            ):
                raise ValueError("FIREVIEWER_BACKEND_BASE_URL must be an allowlisted HTTPS origin")
            if backend_auth_token is None or len(backend_auth_token) < 32:
                raise ValueError("FIREVIEWER_BACKEND_TOKEN must contain at least 32 characters")
        return cls(
            auth_token=auth_token,
            allowed_hosts=allowed_hosts,
            model_cache=Path(os.getenv("FW_YOLO_MODEL_CACHE", "/opt/huggingface")),
            backend_base_url=backend_base_url,
            backend_auth_token=backend_auth_token,
            port=int(os.getenv("PORT", os.getenv("FW_YOLO_PORT", "8000"))),
            torch_threads=int(os.getenv("FW_YOLO_TORCH_THREADS", "4")),
            max_download_bytes=int(
                os.getenv("FW_YOLO_MAX_DOWNLOAD_BYTES", str(DEFAULT_MAX_DOWNLOAD_BYTES))
            ),
        )


class YoloEventService:
    def __init__(
        self,
        *,
        settings: YoloCpuServiceSettings,
        model_loader: Callable[[], Any] | None = None,
        backend_adapter: AzureBackendEventEvidenceAdapter | None = None,
        visual_publisher: BackendVisualEvidencePublisher | None = None,
        geographic_service: DurableGeographicHypothesisService | None = None,
        point_bundle_pipeline: GeographicPointBundlePipeline | None = None,
    ) -> None:
        self.settings = settings
        self._model_loader = model_loader or HuggingFaceYoloModelLoader(
            cache_dir=settings.model_cache,
            local_files_only=True,
            torch_threads=settings.torch_threads,
        )
        self._runtime: Any | None = None
        self._lock = threading.Lock()
        self._backend_adapter = backend_adapter
        self._visual_publisher = visual_publisher
        self._geographic_service = geographic_service
        self._point_bundle_pipeline = point_bundle_pipeline
        if settings.backend_base_url is not None and settings.backend_auth_token is not None:
            backend_config = AzureBackendEventEvidenceConfig(
                base_url=settings.backend_base_url,
                bearer_token=SecretStr(settings.backend_auth_token),
                timeout_seconds=30,
            )
            self._backend_adapter = self._backend_adapter or AzureBackendEventEvidenceAdapter(
                backend_config
            )
            self._visual_publisher = self._visual_publisher or BackendVisualEvidencePublisher(
                backend_config
            )
            self._geographic_service = self._geographic_service or (
                DurableGeographicHypothesisService(
                    self._backend_adapter,
                    terrain_resolver=AzureBackendTerrainResolver(backend_config),
                )
            )
        if self._point_bundle_pipeline is None and self._geographic_service is not None:
            self._point_bundle_pipeline = GeographicPointBundlePipeline(
                self._geographic_service
            )

    @property
    def ready(self) -> bool:
        return self._runtime is not None

    def warmup(self) -> None:
        with self._lock:
            if self._runtime is None:
                self._runtime = self._model_loader()

    def analyze(self, request: YoloEventRequest) -> dict[str, Any]:
        run = self._run(request)
        return self._response(run)

    def _run(self, request: YoloEventRequest) -> EventVisionRun:
        bearer_tokens_by_origin: dict[str, str] = {}
        if self.settings.backend_base_url and self.settings.backend_auth_token:
            bearer_tokens_by_origin[self.settings.backend_base_url] = (
                self.settings.backend_auth_token
            )
        loader = HttpEvidenceImageLoader(
            locations={item.media_id: str(item.working_file_url) for item in request.assets},
            allowed_hosts=self.settings.allowed_hosts,
            bearer_tokens_by_origin=bearer_tokens_by_origin,
            max_download_bytes=self.settings.max_download_bytes,
        )
        with self._lock:
            if self._runtime is None:
                self._runtime = self._model_loader()
            provider = YoloCpuVisionProvider(
                image_loader=loader,
                model_loader=lambda: self._require_runtime(),
                config=YoloCpuConfig(torch_threads=self.settings.torch_threads),
            )
            run = EventVisionRunner(provider=provider).run(request.evidence)
        return run

    @staticmethod
    def _response(run: EventVisionRun) -> dict[str, Any]:
        return {
            "evidence": run.evidence.model_dump(mode="json", by_alias=True),
            "artifacts": [
                {
                    "result_reference": item.result_reference,
                    "result": item.result.model_dump(mode="json", by_alias=True),
                }
                for item in run.artifacts
            ],
        }

    def analyze_backend(self, request: YoloBackendEventRequest) -> dict[str, Any]:
        if self._backend_adapter is None or self._visual_publisher is None:
            raise RuntimeError("backend EventEvidence integration is not configured")
        durable = self._backend_adapter.read(request.candidate_id)
        eligible_media_ids = {
            item.media_id for item in durable.event.media if item.kind in {"photo", "keyframe"}
        }
        event_request = YoloEventRequest(
            evidence=durable.event,
            assets=tuple(
                EvidenceAssetLocation.model_validate(
                    {
                        "media_id": item.media_id,
                        "working_file_url": item.working_file_url,
                    }
                )
                for item in durable.media_locations
                if item.media_id in eligible_media_ids
            ),
        )
        before_candidates = durable.event.location_candidates
        run = self._run(event_request)
        if run.evidence.location_candidates != before_candidates:
            raise RuntimeError("YOLO visual analysis attempted to modify geographic candidates")
        artifacts = {item.result_reference: item.result for item in run.artifacts}
        observations = tuple(
            item for item in run.evidence.visual_observations if item.result_reference in artifacts
        )
        receipt = self._visual_publisher.publish(
            candidate_id=request.candidate_id,
            source_revision_sha256=durable.source_revision_sha256,
            observations=observations,
            artifacts=artifacts,
        )
        response = self._response(run)
        response["candidate_id"] = request.candidate_id
        response["persistence"] = receipt.model_dump(mode="json")
        response["geographic_output"] = {
            "location_candidates_before": len(before_candidates),
            "location_candidates_after": len(run.evidence.location_candidates),
            "localization_attempts_created": 0,
        }
        return response

    def analyze_transient(self, request: YoloTransientImageRequest) -> dict[str, Any]:
        """Analyze one in-memory public frame without retaining or geolocating it."""

        image = request.image()
        media = EvidenceMedia(
            media_id=request.media_id,
            source_id="SRC-TRANSIENT-PUBLIC",
            media_group_id="GROUP-TRANSIENT-PUBLIC",
            origin_id="ORIGIN-TRANSIENT-PUBLIC",
            kind="photo",
            sha256=request.content_sha256,
        )
        with self._lock:
            if self._runtime is None:
                self._runtime = self._model_loader()
            provider = YoloCpuVisionProvider(
                image_loader=_StaticImageLoader(media_id=media.media_id, image=image),
                model_loader=lambda: self._require_runtime(),
                config=YoloCpuConfig(torch_threads=self.settings.torch_threads),
            )
            result = provider.detect(media)
        return {
            "result": result.model_dump(mode="json", by_alias=True),
            "input_binary_stored": False,
            "geographic_output_created": False,
        }

    def locate_backend(self, request: YoloBackendEventRequest) -> dict[str, object]:
        if self._geographic_service is None:
            raise RuntimeError("backend geographic integration is not configured")
        return self._geographic_service.locate_payload(
            {"event_id": request.candidate_id}
        )

    def build_point_bundles(
        self,
        request: YoloBackendEventRequest,
    ) -> dict[str, object]:
        if self._point_bundle_pipeline is None:
            raise RuntimeError("backend point bundle integration is not configured")
        return self._point_bundle_pipeline.build_payload(
            request.candidate_id,
            generated_at=datetime.now(UTC),
        )

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("YOLO runtime is not loaded")
        return self._runtime


class YoloRequestHandler(BaseHTTPRequestHandler):
    server: YoloHttpServer

    def _write_json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return hmac.compare_digest(
            self.headers.get("Authorization", ""),
            f"Bearer {self.server.auth_token}",
        )

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "device": "cpu",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                },
            )
            return
        if self.path == "/readyz":
            status = HTTPStatus.OK if self.server.service.ready else HTTPStatus.SERVICE_UNAVAILABLE
            readiness = "ready" if self.server.service.ready else "loading"
            self._write_json(status, {"status": readiness})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path not in {
            "/v1/event-evidence/detect",
            "/v1/backend-event-evidence/detect",
            "/v1/backend-event-evidence/geographic-hypotheses",
            "/v1/backend-event-evidence/point-bundles",
            "/v1/transient-images/detect",
        }:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdecimal():
            self._write_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return
        length = int(raw_length)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "payload_too_large"})
            return
        try:
            raw_body = self.rfile.read(length)
            if self.path in {
                "/v1/backend-event-evidence/detect",
                "/v1/backend-event-evidence/geographic-hypotheses",
                "/v1/backend-event-evidence/point-bundles",
            }:
                backend_request = YoloBackendEventRequest.model_validate_json(raw_body)
            elif self.path == "/v1/transient-images/detect":
                transient_request = YoloTransientImageRequest.model_validate_json(raw_body)
            else:
                event_request = YoloEventRequest.model_validate_json(raw_body)
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_event_evidence_request"})
            return
        try:
            if self.path == "/v1/backend-event-evidence/detect":
                response = self.server.service.analyze_backend(backend_request)
            elif self.path == "/v1/transient-images/detect":
                response = self.server.service.analyze_transient(transient_request)
            elif self.path == "/v1/backend-event-evidence/geographic-hypotheses":
                response = self.server.service.locate_backend(backend_request)
            elif self.path == "/v1/backend-event-evidence/point-bundles":
                response = self.server.service.build_point_bundles(backend_request)
            else:
                response = self.server.service.analyze(event_request)
        except Exception as exc:  # pragma: no cover - last-resort process boundary
            operation = (
                "geographic hypotheses"
                if self.path
                in {
                    "/v1/backend-event-evidence/geographic-hypotheses",
                    "/v1/backend-event-evidence/point-bundles",
                }
                else "yolo-cpu inference"
            )
            error = (
                "geographic_hypotheses_failed"
                if self.path
                in {
                    "/v1/backend-event-evidence/geographic-hypotheses",
                    "/v1/backend-event-evidence/point-bundles",
                }
                else "inference_failed"
            )
            print(f"{operation} failed: {type(exc).__name__}", flush=True)
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": error})
            return
        self._write_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: object) -> None:
        print(f"yolo-http {self.address_string()} {format % args}", flush=True)


class YoloHttpServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        auth_token: str,
        service: YoloEventService,
    ) -> None:
        self.auth_token = auth_token
        self.service = service
        super().__init__(address, YoloRequestHandler)


def main() -> None:
    settings = YoloCpuServiceSettings.from_environment()
    service = YoloEventService(settings=settings)
    service.warmup()
    server = YoloHttpServer(
        ("0.0.0.0", settings.port),  # noqa: S104
        auth_token=settings.auth_token,
        service=service,
    )
    print(
        f"FireViewer YOLO CPU service ready on port {settings.port} "
        f"with {settings.torch_threads} torch threads",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()


__all__ = [
    "EvidenceAssetLocation",
    "EvidenceDownloadError",
    "HttpEvidenceImageLoader",
    "YoloBackendEventRequest",
    "YoloCpuServiceSettings",
    "YoloEventRequest",
    "YoloEventService",
    "YoloHttpServer",
    "YoloTransientImageRequest",
    "main",
]
