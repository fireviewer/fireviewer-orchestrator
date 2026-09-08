from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator
from fireviewer_contracts.geo_gpu import ArtifactFile, EncodedPayload, GeoGpuRequest, GeoGpuResponse, GeoModelArtifactManifest

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel, WorkerInputV2
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_geolocation.mvp.localization.local_megaloc_bundle import (
    LocalMegaLocBundleManifest,
    LocalMegaLocModelLoader,
)
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocConfig, TorchMegaLocEncoder
from fireviewer_vision_runtime.prithvi_burned_area import PrithviBurnedAreaAdapter

MEGALOC_REVISION = "37bb43d65dd6388d1578052de5eb0bcdceb497e7"
PRITHVI_REVISION = "a3f2c410e45b8ac7417976614528a872f024d831"
REQUEST_SCHEMA: Final[Literal["fireviewer.geo-gpu-request.v1"]] = (
    "fireviewer.geo-gpu-request.v1"
)
RESPONSE_SCHEMA: Final[Literal["fireviewer.geo-gpu-response.v1"]] = (
    "fireviewer.geo-gpu-response.v1"
)
ARTIFACT_SCHEMA: Final[Literal["fireviewer.sagemaker-geo-model-artifact.v1"]] = (
    "fireviewer.sagemaker-geo-model-artifact.v1"
)
















































































def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifact(model_root: Path) -> GeoModelArtifactManifest:
    root = model_root.resolve(strict=True)
    manifest_path = root / "fireviewer-model-manifest.json"
    manifest = GeoModelArtifactManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    for expected in manifest.files:
        candidate = (root / expected.path).resolve(strict=True)
        if root not in candidate.parents:
            raise ValueError("model artifact path escapes its root")
        if (
            not candidate.is_file()
            or candidate.stat().st_size != expected.byte_size
            or _file_digest(candidate) != expected.sha256
        ):
            raise ValueError(f"model artifact file failed validation: {expected.path}")
    return manifest


def _decode_payload(payload: EncodedPayload, *, max_bytes: int) -> bytes:
    try:
        decoded = base64.b64decode(payload.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"payload {payload.input_id} is not canonical base64") from exc
    if not decoded or len(decoded) > max_bytes:
        raise ValueError(f"payload {payload.input_id} is outside the byte budget")
    if sha256(decoded).hexdigest() != payload.content_sha256:
        raise ValueError(f"payload {payload.input_id} failed SHA-256 verification")
    return decoded


class _LocalPayloadFetcher:
    def __init__(self, paths_by_url: dict[str, Path]) -> None:
        self.paths_by_url = paths_by_url

    @contextmanager
    def download(self, url: str) -> Iterator[Path]:
        path = self.paths_by_url.get(url)
        if path is None or not path.is_file():
            raise ValueError("Prithvi requested an undeclared local payload")
        yield path


class GeoGpuRuntime:
    """Sequential, lazy-loading runtime for the two interchangeable GPU providers."""

    def __init__(
        self,
        *,
        model_root: Path,
        max_payload_bytes: int = 256 * 1024 * 1024,
        megaloc_factory: Callable[[], Any] | None = None,
        prithvi_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.model_root = model_root
        self.max_payload_bytes = max_payload_bytes
        self._megaloc_factory = megaloc_factory
        self._prithvi_factory = prithvi_factory
        self._manifest: GeoModelArtifactManifest | None = None

    def health(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = validate_model_artifact(self.model_root)
        return {
            "status": "ready",
            "megaloc_revision": self._manifest.megaloc_revision,
            "prithvi_revision": self._manifest.prithvi_revision,
            "gpu_started": False,
        }

    def _new_megaloc(self) -> Any:
        if self._megaloc_factory is not None:
            return self._megaloc_factory()
        bundle_root = self.model_root / "megaloc"
        bundle_manifest = LocalMegaLocBundleManifest.model_validate_json(
            (bundle_root / "bundle-manifest.json").read_text(encoding="utf-8")
        )
        device = os.getenv("FW_GEO_DEVICE", "cuda")
        if device not in {"cpu", "cuda"}:
            raise ValueError("FW_GEO_DEVICE must be cpu or cuda")
        return TorchMegaLocEncoder(
            model_loader=LocalMegaLocModelLoader(
                directory=bundle_root,
                manifest=bundle_manifest,
            ),
            model_version=MEGALOC_REVISION,
            config=MegaLocConfig(device=device, batch_size=4),
        )

    def _new_prithvi(self, fetcher: Any) -> Any:
        if self._prithvi_factory is not None:
            return self._prithvi_factory(fetcher)
        return PrithviBurnedAreaAdapter(
            ModelSpec(
                role="burned_area",
                model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
                revision=PRITHVI_REVISION,
            ),
            cache_root=self.model_root / "prithvi-cache",
            fetcher=fetcher,
        )

    def invoke(self, request: GeoGpuRequest) -> GeoGpuResponse:
        self.health()
        decoded = {
            payload.input_id: _decode_payload(payload, max_bytes=self.max_payload_bytes)
            for payload in request.payloads
        }
        if sum(len(value) for value in decoded.values()) > self.max_payload_bytes:
            raise ValueError("request payloads exceed the aggregate byte budget")
        if request.operation == "megaloc.encode":
            return self._invoke_megaloc(request, decoded)
        return self._invoke_prithvi(request, decoded)

    def _invoke_megaloc(
        self,
        request: GeoGpuRequest,
        decoded: dict[SafeIdentifierV2, bytes],
    ) -> GeoGpuResponse:
        from io import BytesIO

        from PIL import Image

        media = []
        for payload in request.payloads:
            with Image.open(BytesIO(decoded[payload.input_id])) as image:
                image.load()
                media.append((payload.input_id, image.copy()))
        batch, vectors = self._new_megaloc().encode(tuple(media))
        vector_bytes = vectors.astype("<f4", copy=False).tobytes(order="C")
        return GeoGpuResponse(
            request_id=request.request_id,
            operation=request.operation,
            status="completed",
            model_id=batch.model_id,
            model_revision=MEGALOC_REVISION,
            result={
                "batch": batch.model_dump(mode="json"),
                "vector_encoding": "base64-float32-little-endian-row-major",
                "vector_count": len(batch.embeddings),
                "vector_dimension": batch.embeddings[0].dimension,
                "vectors_sha256": sha256(vector_bytes).hexdigest(),
                "vectors_base64": base64.b64encode(vector_bytes).decode("ascii"),
            },
        )

    def _invoke_prithvi(
        self,
        request: GeoGpuRequest,
        decoded: dict[SafeIdentifierV2, bytes],
    ) -> GeoGpuResponse:
        worker_input = request.worker_input
        if worker_input is None:
            raise ValueError("Prithvi request lost its worker_input")
        with tempfile.TemporaryDirectory(prefix="fireviewer-prithvi-") as directory:
            root = Path(directory)
            paths_by_url: dict[str, Path] = {}
            for item in worker_input.items:
                if item.working_file_url is None:
                    raise ValueError("Prithvi worker item has no working_file_url")
                suffix = Path(urlsplit(str(item.working_file_url)).path).suffix or ".tif"
                path = root / f"{item.input_id}{suffix[:16]}"
                path.write_bytes(decoded[item.input_id])
                paths_by_url[str(item.working_file_url)] = path
            adapter = self._new_prithvi(_LocalPayloadFetcher(paths_by_url))
            try:
                adapter.load()
                annotations, proposals = adapter.infer(worker_input)
            finally:
                adapter.unload()
        serialized_annotations = {
            key: [item.model_dump(mode="json") for item in values]
            for key, values in annotations.items()
        }
        serialized_proposals = {
            key: [item.model_dump(mode="json") for item in values]
            for key, values in proposals.items()
        }
        has_result = any(serialized_annotations.values()) or any(serialized_proposals.values())
        return GeoGpuResponse(
            request_id=request.request_id,
            operation=request.operation,
            status="completed" if has_result else "abstained",
            model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
            model_revision=PRITHVI_REVISION,
            result={
                "annotations": serialized_annotations,
                "spatial_proposals": serialized_proposals,
            },
            reason_codes=() if has_result else ("no_burned_area_proposal",),
        )


class SageMakerHandler(BaseHTTPRequestHandler):
    runtime: GeoGpuRuntime
    max_request_bytes = 360_000_000

    def log_message(self, format: str, *args: object) -> None:
        print(json.dumps({"message": format % args}, separators=(",", ":")), flush=True)

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        if self.path != "/ping":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            health = self.runtime.health()
        except Exception:
            self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "model_not_ready"})
            return
        self._write_json(HTTPStatus.OK, health)

    def do_POST(self) -> None:
        if self.path != "/invocations":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length < 1 or content_length > self.max_request_bytes:
            self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
            return
        try:
            raw = self.rfile.read(content_length)
            request = GeoGpuRequest.model_validate_json(raw)
            response = self.runtime.invoke(request)
        except (ValidationError, ValueError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        except Exception:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference_failed"})
            return
        self._write_json(HTTPStatus.OK, response.model_dump(mode="json", by_alias=True))


def main() -> None:
    model_root = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    runtime = GeoGpuRuntime(model_root=model_root)
    runtime.health()
    SageMakerHandler.runtime = runtime
    server = HTTPServer(("0.0.0.0", 8080), SageMakerHandler)  # noqa: S104
    server.serve_forever()


if __name__ == "__main__":
    main()
