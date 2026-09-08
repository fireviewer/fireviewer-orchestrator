from __future__ import annotations

import base64
import http.client
import json
import threading
from datetime import UTC, datetime
from hashlib import sha256
from http import HTTPStatus
from http.server import HTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fireviewer_contracts.contracts import WorkerInputV2
from fireviewer_orchestrator.mvp.gpu.sagemaker_service import (
    MEGALOC_REVISION,
    PRITHVI_REVISION,
    GeoGpuRequest,
    GeoGpuRuntime,
    SageMakerHandler,
    validate_model_artifact,
)
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocBatch, MegaLocEmbedding


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _artifact(tmp_path: Path) -> Path:
    files = []
    for index in range(6):
        relative = f"fixtures/file-{index}.bin"
        value = f"fixture-{index}".encode()
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        files.append({"path": relative, "byte_size": len(value), "sha256": _sha(value)})
    (tmp_path / "fireviewer-model-manifest.json").write_text(
        json.dumps(
            {
                "schema": "fireviewer.sagemaker-geo-model-artifact.v1",
                "megaloc_revision": MEGALOC_REVISION,
                "prithvi_revision": PRITHVI_REVISION,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _encoded_payload(input_id: str, content_type: str, value: bytes) -> dict[str, str]:
    return {
        "input_id": input_id,
        "content_type": content_type,
        "content_sha256": _sha(value),
        "content_base64": base64.b64encode(value).decode("ascii"),
    }


class _FakeMegaLoc:
    def encode(self, media):
        assert [item[0] for item in media] == ["MEDIA-1"]
        vector = np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        return (
            MegaLocBatch(
                model_id="gberton/MegaLoc",
                model_version=MEGALOC_REVISION,
                embeddings=(
                    MegaLocEmbedding(
                        embedding_id="EMB-1234567890ABCDEF12345678",
                        media_id="MEDIA-1",
                        dimension=4,
                        vector_sha256=_sha(vector[0].tobytes()),
                    ),
                ),
            ),
            vector,
        )


class _FakePrithvi:
    def __init__(self, fetcher) -> None:
        self.fetcher = fetcher
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(self, worker_input):
        assert self.loaded is True
        url = str(worker_input.items[0].working_file_url)
        with self.fetcher.download(url) as path:
            assert path.read_bytes() == b"tiff-fixture"
        return {}, {}

    def unload(self) -> None:
        self.unloaded = True


def _worker_input() -> WorkerInputV2:
    now = datetime(2026, 8, 23, 12, tzinfo=UTC)
    return WorkerInputV2.model_validate(
        {
            "schema_version": "2.0",
            "batch_id": "SATELLITE-BATCH-1",
            "batch_type": "satellite_media",
            "priority": "scheduled",
            "analysis_window": {
                "analysis_id": "ANALYSIS-1",
                "fire_id": "FR-26-00001",
                "episode_id": "EPISODE-1",
                "window_start_at": now.isoformat(),
                "window_end_at": datetime(2026, 8, 23, 13, tzinfo=UTC).isoformat(),
                "local_date": "2026-08-23",
                "timezone": "Europe/Paris",
            },
            "reference_bundle": {
                "reference_id": "REFERENCE-1",
                "manifest_sha256": "1" * 64,
                "assets": [
                    {
                        "kind": "source_manifest",
                        "working_file_url": "https://media.internal/reference.json",
                        "sha256": "2" * 64,
                        "crs": "EPSG:4326",
                    }
                ],
            },
            "items": [
                {
                    "input_id": "SATELLITE-1",
                    "media_type": "satellite_image",
                    "working_file_url": "https://media.internal/satellite.tif",
                    "provenance": {
                        "source_key": "SENTINEL-2",
                        "source_reference_url": "https://dataspace.copernicus.eu/",
                        "license_identifier": "COPERNICUS-DATA",
                        "attribution": "Contains modified Copernicus Sentinel data",
                        "trust": "institutional",
                    },
                    "satellite": {
                        "product_id": "PRODUCT-1",
                        "provider": "Copernicus Sentinel-2",
                        "acquired_at": now.isoformat(),
                        "crs": "EPSG:4326",
                        "raster_width_px": 512,
                        "raster_height_px": 512,
                        "geotransform": [2.46, 0.00025, 0, 48.44, 0, -0.00025],
                        "bbox_wgs84": [2.46, 48.31, 2.59, 48.44],
                        "resolution_m": 20,
                        "bands": [
                            "BLUE",
                            "GREEN",
                            "RED",
                            "NIR_NARROW",
                            "SWIR_1",
                            "SWIR_2",
                        ],
                    },
                }
            ],
        }
    )


def test_model_artifact_validation_fails_closed_on_digest_change(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    assert validate_model_artifact(root).megaloc_revision == MEGALOC_REVISION

    (root / "fixtures/file-2.bin").write_bytes(b"changed")

    try:
        validate_model_artifact(root)
    except ValueError as exc:
        assert "failed validation" in str(exc)
    else:
        raise AssertionError("tampered model artifact was accepted")


def test_megaloc_request_returns_digest_qualified_vectors(tmp_path: Path) -> None:
    stream = BytesIO()
    Image.new("RGB", (8, 8), (255, 64, 0)).save(stream, format="PNG")
    payload = stream.getvalue()
    runtime = GeoGpuRuntime(
        model_root=_artifact(tmp_path),
        megaloc_factory=_FakeMegaLoc,
    )
    request = GeoGpuRequest.model_validate(
        {
            "schema": "fireviewer.geo-gpu-request.v1",
            "request_id": "REQUEST-1",
            "operation": "megaloc.encode",
            "payloads": [_encoded_payload("MEDIA-1", "image/png", payload)],
        }
    )

    response = runtime.invoke(request)

    assert response.status == "completed"
    assert response.model_revision == MEGALOC_REVISION
    vector_bytes = base64.b64decode(response.result["vectors_base64"])
    assert response.result["vectors_sha256"] == _sha(vector_bytes)
    assert np.frombuffer(vector_bytes, dtype="<f4").tolist() == [0.5, 0.5, 0.5, 0.5]


def test_prithvi_request_abstains_when_model_returns_no_polygon(tmp_path: Path) -> None:
    payload = b"tiff-fixture"
    runtime = GeoGpuRuntime(
        model_root=_artifact(tmp_path),
        prithvi_factory=_FakePrithvi,
    )
    response = runtime.invoke(
        GeoGpuRequest.model_validate(
            {
                "schema": "fireviewer.geo-gpu-request.v1",
                "request_id": "REQUEST-2",
                "operation": "prithvi.burned_area",
                "payloads": [_encoded_payload("SATELLITE-1", "image/tiff", payload)],
                "worker_input": _worker_input().model_dump(mode="json"),
            }
        )
    )

    assert response.status == "abstained"
    assert response.model_revision == PRITHVI_REVISION
    assert response.reason_codes == ("no_burned_area_proposal",)


def test_http_surface_exposes_ping_and_rejects_invalid_invocation(tmp_path: Path) -> None:
    SageMakerHandler.runtime = GeoGpuRuntime(model_root=_artifact(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), SageMakerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/ping")
        response = connection.getresponse()
        ping = json.loads(response.read())
        assert response.status == 200
        assert ping["status"] == "ready"

        connection.request(
            "POST",
            "/invocations",
            body=b"{}",
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        )
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "invalid_request"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_response_tolerates_a_disconnected_health_client() -> None:
    class DisconnectedWriter:
        def write(self, _value: bytes) -> None:
            raise BrokenPipeError

    handler = object.__new__(SageMakerHandler)
    handler.send_response = lambda _status: None
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None
    handler.wfile = DisconnectedWriter()

    handler._write_json(HTTPStatus.OK, {"status": "ready"})
