"""Authenticated one-shot validation server for a regular RunPod GPU pod."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import tempfile
import threading
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fireviewer_contracts.contracts import WorkerInput
from fireviewer_orchestrator.handler import handle_job
from fireviewer_contracts.client.media_fetcher import MediaFetcher, MediaFetchError
from fireviewer_orchestrator.security import WorkerSettings
from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

MAX_ENCODED_PAYLOAD_BYTES = 1024 * 1024
MAX_DECOMPRESSED_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_ENCODED_ASSET_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_DECOMPRESSED_ASSET_BUNDLE_BYTES = 24 * 1024 * 1024
MAX_ASSET_BUNDLE_RAW_BYTES = 16 * 1024 * 1024
MAX_ASSET_BUNDLE_COUNT = 20
VALIDATION_ASSET_HOST = "validation-assets.internal"
SAFE_ASSET_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
REQUIRED_IMAGE_ROLES = frozenset({"visual_grounding", "multimodal_extraction"})


class PodValidationError(RuntimeError):
    """Raised when the validation mode is unsafe or malformed."""


def _decode_gzip_base64(
    encoded: str,
    *,
    label: str,
    max_encoded_bytes: int,
    max_decompressed_bytes: int,
) -> bytes:
    if not encoded:
        raise PodValidationError(f"{label} is absent")
    try:
        encoded_bytes = encoded.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PodValidationError(f"{label} is not ASCII base64") from exc
    if len(encoded_bytes) > max_encoded_bytes:
        raise PodValidationError(f"{label} exceeds the encoded size cap")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as archive:
            raw = archive.read(max_decompressed_bytes + 1)
    except (ValueError, binascii.Error, EOFError, gzip.BadGzipFile, zlib.error) as exc:
        raise PodValidationError(f"{label} is not valid gzip base64") from exc
    if len(raw) > max_decompressed_bytes:
        raise PodValidationError(f"{label} exceeds the decompressed size cap")
    return raw


def decode_payload(encoded: str) -> dict[str, Any]:
    raw = _decode_gzip_base64(
        encoded,
        label="validation payload",
        max_encoded_bytes=MAX_ENCODED_PAYLOAD_BYTES,
        max_decompressed_bytes=MAX_DECOMPRESSED_PAYLOAD_BYTES,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PodValidationError("validation payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PodValidationError("validation payload must be a JSON object")
    return WorkerInput.model_validate(payload).model_dump(mode="json")


def decode_asset_bundle(encoded: str) -> dict[str, bytes]:
    raw = _decode_gzip_base64(
        encoded,
        label="validation asset bundle",
        max_encoded_bytes=MAX_ENCODED_ASSET_BUNDLE_BYTES,
        max_decompressed_bytes=MAX_DECOMPRESSED_ASSET_BUNDLE_BYTES,
    )
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PodValidationError("validation asset bundle is not valid JSON") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise PodValidationError("validation asset bundle schema is unsupported")
    raw_assets = bundle.get("assets")
    if not isinstance(raw_assets, dict) or not 1 <= len(raw_assets) <= MAX_ASSET_BUNDLE_COUNT:
        raise PodValidationError("validation asset bundle count is outside the allowed range")

    assets: dict[str, bytes] = {}
    raw_bytes = 0
    for name, descriptor in raw_assets.items():
        if not isinstance(name, str) or SAFE_ASSET_NAME.fullmatch(name) is None:
            raise PodValidationError("validation asset bundle contains an unsafe asset name")
        if not isinstance(descriptor, dict):
            raise PodValidationError(f"validation asset descriptor is invalid: {name}")
        digest = descriptor.get("sha256")
        encoded_value = descriptor.get("data_b64")
        if not isinstance(digest, str) or not isinstance(encoded_value, str):
            raise PodValidationError(f"validation asset descriptor is incomplete: {name}")
        try:
            value = base64.b64decode(encoded_value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise PodValidationError(f"validation asset is not valid base64: {name}") from exc
        raw_bytes += len(value)
        if raw_bytes > MAX_ASSET_BUNDLE_RAW_BYTES:
            raise PodValidationError("validation asset bundle exceeds the raw byte cap")
        if not hmac.compare_digest(hashlib.sha256(value).hexdigest(), digest):
            raise PodValidationError(f"validation asset digest mismatch: {name}")
        assets[name] = value
    return assets


def validation_asset_names(payload: Mapping[str, Any]) -> frozenset[str]:
    names: set[str] = set()
    items = payload.get("items")
    if not isinstance(items, list):
        return frozenset()
    for item in items:
        if not isinstance(item, dict) or item.get("working_file_url") is None:
            continue
        parsed = urlsplit(str(item["working_file_url"]))
        if parsed.hostname != VALIDATION_ASSET_HOST:
            continue
        name = parsed.path.removeprefix("/")
        if SAFE_ASSET_NAME.fullmatch(name) is None:
            raise PodValidationError("validation payload contains an unsafe asset URL")
        names.add(name)
    return frozenset(names)


class BundleMediaFetcher(MediaFetcher):
    """Validation-only media transport for the exact private assets injected at boot."""

    def __init__(self, assets: Mapping[str, bytes], *, max_bytes: int) -> None:
        super().__init__(allowed_hosts=frozenset({VALIDATION_ASSET_HOST}), max_bytes=max_bytes)
        self.assets = dict(assets)

    @contextmanager
    def download(self, url: str) -> Iterator[Path]:
        parsed = urlsplit(url)
        name = parsed.path.removeprefix("/")
        if (
            parsed.scheme != "https"
            or parsed.hostname != VALIDATION_ASSET_HOST
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or SAFE_ASSET_NAME.fullmatch(name) is None
        ):
            raise MediaFetchError("validation media URL is outside the bundled HTTPS boundary")
        value = self.assets.get(name)
        if value is None:
            raise MediaFetchError("validation media URL has no matching bundled asset")
        if len(value) > self.max_bytes:
            raise MediaFetchError("bundled validation media exceeds the download budget")

        target: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="fw-validation-media-", suffix=Path(name).suffix, delete=False
            ) as output:
                target = Path(output.name)
                output.write(value)
            yield target
        finally:
            if target is not None:
                target.unlink(missing_ok=True)


def hardware_probe() -> dict[str, Any]:
    import flash_attn
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise PodValidationError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        raise PodValidationError("GPU compute capability is below the FlashAttention 2 minimum")
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": properties.total_memory,
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "flash_attention_version": flash_attn.__version__,
    }


def build_validation_report(
    payload: Mapping[str, Any],
    output: Mapping[str, Any],
    hardware: Mapping[str, Any],
    *,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    runs = output.get("model_runs")
    if not isinstance(runs, list):
        raise PodValidationError("worker output has no model_runs array")
    status_by_role = {
        str(run.get("model_role")): str(run.get("status")) for run in runs if isinstance(run, dict)
    }
    missing_or_failed = sorted(
        role for role in REQUIRED_IMAGE_ROLES if status_by_role.get(role) != "succeeded"
    )
    items = output.get("items")
    expected_items = payload.get("items")
    item_count_matches = (
        len(items) == len(expected_items)
        if isinstance(items, list) and isinstance(expected_items, list)
        else False
    )
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    validation_kind = (
        "synthetic_ground_photo_provenance_pipeline"
        if payload.get("batch_id") == "synthetic-provenance-evaluation"
        else "synthetic_operational_image_pipeline"
    )
    return {
        "schema_version": 1,
        "validation_kind": validation_kind,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "batch_id": payload.get("batch_id"),
        "hardware": dict(hardware),
        "required_image_roles": sorted(REQUIRED_IMAGE_ROLES),
        "role_statuses": status_by_role,
        "missing_or_failed_required_roles": missing_or_failed,
        "item_count_matches": item_count_matches,
        "pipeline_passed": not missing_or_failed and item_count_matches,
        "training_membership": False,
        "human_validation_required": True,
        "deployment_ready": False,
        "known_expected_skips": {
            "asr": "the current deployment-test batch contains no audio",
            "fire_detection": "no approved RT-DETR checkpoint is configured",
        },
        "worker_output": dict(output),
    }


JobHandler = Callable[[dict[str, Any]], dict[str, Any]]
HardwareProbe = Callable[[], dict[str, Any]]


def run_validation(
    payload: dict[str, Any],
    *,
    job_handler: JobHandler = handle_job,
    probe: HardwareProbe = hardware_probe,
) -> dict[str, Any]:
    validated = WorkerInput.model_validate(payload).model_dump(mode="json")
    started_at = datetime.now(UTC)
    hardware = probe()
    output = job_handler({"input": validated})
    return build_validation_report(
        validated,
        output,
        hardware,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )


class ValidationState:
    def __init__(self, *, initial_status: str = "pending") -> None:
        self._lock = threading.Lock()
        self._status = initial_status
        self._report: dict[str, Any] | None = None
        self._error: str | None = None

    def update(
        self,
        status: str,
        *,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._status = status
            self._report = report
            self._error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self._status, "report": self._report, "error": self._error}


class ValidationHTTPServer(ThreadingHTTPServer):
    validation_token: str
    validation_state: ValidationState
    validation_payload: dict[str, Any]
    validation_worker_lock: threading.Lock
    validation_worker_started: bool

    def start_validation(self, assets: Mapping[str, bytes]) -> bool:
        with self.validation_worker_lock:
            if self.validation_worker_started:
                return False
            self.validation_worker_started = True
            self.validation_state.update("pending")
            worker = threading.Thread(
                target=_validation_worker,
                args=(self.validation_payload, self.validation_state, assets),
                name="firewarning-pod-validation",
                daemon=True,
            )
            worker.start()
            return True


class ValidationRequestHandler(BaseHTTPRequestHandler):
    server: ValidationHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.validation_token}"
        return hmac.compare_digest(self.headers.get("Authorization", ""), expected)

    def do_GET(self) -> None:
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        path = urlsplit(self.path).path
        snapshot = self.server.validation_state.snapshot()
        if path == "/health":
            self._write_json(HTTPStatus.OK, {"status": snapshot["status"]})
            return
        if path != "/result":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if snapshot["status"] in {"awaiting_assets", "pending", "running"}:
            self._write_json(HTTPStatus.ACCEPTED, {"status": snapshot["status"]})
            return
        if snapshot["status"] == "failed":
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "failed", "error": snapshot["error"]},
            )
            return
        self._write_json(
            HTTPStatus.OK,
            {"status": "completed", "report": snapshot["report"]},
        )

    def do_POST(self) -> None:
        if not self._authorized():
            self._write_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if urlsplit(self.path).path != "/assets":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.server.validation_state.snapshot()["status"] != "awaiting_assets":
            self._write_json(HTTPStatus.CONFLICT, {"error": "asset_upload_not_available"})
            return
        raw_length = self.headers.get("Content-Length", "")
        try:
            content_length = int(raw_length)
        except ValueError:
            self._write_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
            return
        if not 1 <= content_length <= MAX_ENCODED_ASSET_BUNDLE_BYTES + 2:
            self._write_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "asset_upload_too_large"}
            )
            return
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "incomplete_asset_upload"})
            return
        try:
            encoded = body.decode("ascii").strip()
            assets = decode_asset_bundle(encoded)
            expected_names = validation_asset_names(self.server.validation_payload)
            if set(assets) != expected_names:
                raise PodValidationError("validation asset names do not match the payload")
        except (PodValidationError, UnicodeDecodeError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if not self.server.start_validation(assets):
            self._write_json(HTTPStatus.CONFLICT, {"error": "asset_upload_already_consumed"})
            return
        self._write_json(
            HTTPStatus.ACCEPTED,
            {"status": "accepted", "asset_count": len(assets)},
        )


def _validation_worker(
    payload: dict[str, Any], state: ValidationState, assets: Mapping[str, bytes]
) -> None:
    state.update("running")
    try:
        if assets:
            settings = WorkerSettings.from_environment()
            factory = TransformersAdapterFactory(
                cache_root=Path(settings.hf_cache_root),
                allowed_hosts=settings.allowed_media_hosts,
                max_download_bytes=settings.max_download_bytes,
                fetcher=BundleMediaFetcher(assets, max_bytes=settings.max_download_bytes),
            )
            report = run_validation(
                payload,
                job_handler=lambda job: handle_job(job, factory=factory),
            )
        else:
            report = run_validation(payload)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"firewarning pod validation failed: {error}", flush=True)
        state.update("failed", error=error)
        return
    print(
        f"firewarning pod validation completed: pipeline_passed={report['pipeline_passed']}",
        flush=True,
    )
    state.update("completed", report=report)


def main() -> None:
    token = os.getenv("FW_VALIDATION_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("FW_VALIDATION_TOKEN must contain at least 32 characters")
    encoded = os.getenv("FW_VALIDATION_PAYLOAD_GZIP_B64", "")
    try:
        payload = decode_payload(encoded)
        encoded_assets = os.getenv("FW_VALIDATION_ASSET_BUNDLE_GZIP_B64", "")
        assets = decode_asset_bundle(encoded_assets) if encoded_assets else {}
        required_asset_names = validation_asset_names(payload)
        if assets and set(assets) != required_asset_names:
            raise PodValidationError("validation asset names do not match the payload")
        port = int(os.getenv("FW_VALIDATION_PORT", "8000"))
    except (PodValidationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not 1024 <= port <= 65535:
        raise SystemExit("FW_VALIDATION_PORT must be between 1024 and 65535")

    state = ValidationState(
        initial_status="awaiting_assets" if required_asset_names and not assets else "pending"
    )
    server = ValidationHTTPServer(
        ("0.0.0.0", port),  # noqa: S104 - the authenticated RunPod proxy must reach this port
        ValidationRequestHandler,
    )
    server.validation_token = token
    server.validation_state = state
    server.validation_payload = payload
    server.validation_worker_lock = threading.Lock()
    server.validation_worker_started = False
    if assets or not required_asset_names:
        server.start_validation(assets)
    print(f"firewarning pod validation server listening on port {port}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
