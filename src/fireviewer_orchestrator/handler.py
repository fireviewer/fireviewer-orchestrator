from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from fireviewer_vision_runtime.adapters import (
    AdapterFactory,
    UnavailableAdapterFactory,
    adapter_factory_job_scope,
)
from fireviewer_orchestrator.boot_clock import BOOT_STARTED_AT
from fireviewer_contracts.contracts import (
    ResearchInputV1,
    ResearchOutputV1,
    WorkerInput,
    WorkerInputV2,
)
from fireviewer_contracts.model_registry import (
    RegistryError,
    build_model_group_registry,
    build_registry,
)
from fireviewer_evidence_ingestion.research_client import ResearchServiceError
from fireviewer_orchestrator.security import ConfigurationError, WorkerSettings
from fireviewer_orchestrator.session_runner import SessionRunner
from fireviewer_contracts.stage_contracts import load_stage_contract_registry
from fireviewer_orchestrator.v2_runner import from_legacy_output, to_legacy_input
from fireviewer_contracts.validation import (
    OutputValidationError,
    validate_internal_urls,
    validate_v2_internal_urls,
)

if TYPE_CHECKING:
    from fireviewer_vision_runtime.event_perception import EventPerceptionAdapter
    from fireviewer_geolocation.spatial_pipeline import DeterministicSpatialPipeline
    from fireviewer_vision_runtime.v2_burned_area import BurnedAreaAdapter
    from fireviewer_vision_runtime.v2_pointing import FirePointingAdapter

BOOT_READY_MS = round((perf_counter() - BOOT_STARTED_AT) * 1_000)
_GPU_SESSION_LOCK = threading.Lock()


def _event_candidate_id(raw_input: dict[str, Any]) -> str:
    bundle = raw_input.get("bundle")
    if not isinstance(bundle, dict):
        return "INVALID"
    candidate_id = bundle.get("candidate_id")
    return candidate_id if isinstance(candidate_id, str) else "INVALID"


def _event_validation_codes(exc: ValidationError) -> list[str]:
    """Return closed validation codes without echoing private input values."""

    codes: list[str] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"])
        codes.append(f"event_input_invalid:{location}:{error['type']}")
    return codes or ["event_input_invalid"]


def _research_failure(
    raw_input: dict[str, Any],
    *,
    error_code: str,
    detail: str,
    retryable: bool,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    spec = build_registry()["source_research"]
    research_id = raw_input.get("research_id", "INVALID")
    output = ResearchOutputV1.model_validate(
        {
            "research_id": research_id if isinstance(research_id, str) else "INVALID",
            "status": "failed",
            "retryable": retryable,
            "model_run": {
                "model_id": spec.model_id,
                "revision": spec.revision,
                "status": "failed",
                "started_at": now,
                "finished_at": now,
                "load_ms": 0,
                "inference_ms": 0,
                "error_code": error_code[:128],
            },
            "queries": [],
            "candidates": [],
            "validation_errors": [detail[:1_000]],
        }
    )
    return output.model_dump(mode="json")


def _runtime_factory(settings: WorkerSettings) -> AdapterFactory:
    if os.getenv("FW_ENABLE_TRANSFORMERS_RUNTIME", "false").lower() != "true":
        return UnavailableAdapterFactory()
    from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

    return TransformersAdapterFactory(
        cache_root=Path(settings.hf_cache_root),
        allowed_hosts=settings.allowed_media_hosts,
        max_download_bytes=settings.max_download_bytes,
        max_cache_bytes=settings.max_media_cache_bytes,
    )


def _runtime_spatial_pipeline(
    factory: AdapterFactory,
) -> DeterministicSpatialPipeline | None:
    from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

    if not isinstance(factory, TransformersAdapterFactory):
        return None
    from fireviewer_geolocation.spatial_pipeline import DeterministicSpatialPipeline

    return DeterministicSpatialPipeline(fetcher=factory.fetcher)


def _runtime_fire_pointing_adapter(
    factory: AdapterFactory,
) -> FirePointingAdapter | None:
    from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

    if not isinstance(factory, TransformersAdapterFactory):
        return None
    spec = build_registry().get("fire_pointing")
    if spec is None:
        return None
    return factory.create_fire_pointing(spec)


def _runtime_burned_area_adapter(
    factory: AdapterFactory,
) -> BurnedAreaAdapter | None:
    from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

    if not isinstance(factory, TransformersAdapterFactory):
        return None
    spec = build_registry()["burned_area"]
    return cast("BurnedAreaAdapter", factory.create_burned_area(spec))


def _runtime_event_perception_adapter(
    factory: AdapterFactory,
) -> EventPerceptionAdapter | None:
    from fireviewer_vision_runtime.transformers_adapters import TransformersAdapterFactory

    if not isinstance(factory, TransformersAdapterFactory):
        return None
    spec = build_registry().get("fire_pointing")
    if spec is None:
        return None
    return cast("EventPerceptionAdapter", factory.create_fire_pointing(spec))


def handle_job(
    job: dict[str, Any],
    *,
    factory: AdapterFactory | None = None,
    spatial_pipeline: DeterministicSpatialPipeline | None = None,
    event_perception_adapter: EventPerceptionAdapter | None = None,
) -> dict[str, Any]:
    raw_input = job.get("input")
    if not isinstance(raw_input, dict):
        return {
            "schema_version": "1.0",
            "batch_id": "INVALID",
            "status": "failed",
            "retryable": False,
            "model_runs": [],
            "items": [],
            "validation_errors": ["input:missing_or_not_an_object"],
            "boot_ms": BOOT_READY_MS,
        }
    requested_schema = raw_input.get("schema_version", "1.0")
    if requested_schema == "lab-bonsai-integration-1" and os.getenv("FW_ENABLE_LAB_PROBES") == "true":
        from fireviewer_orchestrator.lab_probe import run_probe
        settings = WorkerSettings.from_environment()
        with _GPU_SESSION_LOCK:
            return run_probe(raw_input, _runtime_factory(settings))

    if requested_schema == "event-2.0":
        from fireviewer_vision_runtime.event_perception import (
            event_has_working_urls,
            event_requires_image_inference,
            run_event_image_perception,
            validate_event_working_urls,
        )
        from fireviewer_orchestrator.event_pipeline import (
            DeterministicEventPipeline,
            EventPipelineInput,
            PerceptionFailure,
            event_pipeline_enabled,
        )

        if not event_pipeline_enabled():
            return {
                "schema_version": "event-result-2.0",
                "candidate_id": _event_candidate_id(raw_input),
                "status": "failed",
                "view_profile": None,
                "perception_anchors": [],
                "spatial_evidence": [],
                "localization_attempts": [],
                "event_proposals": [],
                "independent_external_families": [],
                "contradictions": [],
                "reason_codes": ["event_pipeline_disabled"],
                "requires_human_review": True,
            }
        try:
            request = EventPipelineInput.model_validate(raw_input)
            url_failures: tuple[PerceptionFailure, ...] = ()
            settings: WorkerSettings | None = None
            if event_has_working_urls(request):
                try:
                    settings = WorkerSettings.from_environment()
                except ConfigurationError:
                    url_failures = tuple(
                        PerceptionFailure(
                            evidence_asset_id=asset.evidence_asset_id,
                            reason_code="media_url_allowlist_unconfigured",
                        )
                        for asset in request.bundle.evidence_assets
                        if asset.working_file_url is not None
                    )
                else:
                    url_failures = validate_event_working_urls(
                        request,
                        settings.allowed_media_hosts,
                    )

            invalid_assets = {
                failure.evidence_asset_id
                for failure in url_failures
                if failure.evidence_asset_id is not None
            }
            if invalid_assets:
                safe_anchors = tuple(
                    anchor
                    for anchor in request.perception_anchors
                    if anchor.evidence_asset_id not in invalid_assets
                )
                safe_anchor_ids = {anchor.anchor_id for anchor in safe_anchors}
                request = EventPipelineInput.model_validate(
                    request.model_copy(
                        update={
                            "perception_anchors": safe_anchors,
                            "spatial_evidence": tuple(
                                item
                                for item in request.spatial_evidence
                                if item.anchor_id in safe_anchor_ids
                            ),
                        }
                    )
                )

            requires_inference = event_requires_image_inference(request, url_failures)
            adapter_factory: AdapterFactory | None = factory
            if requires_inference and adapter_factory is None and settings is not None:
                adapter_factory = _runtime_factory(settings)
            adapter = event_perception_adapter
            if adapter is None and adapter_factory is not None:
                adapter = _runtime_event_perception_adapter(adapter_factory)

            if requires_inference and adapter is not None:
                if not _GPU_SESSION_LOCK.acquire(blocking=False):
                    enriched, failures = run_event_image_perception(
                        request,
                        adapter=None,
                        url_failures=url_failures,
                        unavailable_reason_code="gpu_session_already_active",
                    )
                else:
                    try:
                        assert adapter_factory is not None
                        with adapter_factory_job_scope(adapter_factory):
                            enriched, failures = run_event_image_perception(
                                request,
                                adapter=adapter,
                                url_failures=url_failures,
                            )
                    finally:
                        _GPU_SESSION_LOCK.release()
            else:
                enriched, failures = run_event_image_perception(
                    request,
                    adapter=adapter,
                    url_failures=url_failures,
                )
            return (
                DeterministicEventPipeline()
                .run(
                    enriched,
                    perception_failures=failures,
                )
                .model_dump(mode="json")
            )
        except ValidationError as exc:
            return {
                "schema_version": "event-result-2.0",
                "candidate_id": _event_candidate_id(raw_input),
                "status": "failed",
                "view_profile": None,
                "perception_anchors": [],
                "spatial_evidence": [],
                "localization_attempts": [],
                "event_proposals": [],
                "independent_external_families": [],
                "contradictions": [],
                "reason_codes": _event_validation_codes(exc),
                "requires_human_review": True,
            }
    if requested_schema == "research-1.0":
        try:
            research = ResearchInputV1.model_validate(raw_input)
            if not _GPU_SESSION_LOCK.acquire(blocking=False):
                return _research_failure(
                    raw_input,
                    error_code="gpu_session_already_active",
                    detail="worker:gpu_session_already_active",
                    retryable=True,
                )
            try:
                from fireviewer_evidence_ingestion.research_client import run_isolated_research

                return run_isolated_research(research).model_dump(mode="json")
            finally:
                _GPU_SESSION_LOCK.release()
        except ResearchServiceError as exc:
            return _research_failure(
                raw_input,
                error_code=exc.code,
                detail=exc.detail,
                retryable=True,
            )
        except (RegistryError, ValidationError) as exc:
            return _research_failure(
                raw_input,
                error_code="research_input_invalid",
                detail=f"input:{type(exc).__name__}:{exc}",
                retryable=False,
            )
        except Exception as exc:  # isolated service is the runtime failure boundary
            return _research_failure(
                raw_input,
                error_code="research_worker_unhandled_exception",
                detail=f"research:{type(exc).__name__}:{exc}",
                retryable=True,
            )
    try:
        settings = WorkerSettings.from_environment()
        batch_v2 = None
        if requested_schema == "2.0":
            batch_v2 = WorkerInputV2.model_validate(raw_input)
            validate_v2_internal_urls(batch_v2, settings.allowed_media_hosts)
            batch = to_legacy_input(batch_v2)
        else:
            batch = WorkerInput.model_validate(raw_input)
            validate_internal_urls(batch.items, settings.allowed_media_hosts)
        registry = build_model_group_registry()
        adapter_factory = factory or _runtime_factory(settings)
        runner = SessionRunner(
            registry=registry,
            adapter_factory=adapter_factory,
            boot_ms=BOOT_READY_MS,
        )
        if not _GPU_SESSION_LOCK.acquire(blocking=False):
            busy: dict[str, Any] = {
                "schema_version": "2.0" if batch_v2 is not None else "1.0",
                "batch_id": batch.batch_id,
                "status": "failed",
                "retryable": True,
                "model_runs": [],
                "items": [],
                "validation_errors": ["worker:gpu_session_already_active"],
                "boot_ms": BOOT_READY_MS,
            }
            if batch_v2 is not None:
                busy.update(
                    {
                        "analysis_id": batch_v2.analysis_window.analysis_id,
                        "orchestration_contract_digest": runner.contracts.digest,
                        "stage_traces": [],
                        "report_draft": None,
                    }
                )
            return busy
        try:
            with adapter_factory_job_scope(adapter_factory):
                execution = runner.run_with_trace(batch)
                if batch_v2 is not None:
                    from fireviewer_vision_runtime.v2_burned_area import run_burned_area_stage
                    from fireviewer_vision_runtime.v2_pointing import run_fire_pointing_stage

                    pointing_execution = run_fire_pointing_stage(
                        batch_v2,
                        execution.output,
                        adapter=_runtime_fire_pointing_adapter(adapter_factory),
                        sequence=len(execution.stage_traces) + 1,
                    )
                    burned_area_execution = run_burned_area_stage(
                        batch_v2,
                        adapter=_runtime_burned_area_adapter(adapter_factory),
                        sequence=len(execution.stage_traces) + 2,
                    )
                    resolved_spatial_pipeline = spatial_pipeline or _runtime_spatial_pipeline(
                        adapter_factory
                    )
                    return from_legacy_output(
                        batch_v2,
                        execution.output,
                        stage_traces=execution.stage_traces,
                        candidate_runs=execution.candidate_runs,
                        consensus_results=execution.consensus_results,
                        contract_digest=execution.contract_digest,
                        fire_pointing_execution=pointing_execution,
                        burned_area_execution=burned_area_execution,
                        spatial_pipeline=resolved_spatial_pipeline,
                    ).model_dump(mode="json")
            return execution.output.model_dump(mode="json")
        finally:
            _GPU_SESSION_LOCK.release()
    except (ConfigurationError, OutputValidationError, RegistryError, ValidationError) as exc:
        batch_id = raw_input.get("batch_id", "INVALID")
        failed: dict[str, Any] = {
            "schema_version": "2.0" if requested_schema == "2.0" else "1.0",
            "batch_id": batch_id if isinstance(batch_id, str) else "INVALID",
            "status": "failed",
            "retryable": False,
            "model_runs": [],
            "items": [],
            "validation_errors": [f"input:{type(exc).__name__}:{exc}"],
            "boot_ms": BOOT_READY_MS,
        }
        if requested_schema == "2.0":
            window = raw_input.get("analysis_window")
            failed["analysis_id"] = (
                window.get("analysis_id", "INVALID") if isinstance(window, dict) else "INVALID"
            )
            failed["orchestration_contract_digest"] = load_stage_contract_registry().digest
            failed["stage_traces"] = []
            failed["report_draft"] = None
        return failed


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handle_job})


if __name__ == "__main__":
    main()
