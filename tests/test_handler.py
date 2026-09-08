from __future__ import annotations

from fireviewer_contracts.resources import schema_path

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fireviewer_vision_runtime.adapters import UnavailableAdapterFactory
from fireviewer_contracts.contracts import ResearchOutputV1
from fireviewer_orchestrator.handler import handle_job

EXAMPLES = schema_path("agent-worker/v2/examples/valid-input.json").parent


class ScopedUnavailableFactory(UnavailableAdapterFactory):
    def __init__(self) -> None:
        super().__init__()
        self.entered = 0
        self.exited = 0

    @contextmanager
    def job_scope(self) -> Iterator[None]:
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1


def test_handler_fails_closed_when_input_is_missing() -> None:
    result = handle_job({})
    assert result["status"] == "failed"
    assert result["retryable"] is False
    assert result["items"] == []


def test_handler_rejects_an_external_media_url(monkeypatch) -> None:
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    result = handle_job(
        {
            "input": {
                "batch_id": "BATCH-1",
                "batch_type": "user_media",
                "priority": "user_deadline",
                "items": [
                    {
                        "input_id": "INPUT-1",
                        "media_type": "image",
                        "working_file_url": "https://example.org/image.jpg",
                    }
                ],
            }
        }
    )
    assert result["status"] == "failed"
    assert "not allowed" in result["validation_errors"][0]


def test_handler_reports_invalid_research_input_without_hiding_its_failure() -> None:
    result = handle_job(
        {
            "input": {
                "schema_version": "research-1.0",
                "research_id": "research-invalid-0001",
            }
        }
    )

    assert result["status"] == "failed"
    assert result["retryable"] is False
    assert result["model_run"]["status"] == "failed"
    assert result["model_run"]["error_code"] == "research_input_invalid"


def test_handler_keeps_v2_stage_traces_at_the_transport_boundary(monkeypatch) -> None:
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    payload = json.loads((EXAMPLES / "valid-input.json").read_text(encoding="utf-8"))

    result = handle_job({"input": payload}, factory=UnavailableAdapterFactory())

    assert result["schema_version"] == "2.0"
    assert len(result["orchestration_contract_digest"]) == 64
    assert [trace["stage_role"] for trace in result["stage_traces"]] == [
        "asr",
        "fire_detection",
        "visual_grounding",
        "multimodal_extraction",
        "fire_pointing",
        "burned_area",
        "evidence_fusion",
        "situation_report",
    ]


def test_handler_closes_the_adapter_factory_job_scope(monkeypatch) -> None:
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    factory = ScopedUnavailableFactory()
    payload = {
        "batch_id": "BATCH-SCOPE",
        "batch_type": "user_media",
        "priority": "user_deadline",
        "items": [
            {
                "input_id": "IMAGE-1",
                "media_type": "image",
                "working_file_url": "https://media.internal/image.jpg",
            }
        ],
    }

    handle_job({"input": payload}, factory=factory)

    assert factory.entered == 1
    assert factory.exited == 1


def test_handler_routes_research_contract_to_isolated_service(monkeypatch) -> None:
    now = datetime.now(UTC)
    cutoff = now + timedelta(hours=1)
    captured = []

    def fake_isolated(research):
        captured.append(research)
        return ResearchOutputV1.model_validate(
            {
                "research_id": research.research_id,
                "status": "succeeded",
                "retryable": False,
                "model_run": {
                    "model_id": "Qwen/Qwen3-14B",
                    "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
                    "status": "succeeded",
                    "started_at": now,
                    "finished_at": now,
                    "load_ms": 1,
                    "inference_ms": 1,
                },
            }
        )

    monkeypatch.setattr(
        "fireviewer_evidence_ingestion.research_client.run_isolated_research",
        fake_isolated,
    )
    result = handle_job(
        {
            "input": {
                "schema_version": "research-1.0",
                "operation": "source_research",
                "research_id": "research-synthetic-0001",
                "analysis_window": {
                    "analysis_id": "analysis-synthetic-0001",
                    "fire_id": "FR-99-00001",
                    "episode_id": "E01",
                    "window_start_at": now.isoformat(),
                    "window_end_at": cutoff.isoformat(),
                    "local_date": now.date().isoformat(),
                    "timezone": "Europe/Paris",
                },
                "incident_name": "Incident synthétique",
                "incident_reference": [5.37, 44.75],
                "cutoff_at": cutoff.isoformat(),
                "location_hint": "Commune fictive, secteur de démonstration",
                "source_registry_version": "firewarning-fr-sources-2026-07-19-v1",
                "allowed_domains": ["authority.example"],
                "source_policies": {
                    "authority.example": {
                        "source_name": "Autorité de démonstration",
                        "kind": "authority",
                        "scope": "local",
                        "confidence_level": "A+",
                        "claim_types": [
                            "operational_confirmation",
                            "local_instruction",
                        ],
                        "publication_policy": "per_item_license_check",
                        "minimum_refresh_minutes": 10,
                    }
                },
                "search_templates": {
                    "search.example": "https://search.example/recherche?q={query}"
                },
                "max_fetch_bytes": 1_048_576,
                "request_timeout_seconds": 20,
                "private_upload": {
                    "pathname_prefix": "firewarning/source-packages/upload-test",
                    "upload_grant": "g" * 128,
                    "token_endpoint": "https://fireviewer.example/api/v1/admin/blob-upload-token",
                    "resource_id": "research-synthetic-0001",
                    "maximum_file_size_bytes": 10_485_760,
                    "allowed_content_types": ["image/jpeg", "text/html"],
                },
            }
        }
    )

    assert result["schema_version"] == "research-1.0"
    assert result["status"] == "succeeded"
    assert len(captured) == 1
    assert captured[0].private_upload.upload_grant == "g" * 128
