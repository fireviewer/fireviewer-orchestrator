from datetime import UTC, datetime

from fireviewer_contracts.backend.incident_state_schemas import TemporalEvidence
from fireviewer_orchestrator.incident_updates import plan_incident_enrichment


def proof(**changes):
    return TemporalEvidence.model_validate({
        "evidence_id": "test-proof", "revision": 1, "incident_id": "FR-test", "source_id": "fixture",
        "content_sha256": "a" * 64, "license": "test", "media_kind": "image", "admissibility": "admitted",
        "observed_at": datetime(2026, 7, 1, tzinfo=UTC), "retrieved_at": datetime(2026, 7, 2, tzinfo=UTC),
        "recorded_at": datetime(2026, 7, 2, tzinfo=UTC), **changes,
    })


def test_image_waits_for_vision_and_geolocation_without_premature_fusion():
    steps = plan_incident_enrichment(proof())
    assert {step.component for step in steps if step.status == "required"} == {"vision", "geolocation"}
    assert steps[-1].component == "fusion" and steps[-1].status == "blocked"


def test_audio_requests_transcription_but_not_vision():
    steps = plan_incident_enrichment(proof(media_kind="audio"))
    assert {step.component for step in steps if step.status == "required"} == {"transcription", "geolocation"}


def test_withdrawal_never_runs_a_model():
    steps = plan_incident_enrichment(proof(admissibility="withdrawn"))
    assert len(steps) == 1 and steps[0].component == "fusion"


def test_unknown_time_cannot_trigger_a_speculative_geometry():
    steps = plan_incident_enrichment(proof(time_precision="unknown", observed_at=None))
    assert len(steps) == 1 and steps[0].status == "blocked"
