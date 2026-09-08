from __future__ import annotations

import pytest

from fireviewer_contracts.mvp.contracts import (
    EventEvidenceV1,
    LocationCandidate,
    Uncertainty,
    VisualObservation,
)
from fireviewer_orchestrator.mvp.orchestration import merge_event_evidence


def _base() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EVENT-MERGE-1",
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "Fixture",
                    "retrieved_at": "2026-08-21T10:00:00Z",
                    "source_type": "witness",
                    "independence_weight": 1,
                }
            ],
            "media": [
                {
                    "media_id": "MEDIA-1",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "photo",
                    "sha256": "a" * 64,
                }
            ],
        }
    )


def test_merge_event_evidence_combines_provider_outputs() -> None:
    base = _base()
    vision = EventEvidenceV1.model_validate(
        base.model_copy(
            update={
                "visual_observations": (
                    VisualObservation(
                        observation_id="OBS-1",
                        media_id="MEDIA-1",
                        observation_type="detection",
                        result_reference="GDN-1",
                        confidence=0.8,
                    ),
                )
            }
        )
    )
    localization = EventEvidenceV1.model_validate(
        base.model_copy(
            update={
                "location_candidates": (
                    LocationCandidate(
                        candidate_id="CANDIDATE-1",
                        longitude=5.3,
                        latitude=44.7,
                        score=0.7,
                        rank=1,
                        evidence_kind="visual_retrieval",
                        provider_id="megaloc-faiss",
                        provider_version="1.0.0",
                        media_id="MEDIA-1",
                    ),
                ),
                "uncertainties": (
                    Uncertainty(
                        uncertainty_id="UNC-1",
                        code="candidate_clusters_below_threshold",
                        scope_type="event",
                        scope_id="EVENT-MERGE-1",
                        description="Fixture uncertainty.",
                    ),
                ),
                "needs_human_review": True,
            }
        )
    )

    merged = merge_event_evidence(vision, localization)

    assert len(merged.visual_observations) == 1
    assert len(merged.location_candidates) == 1
    assert len(merged.uncertainties) == 1
    assert merged.needs_human_review is True


def test_merge_event_evidence_rejects_conflicting_identity_records() -> None:
    base = _base()
    payload = base.model_dump(mode="python")
    payload["media"][0]["sha256"] = "b" * 64
    conflicting = EventEvidenceV1.model_validate(payload)

    with pytest.raises(ValueError, match="conflicting media"):
        merge_event_evidence(base, conflicting)
