from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fireviewer_contracts.mvp.contracts import EventEvidenceV1


def _merge_records(
    records: tuple[tuple[Any, ...], ...],
    *,
    identifier: Callable[[Any], str],
    label: str,
) -> tuple[Any, ...]:
    merged: dict[str, Any] = {}
    for collection in records:
        for item in collection:
            item_id = identifier(item)
            existing = merged.get(item_id)
            if existing is not None and existing != item:
                raise ValueError(f"conflicting {label} record {item_id}")
            merged[item_id] = item
    return tuple(merged[item_id] for item_id in sorted(merged))


def merge_event_evidence(*records: EventEvidenceV1) -> EventEvidenceV1:
    """Merge additive provider outputs for one immutable event evidence graph."""

    if not records:
        raise ValueError("at least one EventEvidence record is required")
    reference = records[0]
    for record in records[1:]:
        if record.event_id != reference.event_id:
            raise ValueError("EventEvidence records belong to different events")
        if record.time_window != reference.time_window:
            raise ValueError("EventEvidence records disagree on the event time window")
        if record.candidate_area != reference.candidate_area:
            raise ValueError("EventEvidence records disagree on the candidate area")

    return EventEvidenceV1.model_validate(
        reference.model_copy(
            update={
                "sources": _merge_records(
                    tuple(record.sources for record in records),
                    identifier=lambda item: item.source_id,
                    label="source",
                ),
                "claims": _merge_records(
                    tuple(record.claims for record in records),
                    identifier=lambda item: item.claim_id,
                    label="claim",
                ),
                "media": _merge_records(
                    tuple(record.media for record in records),
                    identifier=lambda item: item.media_id,
                    label="media",
                ),
                "visual_observations": _merge_records(
                    tuple(record.visual_observations for record in records),
                    identifier=lambda item: item.observation_id,
                    label="visual observation",
                ),
                "satellite_observations": _merge_records(
                    tuple(record.satellite_observations for record in records),
                    identifier=lambda item: item.observation_id,
                    label="satellite observation",
                ),
                "location_candidates": _merge_records(
                    tuple(record.location_candidates for record in records),
                    identifier=lambda item: item.candidate_id,
                    label="location candidate",
                ),
                "candidate_clusters": _merge_records(
                    tuple(record.candidate_clusters for record in records),
                    identifier=lambda item: item.cluster_id,
                    label="candidate cluster",
                ),
                "contradictions": _merge_records(
                    tuple(record.contradictions for record in records),
                    identifier=lambda item: item.contradiction_id,
                    label="contradiction",
                ),
                "uncertainties": _merge_records(
                    tuple(record.uncertainties for record in records),
                    identifier=lambda item: item.uncertainty_id,
                    label="uncertainty",
                ),
                "needs_human_review": any(record.needs_human_review for record in records),
            }
        )
    )


__all__ = ["merge_event_evidence"]
