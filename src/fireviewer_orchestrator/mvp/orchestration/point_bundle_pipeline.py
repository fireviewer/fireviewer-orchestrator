from __future__ import annotations

from datetime import datetime
from hashlib import sha256

from fireviewer_contracts.mvp.contracts import (
    EventEvidenceV1,
    GeographicHypothesis,
    GeographicHypothesisResultV1,
    GeographicReference,
    GeospatialConsistencyCheck,
    LocationCandidate,
    PointEvidenceBundleV1,
)
from fireviewer_geolocation.mvp.localization.geographic_endpoint import (
    DurableGeographicHypothesisService,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import DurableEventEvidence
from fireviewer_evidence_supervisor.mvp.supervision.point_evidence import (
    PointEvidenceAssembler,
    canonical_model_sha256,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _candidate_for(
    hypothesis: GeographicHypothesis,
    result: GeographicHypothesisResultV1,
    event: EventEvidenceV1,
    *,
    rank: int,
) -> LocationCandidate:
    media_by_id = {item.media_id: item for item in event.media}
    media = media_by_id.get(hypothesis.media_id)
    if media is None:
        raise ValueError("geographic hypothesis references unknown EventEvidence media")
    geometric_consistency = (
        hypothesis.score_breakdown.camera_bearing
        + hypothesis.score_breakdown.terrain_visibility
    ) / 2
    return LocationCandidate(
        candidate_id=hypothesis.hypothesis_id,
        longitude=hypothesis.longitude,
        latitude=hypothesis.latitude,
        radius_m=hypothesis.horizontal_uncertainty_m,
        score=hypothesis.score,
        rank=rank,
        evidence_kind="geometric_verification",
        provider_id=result.provider_run.provider_id,
        provider_version=result.provider_run.provider_version,
        source_id=media.source_id,
        media_id=hypothesis.media_id,
        reference_id=hypothesis.observation_id,
        raw_score=hypothesis.score_breakdown.visual,
        temporal_consistency=hypothesis.score_breakdown.temporal_alignment,
        geometric_consistency=geometric_consistency,
    )


def _supported_references(
    hypothesis: GeographicHypothesis,
    durable: DurableEventEvidence,
) -> tuple[GeographicReference, ...]:
    supported_ids = set(hypothesis.supporting_reference_ids)
    references = tuple(
        item
        for item in durable.geographic_references
        if item.reference_id in supported_ids
    )
    if {item.reference_id for item in references} != supported_ids:
        missing = supported_ids - {item.reference_id for item in references}
        raise ValueError(
            "geographic hypothesis has unresolved durable references: "
            + ", ".join(sorted(missing))
        )
    return references


def _checks_for(
    hypothesis: GeographicHypothesis,
    references: tuple[GeographicReference, ...],
    durable: DurableEventEvidence,
) -> tuple[GeospatialConsistencyCheck, ...]:
    common_ids = (
        hypothesis.hypothesis_id,
        hypothesis.observation_id,
        hypothesis.media_id,
    )
    checks = [
        GeospatialConsistencyCheck(
            check_id=_stable_id("CHECK-DISTANCE", hypothesis.hypothesis_id),
            check_type="camera_distance",
            status="supported",
            reason_code="camera_distance_within_geographic_gate",
            evidence_ids=common_ids,
        ),
        GeospatialConsistencyCheck(
            check_id=_stable_id("CHECK-BEARING", hypothesis.hypothesis_id),
            check_type="camera_bearing",
            status="supported",
            score=hypothesis.score_breakdown.camera_bearing,
            reason_code="visual_box_bearing_supported",
            evidence_ids=common_ids,
        ),
        GeospatialConsistencyCheck(
            check_id=_stable_id("CHECK-TERRAIN", hypothesis.hypothesis_id),
            check_type="terrain_visibility",
            status="supported",
            score=hypothesis.score_breakdown.terrain_visibility,
            reason_code="terrain_line_of_sight_supported",
            evidence_ids=common_ids,
        ),
    ]
    satellite_ids = tuple(
        item.reference_id
        for item in references
        if item.reference_kind in {"satellite_hotspot", "satellite_active_area"}
    )
    if not satellite_ids:
        raise ValueError("accepted geographic hypothesis lacks satellite provenance")
    satellite_evidence = tuple(dict.fromkeys((*common_ids, *satellite_ids)))
    checks.append(
        GeospatialConsistencyCheck(
            check_id=_stable_id("CHECK-SATELLITE", hypothesis.hypothesis_id),
            check_type="satellite_overlap",
            status="supported",
            score=hypothesis.score_breakdown.satellite,
            reason_code="satellite_reference_supported",
            evidence_ids=satellite_evidence,
        )
    )
    if hypothesis.score_breakdown.temporal_alignment is not None:
        checks.append(
            GeospatialConsistencyCheck(
                check_id=_stable_id("CHECK-TEMPORAL", hypothesis.hypothesis_id),
                check_type="temporal_alignment",
                status="supported",
                score=hypothesis.score_breakdown.temporal_alignment,
                reason_code="satellite_time_window_supported",
                evidence_ids=satellite_evidence,
            )
        )
    history_ids = tuple(
        item.reference_id
        for item in references
        if item.reference_kind
        in {"prior_active_point", "prior_fire_front", "prior_perimeter"}
    )
    if hypothesis.score_breakdown.history_progression is not None:
        history_evidence = tuple(
            dict.fromkeys(
                (
                    hypothesis.hypothesis_id,
                    *history_ids,
                    *(item.state_id for item in durable.prior_fire_states),
                )
            )
        )
        checks.append(
            GeospatialConsistencyCheck(
                check_id=_stable_id("CHECK-HISTORY", hypothesis.hypothesis_id),
                check_type="history_progression",
                status="supported",
                score=hypothesis.score_breakdown.history_progression,
                reason_code="history_progression_supported_as_non_absolute_prior",
                evidence_ids=history_evidence,
            )
        )
    return tuple(checks)


class GeographicPointBundlePipeline:
    """Create supervisor dossiers from GPS hypotheses without publishing geometry."""

    def __init__(
        self,
        geographic_service: DurableGeographicHypothesisService,
        *,
        assembler: PointEvidenceAssembler | None = None,
    ) -> None:
        self._geographic_service = geographic_service
        self._assembler = assembler or PointEvidenceAssembler()

    def build(
        self,
        event_id: str,
        *,
        generated_at: datetime,
        query_text: str = "preuves visuelles satellite géographiques historique du feu",
        max_context_documents: int = 12,
    ) -> tuple[GeographicHypothesisResultV1, tuple[PointEvidenceBundleV1, ...]]:
        durable, geographic_result = self._geographic_service.locate(event_id)
        if not geographic_result.hypotheses:
            return geographic_result, ()
        existing_ids = {
            item.candidate_id for item in durable.event.location_candidates
        }
        hypotheses = tuple(
            sorted(
                geographic_result.hypotheses,
                key=lambda item: (-item.score, item.hypothesis_id),
            )
        )
        if any(item.hypothesis_id in existing_ids for item in hypotheses):
            raise ValueError("geographic hypothesis collides with an existing candidate")
        candidates = tuple(
            _candidate_for(item, geographic_result, durable.event, rank=rank)
            for rank, item in enumerate(hypotheses, start=1)
        )
        event_payload = durable.event.model_dump(mode="json", by_alias=True)
        event_payload["location_candidates"] = [
            item.model_dump(mode="json")
            for item in (*durable.event.location_candidates, *candidates)
        ]
        event_with_hypotheses = EventEvidenceV1.model_validate(event_payload)
        bundles: list[PointEvidenceBundleV1] = []
        for hypothesis in hypotheses:
            references = _supported_references(hypothesis, durable)
            checks = _checks_for(hypothesis, references, durable)
            bundles.append(
                self._assembler.assemble(
                    event_with_hypotheses,
                    candidate_id=hypothesis.hypothesis_id,
                    upload_locations=tuple(
                        item
                        for item in durable.upload_locations
                        if item.media_id == hypothesis.media_id
                    ),
                    prior_fire_states=durable.prior_fire_states,
                    geographic_references=references,
                    geospatial_checks=checks,
                    generated_at=generated_at,
                    query_text=query_text,
                    max_context_documents=max_context_documents,
                    source_revision_sha256=durable.source_revision_sha256,
                    phenomenon=hypothesis.phenomenon,
                )
            )
        return geographic_result, tuple(bundles)

    def build_payload(
        self,
        event_id: str,
        *,
        generated_at: datetime,
        query_text: str = "preuves visuelles satellite géographiques historique du feu",
        max_context_documents: int = 12,
    ) -> dict[str, object]:
        geographic_result, bundles = self.build(
            event_id,
            generated_at=generated_at,
            query_text=query_text,
            max_context_documents=max_context_documents,
        )
        return {
            "schema": "fireviewer.point-evidence-bundle-batch.v1",
            "event_id": event_id,
            "source_geographic_result_sha256": canonical_model_sha256(
                geographic_result
            ),
            "geographic_status": geographic_result.status,
            "bundles": [
                item.model_dump(mode="json", by_alias=True) for item in bundles
            ],
            "abstentions": [
                item.model_dump(mode="json")
                for item in geographic_result.abstentions
            ],
            "needs_human_review": True,
            "geometry_mutation_allowed": False,
        }


__all__ = ["GeographicPointBundlePipeline"]
