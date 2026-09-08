from __future__ import annotations

import os
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import fmod
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from fireviewer_contracts.event_models import BundleProvenance, EventCandidateBundle, EventConsent, EventModel, EventPipelineInput, EventPipelineOutput, EvidenceAsset, EvidenceAssetKind, ExternalObservation, FireActivityProposal, LocalizationAttempt, LocalizationMethod, LocalizationStatus, ObservedTime, PerceptionAnchor, PerceptionFailure, PhenomenonKind, PipelineStatus, SectorEstimate, SemanticRole, ShotScale, SpatialEvidence, ViewProfile, Viewpoint, ViewpointOrigin

from fireviewer_contracts.geometry_contract import validate_geojson_geometry








Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]









































































ProposalPhenomenon = Literal[
    PhenomenonKind.ACTIVE_FIRE_POINT,
    PhenomenonKind.VISIBLE_FIRE_FRONT,
    PhenomenonKind.SMOKE_ORIGIN,
]





























































































































































































































































































def event_pipeline_enabled() -> bool:
    return os.getenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "false").strip().lower() == "true"


def classify_view_profile(bundle: EventCandidateBundle) -> ViewProfile | None:
    if bundle.shot_scale is None:
        return None
    if bundle.shot_scale == ShotScale.WIDE:
        if bundle.viewpoint.origin == ViewpointOrigin.NAMED_PLACE:
            return ViewProfile.GROUND_WIDE_NAMED_VIEWPOINT
        return ViewProfile.GROUND_WIDE_KNOWN_VIEWPOINT
    if bundle.shot_scale == ShotScale.DISTANT:
        return ViewProfile.GROUND_DISTANT_KNOWN_VIEWPOINT
    if bundle.shot_scale == ShotScale.CLOSE:
        return ViewProfile.GROUND_CLOSE_KNOWN_VIEWPOINT
    return ViewProfile.GROUND_TIGHT_KNOWN_VIEWPOINT


def source_can_seed_private_incident(role: SemanticRole) -> bool:
    """Only an official statement may seed a private matching dossier."""

    return role == SemanticRole.OFFICIAL_INCIDENT_STATEMENT


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _bearing(viewpoint: Viewpoint, anchor: PerceptionAnchor) -> SectorEstimate | None:
    if viewpoint.yaw_deg is None or viewpoint.fov_deg is None:
        return None
    if anchor.source_point_normalized is None:
        return None
    x, _ = anchor.source_point_normalized
    bearing = fmod(viewpoint.yaw_deg + (x - 0.5) * viewpoint.fov_deg + 360.0, 360.0)
    return SectorEstimate(
        bearing_deg=bearing,
        angular_uncertainty_deg=max(viewpoint.fov_deg * 0.05, 1.0),
    )


def _output_phenomenon(anchor: PerceptionAnchor) -> ProposalPhenomenon:
    if anchor.phenomenon == PhenomenonKind.SMOKE_COLUMN_BASE:
        return PhenomenonKind.SMOKE_ORIGIN
    if anchor.phenomenon == PhenomenonKind.VISIBLE_FIRE_FRONT:
        return PhenomenonKind.VISIBLE_FIRE_FRONT
    return PhenomenonKind.ACTIVE_FIRE_POINT


def _external_summary(
    observations: tuple[ExternalObservation, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    families = tuple(sorted({item.lineage_family_id for item in observations}))
    known_ids = {item.observation_id for item in observations}
    pairs: set[tuple[str, str]] = set()
    for item in observations:
        for other in item.conflicts_with:
            if other in known_ids and other != item.observation_id:
                first, second = sorted((item.observation_id, other))
                pairs.add((first, second))
    return families, tuple(sorted(pairs))


class DeterministicEventPipeline:
    """Turns evidence into review proposals without inventing spatial facts.

    Perception models only provide pixel anchors. Geographic geometry is accepted
    exclusively from deterministic spatial evidence. Cross-view remains shadow
    evidence until an independent benchmark authorizes its promotion.
    """

    def run(
        self,
        value: EventPipelineInput,
        *,
        perception_failures: tuple[PerceptionFailure, ...] = (),
    ) -> EventPipelineOutput:
        spatial_by_anchor = {item.anchor_id: item for item in value.spatial_evidence}
        attempts: list[LocalizationAttempt] = []
        proposals: list[FireActivityProposal] = []

        for anchor in value.perception_anchors:
            spatial = spatial_by_anchor.get(anchor.anchor_id)
            attempt_id = _stable_id("LOC", value.bundle.candidate_id, anchor.anchor_id)
            phenomenon = _output_phenomenon(anchor)
            if spatial is not None and spatial.status == "projected":
                if (
                    spatial.method
                    in {
                        LocalizationMethod.CAMERA_RAYCAST,
                        LocalizationMethod.CROSS_VIEW_RAYCAST,
                    }
                    and spatial.reference_revision is None
                ):
                    attempts.append(
                        LocalizationAttempt(
                            attempt_id=attempt_id,
                            anchor_id=anchor.anchor_id,
                            phenomenon=phenomenon,
                            status=LocalizationStatus.ABSTAINED,
                            method=spatial.method,
                            reason_codes=("camera_pose_or_reference_missing",),
                            model_id=anchor.model_id,
                            model_revision=anchor.model_revision,
                        )
                    )
                    continue
                if spatial.method == LocalizationMethod.CROSS_VIEW_RAYCAST:
                    attempts.append(
                        LocalizationAttempt(
                            attempt_id=attempt_id,
                            anchor_id=anchor.anchor_id,
                            phenomenon=phenomenon,
                            status=LocalizationStatus.ABSTAINED,
                            method=spatial.method,
                            reason_codes=("cross_view_shadow_only",),
                            model_id=anchor.model_id,
                            model_revision=anchor.model_revision,
                            reference_revision=spatial.reference_revision,
                            shadow_only=True,
                        )
                    )
                    continue
                assert spatial.geometry_geojson is not None
                assert spatial.horizontal_accuracy_m is not None
                geometry_type = spatial.geometry_geojson.get("type")
                front_geometry_insufficient = (
                    phenomenon == PhenomenonKind.VISIBLE_FIRE_FRONT
                    and geometry_type not in {"LineString", "MultiLineString"}
                )
                attempt = LocalizationAttempt(
                    attempt_id=attempt_id,
                    anchor_id=anchor.anchor_id,
                    phenomenon=phenomenon,
                    status=LocalizationStatus.LOCALIZED,
                    method=spatial.method,
                    geometry_geojson=spatial.geometry_geojson,
                    horizontal_accuracy_m=spatial.horizontal_accuracy_m,
                    direction_uncertainty_deg=spatial.direction_uncertainty_deg,
                    distance_uncertainty_m=spatial.distance_uncertainty_m,
                    model_id=anchor.model_id,
                    model_revision=anchor.model_revision,
                    reference_revision=spatial.reference_revision,
                    reason_codes=(
                        ("front_geometry_insufficient",) if front_geometry_insufficient else ()
                    ),
                )
                attempts.append(attempt)
                if front_geometry_insufficient:
                    # A single front point is valid pixel/localization evidence, but
                    # it cannot be promoted into a geometrically invalid front event.
                    continue
                proposals.append(
                    FireActivityProposal(
                        proposal_id=_stable_id("EVP", value.bundle.candidate_id, anchor.anchor_id),
                        attempt_id=attempt_id,
                        phenomenon=phenomenon,
                        observed_time=value.bundle.observed_time,
                        geometry_geojson=spatial.geometry_geojson,
                        horizontal_accuracy_m=spatial.horizontal_accuracy_m,
                    )
                )
                continue

            sector = _bearing(value.bundle.viewpoint, anchor)
            if sector is not None:
                attempts.append(
                    LocalizationAttempt(
                        attempt_id=attempt_id,
                        anchor_id=anchor.anchor_id,
                        phenomenon=phenomenon,
                        status=LocalizationStatus.SECTOR,
                        method=LocalizationMethod.VIEWPOINT_SECTOR,
                        sector=sector,
                        reason_codes=("distance_unknown",),
                        model_id=anchor.model_id,
                        model_revision=anchor.model_revision,
                    )
                )
            else:
                reason_codes = (
                    spatial.reason_codes
                    if spatial is not None
                    else ("direction_and_distance_missing",)
                )
                attempts.append(
                    LocalizationAttempt(
                        attempt_id=attempt_id,
                        anchor_id=anchor.anchor_id,
                        phenomenon=phenomenon,
                        status=LocalizationStatus.ABSTAINED,
                        reason_codes=reason_codes,
                        model_id=anchor.model_id,
                        model_revision=anchor.model_revision,
                    )
                )

        for failure in perception_failures:
            suffix = failure.evidence_asset_id or failure.reason_code
            attempts.append(
                LocalizationAttempt(
                    attempt_id=_stable_id(
                        "LOC",
                        value.bundle.candidate_id,
                        "perception-failure",
                        suffix,
                        failure.reason_code,
                    ),
                    status=LocalizationStatus.ABSTAINED,
                    reason_codes=(failure.reason_code,),
                    model_id=failure.model_id,
                    model_revision=failure.model_revision,
                )
            )

        if not attempts:
            attempts.append(
                LocalizationAttempt(
                    attempt_id=_stable_id("LOC", value.bundle.candidate_id, "no-anchor"),
                    status=LocalizationStatus.ABSTAINED,
                    reason_codes=("no_visual_anchor",),
                )
            )

        external_families, contradictions = _external_summary(value.bundle.external_observations)
        has_reviewable = bool(proposals) or any(
            attempt.status in {LocalizationStatus.LOCALIZED, LocalizationStatus.SECTOR}
            for attempt in attempts
        )
        status = PipelineStatus.NEEDS_REVIEW if has_reviewable else PipelineStatus.ABSTAINED
        reason_code_set = {code for attempt in attempts for code in attempt.reason_codes}
        view_profile = classify_view_profile(value.bundle)
        if view_profile is None:
            reason_code_set.add("view_profile_unclassified")
        reason_codes = tuple(sorted(reason_code_set))
        return EventPipelineOutput(
            candidate_id=value.bundle.candidate_id,
            status=status,
            view_profile=view_profile,
            perception_anchors=value.perception_anchors,
            spatial_evidence=value.spatial_evidence,
            localization_attempts=tuple(attempts),
            event_proposals=tuple(proposals),
            independent_external_families=external_families,
            contradictions=contradictions,
            reason_codes=reason_codes,
        )


class ActivityEnvelopeCandidate(EventModel):
    geometry_geojson: dict[str, Any]
    support_attempt_ids: tuple[Identifier, ...] = Field(min_length=2, max_length=512)

    @model_validator(mode="after")
    def polygon_only(self) -> ActivityEnvelopeCandidate:
        validate_geojson_geometry(
            self.geometry_geojson,
            allowed_types={"Polygon", "MultiPolygon"},
        )
        if len(set(self.support_attempt_ids)) != len(self.support_attempt_ids):
            raise ValueError("activity envelope support attempts must be unique")
        return self


def validate_activity_envelope_supports(
    candidate: ActivityEnvelopeCandidate,
    attempts: tuple[LocalizationAttempt, ...],
) -> None:
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    if not set(candidate.support_attempt_ids).issubset(by_id):
        raise ValueError("activity envelope references an unknown localization attempt")
    supports = [by_id[item] for item in candidate.support_attempt_ids]
    if any(item.status != LocalizationStatus.LOCALIZED for item in supports):
        raise ValueError("activity envelope supports must be localized")
    observed = {item.phenomenon for item in supports}
    if observed.issubset({PhenomenonKind.SMOKE_ORIGIN, PhenomenonKind.SMOKE_COLUMN_BASE}):
        raise ValueError("smoke-only evidence cannot close an activity envelope")
    forbidden = observed.intersection(
        {PhenomenonKind.THERMAL_HOTSPOT, PhenomenonKind.BURNED_AREA, PhenomenonKind.SIMULATION}
    )
    if forbidden:
        raise ValueError("hotspots, burned areas and simulations cannot support an active envelope")


def handle_event_analysis_payload(raw_input: dict[str, Any]) -> dict[str, Any]:
    request = EventPipelineInput.model_validate(raw_input)
    return DeterministicEventPipeline().run(request).model_dump(mode="json")
