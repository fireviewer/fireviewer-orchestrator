"""Evidence-driven enrichment planning. Durable requests and results belong to the backend."""

from fireviewer_contracts.backend.incident_state_schemas import (
    ProcessingStep,
    TemporalEvidence,
)


def plan_incident_enrichment(evidence: TemporalEvidence) -> tuple[ProcessingStep, ...]:
    if evidence.admissibility in {"withdrawn", "rejected"}:
        return (
            ProcessingStep(
                component="fusion", status="required", reason="replay_without_proof"
            ),
        )
    if evidence.enrichment_state in {"abstained", "failed"}:
        return evidence.processing or (
            ProcessingStep(
                component="fusion",
                status="blocked",
                reason=f"enrichment_{evidence.enrichment_state}",
            ),
        )
    if evidence.observed_at is None:
        return (
            ProcessingStep(
                component="fusion", status="blocked", reason="observation_time_unknown"
            ),
        )
    if evidence.observations:
        steps = [
            ProcessingStep(
                component="vision",
                status="not_applicable",
                reason="spatial_observations_available",
            ),
            ProcessingStep(
                component="geolocation",
                status="not_applicable",
                reason="spatial_observations_available",
            ),
        ]
    else:
        steps = []
        if evidence.media_kind in {"image", "video", "satellite"}:
            steps.append(
                ProcessingStep(
                    component="vision",
                    status="required",
                    reason="visual_observations_missing",
                )
            )
        elif evidence.media_kind == "audio":
            steps.append(
                ProcessingStep(
                    component="transcription",
                    status="required",
                    reason="audio_requires_text",
                )
            )
        steps.append(
            ProcessingStep(
                component="geolocation",
                status="required",
                reason="spatial_observations_missing",
            )
        )
    steps.append(
        ProcessingStep(
            component="supervisor",
            status="required" if evidence.needs_human_review else "not_applicable",
            reason="ambiguous_evidence"
            if evidence.needs_human_review
            else "no_ambiguity_declared",
        )
    )
    steps.append(
        ProcessingStep(
            component="fusion",
            status="required"
            if evidence.observations and evidence.admissibility == "admitted"
            else "blocked",
            reason="admitted_spatial_evidence"
            if evidence.observations and evidence.admissibility == "admitted"
            else "awaiting_admitted_spatial_evidence",
        )
    )
    return tuple(steps)
