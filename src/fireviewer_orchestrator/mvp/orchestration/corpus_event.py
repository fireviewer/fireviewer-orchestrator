from __future__ import annotations

from datetime import datetime, time, timedelta
from hashlib import sha256
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.benchmarks.corpus import Summer2026EventCase
from fireviewer_contracts.mvp.contracts import (
    CandidateArea,
    Claim,
    EventEvidenceV1,
    EvidenceMedia,
    EvidenceSource,
    TimeWindow,
)
from fireviewer_geolocation.mvp.localization.evidence_fusion import haversine_m


class CorpusEventRuntimeInput(StrictModel):
    evidence: EventEvidenceV1
    relative_paths_by_media_id: dict[SafeIdentifierV2, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> CorpusEventRuntimeInput:
        media_ids = {item.media_id for item in self.evidence.media}
        if set(self.relative_paths_by_media_id) != media_ids:
            raise ValueError("runtime media paths must exactly cover the event media")
        return self


def _source_type(kind: str) -> Literal["official", "press", "social", "satellite"]:
    if kind == "press":
        return "press"
    if kind == "official_social":
        return "social"
    if kind == "copernicus_ems":
        return "satellite"
    return "official"


def _candidate_radius_km(case: Summer2026EventCase) -> float:
    center = case.collection_aoi.center_wgs84
    min_lon, min_lat, max_lon, max_lat = case.collection_aoi.bbox_wgs84
    return (
        max(
            haversine_m(center, corner)
            for corner in (
                (min_lon, min_lat),
                (min_lon, max_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
            )
        )
        / 1_000
    )


def prepare_corpus_event(case: Summer2026EventCase) -> CorpusEventRuntimeInput:
    """Convert one reviewed summer case into the production EventEvidence contract."""

    paris = ZoneInfo("Europe/Paris")
    started_at = datetime.combine(case.event_date, time.min, tzinfo=paris)
    source_ids = {source.source_id for source in case.sources}
    sources = tuple(
        EvidenceSource(
            source_id=source.source_id,
            origin_id=source.origin_id,
            source_url=source.url,
            publisher=source.publisher,
            retrieved_at=source.retrieved_at,
            source_type=_source_type(source.kind),
            independence_weight=1,
        )
        for source in case.sources
    )
    claims = tuple(
        Claim(
            claim_id=f"CLAIM-{sha256(f'{case.case_id}:{source.source_id}'.encode()).hexdigest()[:24]}",
            source_id=source.source_id,
            claim_type="event-context",
            text=source.claim_summary,
            observed_at=started_at,
            confidence=0.8 if source.kind != "press" else 0.65,
        )
        for source in case.sources
    )

    media: list[EvidenceMedia] = []
    paths: dict[SafeIdentifierV2, str] = {}
    for item in case.media:
        if item.relative_path is None or item.media_sha256 is None:
            raise ValueError("runtime event media must be materialized and digest-qualified")
        if item.source_id not in source_ids:
            raise ValueError("runtime event media references an unknown source")
        media_group_digest = sha256(f"{case.case_id}:{item.origin_id}".encode()).hexdigest()[:24]
        media.append(
            EvidenceMedia(
                media_id=item.media_id,
                source_id=item.source_id,
                media_group_id=f"GROUP-{media_group_digest}",
                origin_id=item.origin_id,
                kind="photo",
                sha256=item.media_sha256,
                captured_at=item.captured_at,
            )
        )
        paths[item.media_id] = item.relative_path

    evidence = EventEvidenceV1(
        event_id=case.case_id,
        time_window=TimeWindow(from_at=started_at, to_at=started_at + timedelta(days=1)),
        candidate_area=CandidateArea(
            center=case.collection_aoi.center_wgs84,
            radius_km=_candidate_radius_km(case),
            confidence=0.8 if case.collection_aoi.basis == "cems_activation_extent" else 0.65,
            name=case.label,
            supporting_source_ids=(case.collection_aoi.provenance_source_id,),
        ),
        sources=sources,
        claims=claims,
        media=tuple(media),
    )
    return CorpusEventRuntimeInput(
        evidence=evidence,
        relative_paths_by_media_id=paths,
    )


__all__ = ["CorpusEventRuntimeInput", "prepare_corpus_event"]
