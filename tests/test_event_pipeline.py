from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy

import pytest
from pydantic import ValidationError

from fireviewer_vision_runtime.adapters import UnavailableAdapterFactory
from fireviewer_vision_runtime.event_perception import EventPerceptionPoint
from fireviewer_orchestrator.event_pipeline import (
    ActivityEnvelopeCandidate,
    DeterministicEventPipeline,
    EventCandidateBundle,
    EventPipelineInput,
    LocalizationAttempt,
    LocalizationStatus,
    PhenomenonKind,
    SemanticRole,
    source_can_seed_private_incident,
    validate_activity_envelope_supports,
)
from fireviewer_orchestrator.handler import _GPU_SESSION_LOCK, handle_job
from fireviewer_contracts.model_registry import ModelSpec


class _EventPointingAdapter:
    spec = ModelSpec(
        role="fire_pointing",
        model_id="tests/fire-pointing-fixture",
        revision="0000000000000000000000000000000000000000",
    )

    def __init__(
        self,
        *,
        points: tuple[EventPerceptionPoint, ...] = (
            EventPerceptionPoint(
                semantic_anchor="active_fire_point",
                source_point_normalized=(0.75, 0.5),
                model_score=0.91,
            ),
        ),
        fail_load: bool = False,
        fail_infer: bool = False,
    ) -> None:
        self.points = points
        self.fail_load = fail_load
        self.fail_infer = fail_infer
        self.loaded = 0
        self.unloaded = 0
        self.inferred: list[tuple[str, str]] = []

    def load(self) -> None:
        self.loaded += 1
        if self.fail_load:
            raise RuntimeError("synthetic load failure")

    def infer_event_image(
        self,
        *,
        evidence_asset_id: str,
        working_file_url: str,
    ) -> tuple[EventPerceptionPoint, ...]:
        self.inferred.append((evidence_asset_id, working_file_url))
        if self.fail_infer:
            raise RuntimeError("synthetic inference failure")
        return self.points

    def unload(self) -> None:
        self.unloaded += 1


class _ScopedEventFactory(UnavailableAdapterFactory):
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


def _bundle(*, scale: str = "wide", yaw: float | None = None) -> dict[str, object]:
    viewpoint: dict[str, object] = {
        "longitude": 5.39,
        "latitude": 43.29,
        "horizontal_accuracy_m": 12,
        "origin": "USER_PLACED",
    }
    if yaw is not None:
        viewpoint.update({"yaw_deg": yaw, "fov_deg": 60})
    return {
        "schema_version": "event-2.0",
        "candidate_id": "EC-1",
        "incident_candidate_id": "IC-1",
        "viewpoint": viewpoint,
        "observed_time": {"start_at": "2026-08-03T12:00:00+02:00"},
        "shot_scale": scale,
        "message": "Flammes visibles au-delà de la crête.",
        "evidence_assets": [],
        "consent": {
            "analysis": True,
            "retention": True,
            "public_derivative": False,
        },
        "provenance": {
            "received_at": "2026-08-03T12:01:00+02:00",
            "idempotency_key": "idem-1",
        },
    }


def _request(*, scale: str = "wide", yaw: float | None = None) -> dict[str, object]:
    bundle = _bundle(scale=scale, yaw=yaw)
    bundle["evidence_assets"] = [
        {
            "evidence_asset_id": "ASSET-1",
            "kind": "image",
            "sha256": "a" * 64,
            "object_uri": "blob://private/asset-1",
            "declared_media_type": "image/jpeg",
            "size_bytes": 1_024,
        }
    ]
    return {
        "schema_version": "event-2.0",
        "bundle": bundle,
        "perception_anchors": [
            {
                "anchor_id": "ANCHOR-1",
                "evidence_asset_id": "ASSET-1",
                "phenomenon": "active_fire_point",
                "source_point_normalized": [0.75, 0.5],
                "model_id": "fireviewer/detector",
                "model_revision": "immutable-revision",
                "model_score": 0.9,
            }
        ],
        "spatial_evidence": [],
    }


def _image_only_request(*, url: str = "https://media.internal/event.jpg") -> dict[str, object]:
    payload = _request(yaw=90)
    payload["perception_anchors"] = []
    payload["bundle"]["evidence_assets"][0]["working_file_url"] = url
    return payload


def test_message_only_candidate_is_valid_and_abstains_without_inventing_coordinates() -> None:
    bundle = EventCandidateBundle.model_validate(_bundle(scale="close"))
    result = DeterministicEventPipeline().run(
        EventPipelineInput(bundle=bundle, perception_anchors=(), spatial_evidence=())
    )

    assert result.view_profile == "ground_close_known_viewpoint"
    assert result.status == "abstained"
    assert result.localization_attempts[0].geometry_geojson is None
    assert result.localization_attempts[0].reason_codes == ("no_visual_anchor",)


def test_backend_bundle_may_defer_view_profile_classification() -> None:
    payload = _bundle()
    payload.pop("shot_scale")
    payload["provenance"]["trace_id"] = "TRACE-1"
    result = DeterministicEventPipeline().run(
        EventPipelineInput(
            bundle=EventCandidateBundle.model_validate(payload),
            perception_anchors=(),
            spatial_evidence=(),
        )
    )

    assert result.view_profile is None
    assert "view_profile_unclassified" in result.reason_codes


def test_viewpoint_is_never_treated_as_the_active_fire_point() -> None:
    request = EventPipelineInput.model_validate(_request(scale="tight"))
    result = DeterministicEventPipeline().run(request)

    assert result.status == "abstained"
    assert result.event_proposals == ()
    assert all(attempt.geometry_geojson is None for attempt in result.localization_attempts)
    assert result.view_profile == "ground_tight_known_viewpoint"


def test_direction_without_distance_produces_a_non_publishable_sector() -> None:
    request = EventPipelineInput.model_validate(_request(yaw=90))
    result = DeterministicEventPipeline().run(request)

    attempt = result.localization_attempts[0]
    assert result.status == "needs_review"
    assert attempt.status == "sector"
    assert attempt.sector is not None and attempt.sector.bearing_deg == 105
    assert attempt.sector.distance_max_m is None
    assert result.event_proposals == ()


def test_deterministic_raycast_can_propose_but_cross_view_stays_shadow() -> None:
    payload = _request()
    payload["spatial_evidence"] = [
        {
            "anchor_id": "ANCHOR-1",
            "status": "projected",
            "method": "camera_raycast",
            "geometry_geojson": {"type": "Point", "coordinates": [5.4, 43.3]},
            "horizontal_accuracy_m": 80,
            "reference_revision": "terrain-r1",
        }
    ]
    result = DeterministicEventPipeline().run(EventPipelineInput.model_validate(payload))
    assert result.event_proposals[0].geometry_geojson["coordinates"] == [5.4, 43.3]
    assert result.event_proposals[0].requires_human_review is True
    assert result.perception_anchors[0].anchor_id == "ANCHOR-1"
    assert result.spatial_evidence[0].reference_revision == "terrain-r1"

    shadow = deepcopy(payload)
    shadow["spatial_evidence"][0]["method"] = "cross_view_raycast"
    shadow_result = DeterministicEventPipeline().run(EventPipelineInput.model_validate(shadow))
    assert shadow_result.event_proposals == ()
    assert shadow_result.localization_attempts[0].shadow_only is True
    assert shadow_result.localization_attempts[0].reason_codes == ("cross_view_shadow_only",)


def test_external_mirrors_count_as_one_family_and_contradictions_are_preserved() -> None:
    payload = _request()
    payload["bundle"]["external_observations"] = [
        {
            "observation_id": "FIRMS-1",
            "artifact_revision_id": "AR-1",
            "lineage_family_id": "VIIRS-GRANULE-1",
            "semantic_role": "sensor_detection",
            "phenomenon": "thermal_hotspot",
            "conflicts_with": ["GROUND-1"],
        },
        {
            "observation_id": "EFFIS-1",
            "artifact_revision_id": "AR-2",
            "lineage_family_id": "VIIRS-GRANULE-1",
            "semantic_role": "sensor_detection",
            "phenomenon": "thermal_hotspot",
        },
        {
            "observation_id": "GROUND-1",
            "artifact_revision_id": "AR-3",
            "lineage_family_id": "GROUND-FAMILY-1",
            "semantic_role": "interpreted_observation",
            "phenomenon": "active_fire_point",
        },
    ]
    result = DeterministicEventPipeline().run(EventPipelineInput.model_validate(payload))

    assert result.independent_external_families == (
        "GROUND-FAMILY-1",
        "VIIRS-GRANULE-1",
    )
    assert result.contradictions == (("FIRMS-1", "GROUND-1"),)


def test_only_official_statement_can_seed_a_private_incident_candidate() -> None:
    assert source_can_seed_private_incident(SemanticRole.OFFICIAL_INCIDENT_STATEMENT)
    assert not source_can_seed_private_incident(SemanticRole.SENSOR_DETECTION)
    assert not source_can_seed_private_incident(SemanticRole.WEATHER_FORECAST)


def _localized(attempt_id: str, phenomenon: PhenomenonKind) -> LocalizationAttempt:
    return LocalizationAttempt(
        attempt_id=attempt_id,
        phenomenon=phenomenon,
        status=LocalizationStatus.LOCALIZED,
        method="camera_raycast",
        geometry_geojson={"type": "Point", "coordinates": [5.4, 43.3]},
        horizontal_accuracy_m=50,
    )


def test_smoke_alone_and_non_active_products_cannot_close_an_envelope() -> None:
    envelope = ActivityEnvelopeCandidate(
        geometry_geojson={
            "type": "Polygon",
            "coordinates": [[[5.3, 43.2], [5.5, 43.2], [5.5, 43.4], [5.3, 43.2]]],
        },
        support_attempt_ids=("LOC-1", "LOC-2"),
    )
    smoke = (
        _localized("LOC-1", PhenomenonKind.SMOKE_ORIGIN),
        _localized("LOC-2", PhenomenonKind.SMOKE_ORIGIN),
    )
    with pytest.raises(ValueError, match="smoke-only"):
        validate_activity_envelope_supports(envelope, smoke)

    burned = (
        _localized("LOC-1", PhenomenonKind.ACTIVE_FIRE_POINT),
        _localized("LOC-2", PhenomenonKind.BURNED_AREA),
    )
    with pytest.raises(ValueError, match="cannot support"):
        validate_activity_envelope_supports(envelope, burned)


def test_contract_rejects_zero_message_and_zero_media() -> None:
    payload = _bundle()
    payload["message"] = None
    with pytest.raises(ValidationError, match="message or at least one media"):
        EventCandidateBundle.model_validate(payload)


def test_handler_is_fail_closed_behind_feature_flag(monkeypatch) -> None:
    request = _request()
    monkeypatch.delenv("FV_AGENT_EVENT_PIPELINE_ENABLED", raising=False)
    disabled = handle_job({"input": request})
    assert disabled["status"] == "failed"
    assert disabled["reason_codes"] == ["event_pipeline_disabled"]

    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    enabled = handle_job({"input": request})
    assert enabled["status"] == "abstained"


def test_handler_validation_error_does_not_echo_private_message(monkeypatch) -> None:
    request = _request()
    request["bundle"]["message"] = "PRIVATE CONTRIBUTOR TEXT"
    request["bundle"]["viewpoint"] = []
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")

    result = handle_job({"input": request})

    assert result["status"] == "failed"
    assert result["candidate_id"] == "EC-1"
    assert "PRIVATE CONTRIBUTOR TEXT" not in " ".join(result["reason_codes"])


def test_handler_runs_real_image_bridge_inside_gpu_and_factory_scope(monkeypatch) -> None:
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    adapter = _EventPointingAdapter()
    factory = _ScopedEventFactory()

    result = handle_job(
        {"input": _image_only_request()},
        factory=factory,
        event_perception_adapter=adapter,
    )

    assert result["status"] == "needs_review"
    assert result["event_proposals"] == []
    assert len(result["localization_attempts"]) == 1
    attempt = result["localization_attempts"][0]
    assert attempt["status"] == "sector"
    assert attempt["geometry_geojson"] is None
    assert attempt["sector"]["bearing_deg"] == 105
    assert attempt["model_id"] == adapter.spec.model_id
    assert attempt["model_revision"] == adapter.spec.revision
    assert len(result["perception_anchors"]) == 1
    assert result["perception_anchors"][0]["evidence_asset_id"] == "ASSET-1"
    assert result["perception_anchors"][0]["anchor_id"] == attempt["anchor_id"]
    assert result["spatial_evidence"] == []
    assert adapter.inferred == [("ASSET-1", "https://media.internal/event.jpg")]
    assert (adapter.loaded, adapter.unloaded) == (1, 1)
    assert (factory.entered, factory.exited) == (1, 1)
    assert "fire_id" not in result
    assert "episode_id" not in result


@pytest.mark.parametrize(
    ("url", "reason_code"),
    [
        ("https://evil.example/event.jpg", "media_url_not_allowed"),
        ("http://media.internal/event.jpg", "media_url_not_allowed"),
        ("https://media.internal:444/event.jpg", "media_url_not_allowed"),
        ("not-a-url", "media_url_not_allowed"),
    ],
)
def test_event_image_bridge_rejects_non_allowlisted_https_url_without_loading_model(
    monkeypatch,
    url: str,
    reason_code: str,
) -> None:
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    adapter = _EventPointingAdapter()

    result = handle_job(
        {"input": _image_only_request(url=url)},
        factory=_ScopedEventFactory(),
        event_perception_adapter=adapter,
    )

    assert result["status"] == "abstained"
    assert reason_code in result["reason_codes"]
    assert result["event_proposals"] == []
    assert all(item["geometry_geojson"] is None for item in result["localization_attempts"])
    assert adapter.loaded == 0
    assert adapter.inferred == []


@pytest.mark.parametrize(
    ("fail_load", "fail_infer", "reason_code"),
    [
        (True, False, "fire_pointing_model_unavailable"),
        (False, True, "fire_pointing_inference_failed"),
    ],
)
def test_event_image_bridge_model_failure_abstains_without_coordinates(
    monkeypatch,
    fail_load: bool,
    fail_infer: bool,
    reason_code: str,
) -> None:
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    adapter = _EventPointingAdapter(fail_load=fail_load, fail_infer=fail_infer)

    result = handle_job(
        {"input": _image_only_request()},
        factory=_ScopedEventFactory(),
        event_perception_adapter=adapter,
    )

    assert result["status"] == "abstained"
    assert reason_code in result["reason_codes"]
    assert result["event_proposals"] == []
    assert all(item["geometry_geojson"] is None for item in result["localization_attempts"])
    failure = next(
        item for item in result["localization_attempts"] if reason_code in item["reason_codes"]
    )
    assert failure["model_id"] == adapter.spec.model_id
    assert failure["model_revision"] == adapter.spec.revision


def test_event_video_without_frames_abstains_before_model_load(monkeypatch) -> None:
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    payload = _image_only_request(url="https://media.internal/event.mp4")
    payload["bundle"]["evidence_assets"][0].update(
        {"kind": "video", "declared_media_type": "video/mp4"}
    )
    adapter = _EventPointingAdapter()

    result = handle_job(
        {"input": payload},
        factory=_ScopedEventFactory(),
        event_perception_adapter=adapter,
    )

    assert result["status"] == "abstained"
    assert "video_frames_missing" in result["reason_codes"]
    assert result["event_proposals"] == []
    assert adapter.loaded == 0


def test_event_image_bridge_reports_busy_gpu_without_running_model(monkeypatch) -> None:
    monkeypatch.setenv("FV_AGENT_EVENT_PIPELINE_ENABLED", "true")
    monkeypatch.setenv("FW_ALLOWED_MEDIA_HOSTS", "media.internal")
    adapter = _EventPointingAdapter()
    assert _GPU_SESSION_LOCK.acquire(blocking=False)
    try:
        result = handle_job(
            {"input": _image_only_request()},
            factory=_ScopedEventFactory(),
            event_perception_adapter=adapter,
        )
    finally:
        _GPU_SESSION_LOCK.release()

    assert result["status"] == "abstained"
    assert "gpu_session_already_active" in result["reason_codes"]
    assert result["event_proposals"] == []
    assert adapter.loaded == 0


def test_point_front_does_not_create_an_invalid_linestring_proposal() -> None:
    payload = _request()
    payload["perception_anchors"][0]["phenomenon"] = "visible_fire_front"
    payload["spatial_evidence"] = [
        {
            "anchor_id": "ANCHOR-1",
            "status": "projected",
            "method": "camera_raycast",
            "geometry_geojson": {"type": "Point", "coordinates": [5.4, 43.3]},
            "horizontal_accuracy_m": 80,
            "reference_revision": "pose-and-terrain-r1",
        }
    ]

    result = DeterministicEventPipeline().run(EventPipelineInput.model_validate(payload))

    assert result.status == "needs_review"
    assert result.event_proposals == ()
    assert result.localization_attempts[0].status == "localized"
    assert result.localization_attempts[0].reason_codes == ("front_geometry_insufficient",)


def test_raycast_without_reference_abstains_instead_of_publishing_geometry() -> None:
    payload = _request()
    payload["spatial_evidence"] = [
        {
            "anchor_id": "ANCHOR-1",
            "status": "projected",
            "method": "camera_raycast",
            "geometry_geojson": {"type": "Point", "coordinates": [5.4, 43.3]},
            "horizontal_accuracy_m": 80,
        }
    ]

    result = DeterministicEventPipeline().run(EventPipelineInput.model_validate(payload))

    assert result.status == "abstained"
    assert result.event_proposals == ()
    assert result.localization_attempts[0].geometry_geojson is None
    assert result.localization_attempts[0].reason_codes == ("camera_pose_or_reference_missing",)
