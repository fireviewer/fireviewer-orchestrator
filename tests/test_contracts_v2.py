from __future__ import annotations

from fireviewer_contracts.resources import schema_path

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from fireviewer_contracts.contracts import (
    ReportSectionV2,
    SourceAnnotationV2,
    SpatialProposalV2,
    WorkerInputV2,
    WorkerOutputV2,
)
from fireviewer_orchestrator.v2_runner import _hotspot_spatial_proposals, to_legacy_input

EXAMPLES = schema_path("agent-worker/v2/examples/valid-input.json").parent


def _example(name: str) -> dict[str, object]:
    value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_worker_v2_accepts_the_shared_backend_examples() -> None:
    worker_input = WorkerInputV2.model_validate(_example("valid-input.json"))
    worker_output = WorkerOutputV2.model_validate(_example("valid-output.json"))

    assert worker_input.model_dump(mode="json")["schema_version"] == "2.0"
    assert worker_output.model_dump(mode="json")["analysis_id"] == "ANALYSIS-SYNTH-2026-07-09"
    assert worker_output.items[0].requires_human_review is True


def test_worker_v2_is_closed() -> None:
    payload = _example("valid-input.json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WorkerInputV2.model_validate(payload)


def test_worker_v2_requires_complete_camera_orientation() -> None:
    payload = _example("valid-input.json")
    items = payload["items"]
    assert isinstance(items, list)
    assert isinstance(items[0], dict)
    camera = items[0]["camera"]
    assert isinstance(camera, dict)
    camera.pop("roll_deg")

    with pytest.raises(ValidationError, match="yaw, pitch, and roll"):
        WorkerInputV2.model_validate(payload)


def test_worker_v2_abstention_requires_a_reason() -> None:
    with pytest.raises(ValidationError, match="requires an uncertainty code"):
        SpatialProposalV2.model_validate(
            {
                "proposal_id": "SP-ABSTAIN",
                "annotation_id": "ANN-1",
                "status": "insufficient_geometry",
            }
        )


def test_worker_v2_supports_front_geometry_without_a_fake_point() -> None:
    annotation = SourceAnnotationV2.model_validate(
        {
            "annotation_id": "ANN-FRONT",
            "evidence_id": "IMAGE-1",
            "evidence_kind": "image",
            "semantic_anchor": "visible_fire_front",
            "source_geometry_normalized": {
                "type": "LineString",
                "coordinates": [[0.15, 0.8], [0.5, 0.7], [0.85, 0.75]],
            },
        }
    )
    proposal = SpatialProposalV2.model_validate(
        {
            "proposal_id": "SP-FRONT",
            "annotation_id": "ANN-FRONT",
            "status": "projected_geometry",
            "proposal_kind": "visible_fire_front",
            "observed_at": "2026-07-12T15:00:00+02:00",
            "geometry_origin": "CROSS_VIEW_RAYCAST",
            "geometry_geojson": {
                "type": "LineString",
                "coordinates": [[2.65, 48.39], [2.66, 48.395]],
            },
            "horizontal_accuracy_m": 120,
            "reference_bundle_sha256": "b" * 64,
        }
    )

    assert annotation.source_point_normalized is None
    assert proposal.proposal_kind == "visible_fire_front"


def test_worker_v2_abstention_does_not_require_a_source_annotation() -> None:
    proposal = SpatialProposalV2.model_validate(
        {
            "proposal_id": "SP-NO-ANCHOR",
            "status": "insufficient_geometry",
            "uncertainty_codes": ["anchor_not_visible"],
        }
    )

    assert proposal.annotation_id is None


def test_worker_v2_limitations_can_use_an_explicit_abstention_basis() -> None:
    section = ReportSectionV2.model_validate(
        {
            "key": "limitations",
            "heading": "Limites",
            "body": "La pose caméra ne permet pas une projection fiable.",
            "basis_codes": ["camera_pose_missing"],
        }
    )

    assert section.fact_ids == ()


def test_worker_v2_report_references_are_closed() -> None:
    payload = _example("valid-output.json")
    report = payload["report_draft"]
    assert isinstance(report, dict)
    sections = report["sections"]
    assert isinstance(sections, list)
    assert isinstance(sections[0], dict)
    sections[0]["fact_ids"] = ["FACT-NOT-PRESENT"]

    with pytest.raises(ValidationError, match="unknown fact"):
        WorkerOutputV2.model_validate(payload)


def test_worker_v2_stage_trace_sequences_cannot_overlap() -> None:
    payload = _example("valid-output.json")
    traces = payload["stage_traces"]
    assert isinstance(traces, list)
    duplicate = deepcopy(traces[0])
    assert isinstance(duplicate, dict)
    duplicate["stage_role"] = "asr"
    duplicate["contract_id"] = "stage.asr.v1"
    traces.append(duplicate)

    with pytest.raises(ValidationError, match="duplicate stage sequences"):
        WorkerOutputV2.model_validate(payload)


def test_satellite_hotspot_geojson_keeps_explicit_sensor_points() -> None:
    payload = _example("valid-input.json")
    payload["batch_type"] = "satellite_media"
    items = payload["items"]
    assert isinstance(items, list)
    items[0] = {
        "input_id": "INPUT-HOTSPOTS",
        "media_type": "satellite_data",
        "provenance": {
            "source_key": "NASA-FIRMS-2026-07-13",
            "source_reference_url": "https://firms.modaps.eosdis.nasa.gov/",
            "attribution": "NASA FIRMS",
            "license_identifier": "NASA-OPEN-DATA",
            "trust": "institutional",
        },
        "hotspot": {
            "product_id": "VIIRS-FONT-2026-07-13",
            "provider": "NASA FIRMS",
            "acquired_at": "2026-07-13T18:00:00Z",
            "sensor_names": ["VIIRS SNPP", "VIIRS NOAA20"],
            "resolution_m": 375,
            "bbox_wgs84": [2.46, 48.34, 2.72, 48.44],
        },
        "article_text": json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [2.61, 48.39],
                        },
                        "properties": {"sensor": "VIIRS SNPP"},
                    }
                ],
            }
        ),
    }
    batch = WorkerInputV2.model_validate(payload)

    legacy = to_legacy_input(batch)
    proposals = _hotspot_spatial_proposals(batch, batch.items[0])

    assert legacy.items[0].media_type.value == "article"
    assert len(proposals) == 1
    assert proposals[0].geometry_origin == "EXPLICIT_SOURCE_GEOMETRY"
    assert proposals[0].longitude == 2.61
    assert proposals[0].latitude == 48.39


def test_satellite_hotspot_geojson_rejects_points_outside_declared_product_bbox() -> None:
    payload = _example("valid-input.json")
    payload["batch_type"] = "satellite_media"
    items = payload["items"]
    assert isinstance(items, list)
    items[0] = {
        "input_id": "INPUT-HOTSPOTS",
        "media_type": "satellite_data",
        "provenance": {
            "source_key": "NASA-FIRMS-2026-07-13",
            "source_reference_url": "https://firms.modaps.eosdis.nasa.gov/",
            "attribution": "NASA FIRMS",
            "license_identifier": "NASA-OPEN-DATA",
            "trust": "institutional",
        },
        "hotspot": {
            "product_id": "VIIRS-FONT-2026-07-13",
            "provider": "NASA FIRMS",
            "acquired_at": "2026-07-13T18:00:00Z",
            "sensor_names": ["VIIRS SNPP"],
            "resolution_m": 375,
            "bbox_wgs84": [2.46, 48.34, 2.72, 48.44],
        },
        "article_text": json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [5.0, 45.0]},
                        "properties": {},
                    }
                ],
            }
        ),
    }
    batch = WorkerInputV2.model_validate(payload)

    proposals = _hotspot_spatial_proposals(batch, batch.items[0])

    assert len(proposals) == 1
    assert proposals[0].status == "insufficient_geometry"
    assert proposals[0].uncertainty_codes == ("hotspot_observations_empty",)
