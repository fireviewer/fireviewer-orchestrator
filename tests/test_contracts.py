from __future__ import annotations

import pytest
from pydantic import ValidationError

from fireviewer_contracts.contracts import (
    BatchItem,
    InputMetadata,
    ItemResult,
    LocationStatus,
    MediaType,
    MetadataResult,
    WorkerInput,
)
from fireviewer_contracts.validation import (
    OutputValidationError,
    validate_forbidden_output,
    validate_internal_urls,
    validate_item_result,
)


def test_worker_input_is_closed_and_coordinates_require_an_origin() -> None:
    with pytest.raises(ValidationError):
        WorkerInput.model_validate(
            {
                "batch_id": "BATCH-1",
                "batch_type": "user_media",
                "priority": "user_deadline",
                "items": [
                    {
                        "input_id": "INPUT-1",
                        "media_type": "image",
                        "working_file_url": "https://media.internal/image.jpg",
                        "metadata": {"latitude": 44.0, "longitude": 5.0},
                        "unexpected": True,
                    }
                ],
            }
        )


def test_only_configured_internal_https_hosts_are_accepted() -> None:
    item = BatchItem(
        input_id="INPUT-1",
        media_type=MediaType.IMAGE,
        working_file_url="https://external.example/image.jpg",
    )
    with pytest.raises(OutputValidationError, match="not allowed"):
        validate_internal_urls((item,), frozenset({"media.internal"}))


def test_forbidden_speculation_is_rejected() -> None:
    with pytest.raises(OutputValidationError, match="speculative"):
        validate_forbidden_output({"description": "Le feu pourrait atteindre la route."})


def test_an_unlocated_image_cannot_create_a_geographic_marker() -> None:
    source = BatchItem(
        input_id="INPUT-1",
        media_type=MediaType.IMAGE,
        working_file_url="https://media.internal/image.jpg",
    )
    result = ItemResult(
        input_id="INPUT-1",
        metadata_result=MetadataResult(capture_location_available=True),
        location_status=LocationStatus.CAPTURE_LOCATION_ONLY,
    )
    with pytest.raises(OutputValidationError, match="availability"):
        validate_item_result(source, result)


def test_metadata_gps_can_only_create_a_capture_marker() -> None:
    source = BatchItem(
        input_id="INPUT-1",
        media_type=MediaType.IMAGE,
        working_file_url="https://media.internal/image.jpg",
        metadata=InputMetadata(
            latitude=44.0,
            longitude=5.0,
            location_origin="METADATA",
        ),
    )
    from fireviewer_orchestrator.session_runner import _initial_result

    result = _initial_result(source)
    validate_item_result(source, result)
    assert result.location_status == LocationStatus.CAPTURE_LOCATION_ONLY
    assert result.geographic_marker_candidate is not None
    assert result.geographic_marker_candidate.type == "media_capture"
    assert result.observed_phenomenon_marker is None
