from __future__ import annotations

from fireviewer_contracts.resources import schema_path

import json
from pathlib import Path

from fireviewer_contracts.contracts import (
    SourceAnnotationV2,
    WorkerInputV2,
    WorkerOutput,
)
from fireviewer_contracts.model_registry import ModelSpec
from fireviewer_vision_runtime.v2_pointing import run_fire_pointing_stage
from fireviewer_orchestrator.v2_runner import to_legacy_input

EXAMPLES = schema_path("agent-worker/v2/examples/valid-input.json").parent


class _PointingAdapter:
    spec = ModelSpec(
        role="fire_pointing",
        model_id="tests/fire-pointing-fixture",
        revision="0000000000000000000000000000000000000000",
    )

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def infer(
        self,
        batch: WorkerInputV2,
        legacy: WorkerOutput,
    ) -> dict[str, tuple[SourceAnnotationV2, ...]]:
        assert self.loaded
        assert legacy.batch_id == batch.batch_id
        return {
            batch.items[0].input_id: (
                SourceAnnotationV2(
                    annotation_id="ANN-native-pointing-0001",
                    evidence_id=batch.items[0].input_id,
                    evidence_kind="image",
                    semantic_anchor="active_fire_point",
                    source_point_normalized=(0.25, 0.75),
                ),
            )
        }

    def unload(self) -> None:
        self.unloaded = True


def _legacy_output(batch: WorkerInputV2) -> WorkerOutput:
    legacy = to_legacy_input(batch)
    return WorkerOutput.model_validate(
        {
            "batch_id": legacy.batch_id,
            "status": "succeeded",
            "retryable": False,
            "model_runs": [],
            "items": [
                {
                    "input_id": item.input_id,
                    "metadata_result": {"capture_location_available": False},
                    "location_status": "NO_LOCATION",
                }
                for item in legacy.items
            ],
            "boot_ms": 0,
        }
    )


def test_native_pointing_stage_emits_linked_pixel_annotations() -> None:
    batch = WorkerInputV2.model_validate_json(
        (EXAMPLES / "valid-input.json").read_text(encoding="utf-8")
    )
    adapter = _PointingAdapter()

    execution = run_fire_pointing_stage(
        batch,
        _legacy_output(batch),
        adapter=adapter,
        sequence=5,
    )

    annotation = execution.annotations_by_input[batch.items[0].input_id][0]
    assert annotation.semantic_anchor == "active_fire_point"
    assert annotation.source_point_normalized == (0.25, 0.75)
    assert execution.stage_trace.stage_role == "fire_pointing"
    assert execution.stage_trace.status == "succeeded"
    assert execution.model_run is not None
    assert execution.model_run.model_role == "fire_pointing"
    assert adapter.unloaded is True


def test_missing_runtime_records_degraded_fallback_without_inventing_points() -> None:
    payload = json.loads((EXAMPLES / "valid-input.json").read_text(encoding="utf-8"))
    batch = WorkerInputV2.model_validate(payload)

    execution = run_fire_pointing_stage(
        batch,
        _legacy_output(batch),
        adapter=None,
        sequence=5,
    )

    assert execution.annotations_by_input == {}
    assert execution.stage_trace.status == "skipped"
    assert execution.stage_trace.preflight.reason_codes == ("fire_pointing_model_unavailable",)
    assert execution.model_run is None
