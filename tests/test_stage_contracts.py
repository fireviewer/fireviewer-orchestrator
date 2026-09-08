from __future__ import annotations

from fireviewer_contracts.model_registry import PUBLIC_MODELS, RTDETR_BASELINE
from fireviewer_contracts.stage_contracts import (
    StageCapability,
    StageRole,
    load_stage_contract_registry,
)
from fireviewer_orchestrator.stage_gates import GateDecision, StageGateEngine


def test_all_planned_stages_have_one_packaged_contract() -> None:
    registry = load_stage_contract_registry()

    assert set(registry) == set(StageRole)
    assert len(registry.digest) == 64
    assert registry.digest == load_stage_contract_registry().digest


def test_every_configured_model_role_is_bound_to_its_contract() -> None:
    contracts = load_stage_contract_registry()

    for spec in (*PUBLIC_MODELS, RTDETR_BASELINE):
        contract = contracts[StageRole(spec.role)]
        assert contract.execution_kind == "model"
        assert contract.model_binding == spec.role


def test_spatial_boundary_keeps_model_pixels_separate_from_coordinates() -> None:
    contracts = load_stage_contract_registry()
    pointing = contracts[StageRole.FIRE_POINTING]
    projection = contracts[StageRole.SPATIAL_PROJECTION]

    assert pointing.produces == (
        StageCapability.FIRE_POINT_PIXEL,
        StageCapability.EXPLICIT_ABSTENTION,
    )
    assert all("latitude" not in capability.value for capability in pointing.produces)
    assert any("latitude" in rule for rule in pointing.forbidden_outputs)
    assert projection.execution_kind == "deterministic"
    assert StageCapability.CAMERA_POSE in projection.required_all
    assert StageCapability.TERRAIN_REFERENCE in projection.required_all
    assert StageCapability.SPATIAL_PROPOSALS in projection.produces


def test_preflight_distinguishes_not_applicable_from_blocked_semantic_analysis() -> None:
    contracts = load_stage_contract_registry()
    engine = StageGateEngine()

    asr = engine.preflight(
        contracts[StageRole.ASR],
        frozenset({StageCapability.TEXT_INPUT}),
        input_items=1,
    )
    multimodal = engine.preflight(
        contracts[StageRole.MULTIMODAL_EXTRACTION],
        frozenset({StageCapability.AUDIO_INPUT}),
        input_items=1,
    )

    assert asr.decision == GateDecision.NOT_APPLICABLE
    assert asr.downstream_possible is True
    assert multimodal.decision == GateDecision.HUMAN_REVIEW
    assert set(multimodal.missing_capabilities) == {
        StageCapability.TRANSCRIPT,
        StageCapability.TEXT_INPUT,
        StageCapability.SELECTED_VISUAL,
    }


def test_postflight_marks_runtime_failure_retryable_and_empty_report_input_for_review() -> None:
    contracts = load_stage_contract_registry()
    engine = StageGateEngine()
    before = frozenset({StageCapability.SELECTED_VISUAL})

    runtime_failure = engine.postflight(
        contracts[StageRole.VISUAL_GROUNDING],
        before=before,
        after=before,
        status="failed",
        error_code="model_runtime_error",
        elapsed_seconds=1,
        maximum_output_items=0,
    )
    empty_multimodal = engine.postflight(
        contracts[StageRole.MULTIMODAL_EXTRACTION],
        before=before,
        after=before,
        status="succeeded",
        error_code=None,
        elapsed_seconds=1,
        maximum_output_items=0,
    )

    assert runtime_failure.decision == GateDecision.FAILED_RETRYABLE
    assert empty_multimodal.decision == GateDecision.HUMAN_REVIEW
    assert empty_multimodal.reason_codes == ("minimum_output_missing",)
