from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import Field

from fireviewer_contracts.contracts import ItemResult, StrictModel, WorkerInput
from fireviewer_contracts.stage_contracts import (
    FailurePolicy,
    StageCapability,
    StageContract,
    StageRole,
)


class GatePhase(StrEnum):
    PREFLIGHT = "preflight"
    POSTFLIGHT = "postflight"


class GateDecision(StrEnum):
    PROCEED = "pass"
    NOT_APPLICABLE = "not_applicable"
    ABSTAIN = "abstain"
    HUMAN_REVIEW = "human_review"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


class StageGateRecord(StrictModel):
    role: StageRole
    phase: GatePhase
    decision: GateDecision
    reason_codes: tuple[str, ...] = Field(min_length=1)
    available_capabilities: tuple[StageCapability, ...]
    missing_capabilities: tuple[StageCapability, ...] = ()
    downstream_possible: bool


def derive_capabilities(
    batch: WorkerInput,
    results: Mapping[str, ItemResult],
) -> frozenset[StageCapability]:
    capabilities: set[StageCapability] = set()
    if any(item.audio_url is not None for item in batch.items):
        capabilities.add(StageCapability.AUDIO_INPUT)
    if any(item.working_file_url is not None or item.frames for item in batch.items):
        capabilities.add(StageCapability.VISUAL_INPUT)
    if any(item.article_text for item in batch.items):
        capabilities.add(StageCapability.TEXT_INPUT)

    for result in results.values():
        if result.transcript.segments:
            capabilities.add(StageCapability.TRANSCRIPT)
        if any(selection.selected_for_grounding for selection in result.visual_evidence_selection):
            capabilities.add(StageCapability.SELECTED_VISUAL)
        if any(region.task == "fire_detection" for region in result.pixel_regions):
            capabilities.add(StageCapability.DETECTION_REGIONS)
        if any(region.task in {"phrase_grounding", "ocr"} for region in result.pixel_regions):
            capabilities.add(StageCapability.GROUNDED_REGIONS)
        if result.factual_observations:
            capabilities.add(StageCapability.FACTUAL_OBSERVATIONS)
    return frozenset(capabilities)


def _ordered(
    capabilities: set[StageCapability] | frozenset[StageCapability],
) -> tuple[StageCapability, ...]:
    return tuple(sorted(capabilities, key=lambda capability: capability.value))


def _missing_requirements(
    contract: StageContract,
    capabilities: frozenset[StageCapability],
) -> tuple[StageCapability, ...]:
    missing = set(contract.required_all) - capabilities
    if contract.required_any and not capabilities.intersection(contract.required_any):
        missing.update(contract.required_any)
    return _ordered(missing)


def _is_not_applicable(
    role: StageRole,
    capabilities: frozenset[StageCapability],
) -> bool:
    direct_input = {
        StageRole.ASR: StageCapability.AUDIO_INPUT,
        StageRole.FIRE_DETECTION: StageCapability.VISUAL_INPUT,
        StageRole.VISUAL_GROUNDING: StageCapability.VISUAL_INPUT,
        StageRole.FIRE_POINTING: StageCapability.VISUAL_INPUT,
    }.get(role)
    return direct_input is not None and direct_input not in capabilities


class StageGateEngine:
    def preflight(
        self,
        contract: StageContract,
        capabilities: frozenset[StageCapability],
        *,
        input_items: int,
    ) -> StageGateRecord:
        if input_items > contract.max_input_items:
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.PREFLIGHT,
                decision=GateDecision.FAILED_TERMINAL,
                reason_codes=("stage_input_limit_exceeded",),
                available_capabilities=_ordered(capabilities),
                downstream_possible=False,
            )
        missing = _missing_requirements(contract, capabilities)
        if not missing:
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.PREFLIGHT,
                decision=GateDecision.PROCEED,
                reason_codes=("requirements_satisfied",),
                available_capabilities=_ordered(capabilities),
                downstream_possible=True,
            )
        if contract.skip_when_not_applicable and _is_not_applicable(contract.role, capabilities):
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.PREFLIGHT,
                decision=GateDecision.NOT_APPLICABLE,
                reason_codes=("no_applicable_input",),
                available_capabilities=_ordered(capabilities),
                missing_capabilities=missing,
                downstream_possible=True,
            )
        decision = (
            GateDecision.HUMAN_REVIEW
            if contract.failure_policy == FailurePolicy.HUMAN_REVIEW
            else GateDecision.FAILED_TERMINAL
            if contract.failure_policy == FailurePolicy.BLOCK
            else GateDecision.ABSTAIN
        )
        return StageGateRecord(
            role=contract.role,
            phase=GatePhase.PREFLIGHT,
            decision=decision,
            reason_codes=("required_capability_missing",),
            available_capabilities=_ordered(capabilities),
            missing_capabilities=missing,
            downstream_possible=contract.failure_policy != FailurePolicy.BLOCK,
        )

    def postflight(
        self,
        contract: StageContract,
        *,
        before: frozenset[StageCapability],
        after: frozenset[StageCapability],
        status: Literal["succeeded", "failed", "skipped"],
        error_code: str | None,
        elapsed_seconds: float,
        maximum_output_items: int,
    ) -> StageGateRecord:
        if status == "failed":
            retryable = error_code == "model_runtime_error"
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.POSTFLIGHT,
                decision=(
                    GateDecision.FAILED_RETRYABLE if retryable else GateDecision.FAILED_TERMINAL
                ),
                reason_codes=(error_code or "stage_failed",),
                available_capabilities=_ordered(after),
                downstream_possible=contract.failure_policy != FailurePolicy.BLOCK,
            )
        if status == "skipped":
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.POSTFLIGHT,
                decision=(GateDecision.ABSTAIN if error_code else GateDecision.NOT_APPLICABLE),
                reason_codes=(error_code or "stage_skipped",),
                available_capabilities=_ordered(after),
                downstream_possible=True,
            )

        if elapsed_seconds > contract.max_wall_time_seconds:
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.POSTFLIGHT,
                decision=GateDecision.FAILED_RETRYABLE,
                reason_codes=("stage_wall_time_exceeded",),
                available_capabilities=_ordered(after),
                downstream_possible=False,
            )
        if maximum_output_items > contract.max_output_items_per_input:
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.POSTFLIGHT,
                decision=GateDecision.FAILED_TERMINAL,
                reason_codes=("stage_output_limit_exceeded",),
                available_capabilities=_ordered(after),
                downstream_possible=False,
            )

        produced = (after - before).intersection(contract.minimum_output_any)
        if produced:
            return StageGateRecord(
                role=contract.role,
                phase=GatePhase.POSTFLIGHT,
                decision=GateDecision.PROCEED,
                reason_codes=("minimum_output_satisfied",),
                available_capabilities=_ordered(after),
                downstream_possible=True,
            )

        decision = (
            GateDecision.HUMAN_REVIEW
            if contract.failure_policy == FailurePolicy.HUMAN_REVIEW
            else GateDecision.ABSTAIN
        )
        return StageGateRecord(
            role=contract.role,
            phase=GatePhase.POSTFLIGHT,
            decision=decision,
            reason_codes=("minimum_output_missing",),
            available_capabilities=_ordered(after),
            missing_capabilities=contract.minimum_output_any,
            downstream_possible=contract.failure_policy != FailurePolicy.BLOCK,
        )
