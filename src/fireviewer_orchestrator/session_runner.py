from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter
from typing import Literal, cast

from pydantic import JsonValue

from fireviewer_vision_runtime.adapters import AdapterFactory, ItemPatch, ModelOutputError
from fireviewer_vision_runtime.consensus import (
    ConsensusEvaluation,
    ConsensusJudgeFactory,
    ConsensusJudgeVerdict,
    JudgeCandidate,
    SuccessfulCandidate,
    apply_adjudication,
    candidate_requires_challenge,
    evaluate_consensus,
)
from fireviewer_contracts.contracts import (
    GeographicMarkerCandidate,
    ItemResult,
    LocationOrigin,
    LocationStatus,
    MetadataResult,
    ModelRun,
    Transcript,
    VisualEvidenceSelection,
    WorkerConsensusResultV2,
    WorkerInput,
    WorkerModelCandidateRunV2,
    WorkerModelRoleV2,
    WorkerOutput,
    WorkerStageAttemptV2,
    WorkerStageGateV2,
    WorkerStageTraceV2,
)
from fireviewer_vision_runtime.memory_manager import MemoryManager, synchronize_cuda
from fireviewer_contracts.model_registry import (
    ConsensusStrategy,
    ModelCandidateSpec,
    ModelGroupSpec,
    ModelRole,
    ModelSpec,
)
from fireviewer_contracts.stage_contracts import (
    StageContractRegistry,
    StageRole,
    load_stage_contract_registry,
)
from fireviewer_orchestrator.stage_gates import (
    GateDecision,
    StageGateEngine,
    StageGateRecord,
    derive_capabilities,
)
from fireviewer_contracts.validation import OutputValidationError, validate_item_result

PipelineModelRole = Literal["asr", "fire_detection", "visual_grounding", "multimodal_extraction"]
ROLE_ORDER: tuple[PipelineModelRole, ...] = (
    "asr",
    "fire_detection",
    "visual_grounding",
    "multimodal_extraction",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _json_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _model_role_v2(role: PipelineModelRole) -> WorkerModelRoleV2:
    return "visual_filtering" if role == "fire_detection" else role


def _candidate_payload(patches: Mapping[str, ItemPatch]) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for input_id in sorted(patches):
        patch = patches[input_id]
        output: dict[str, object] = {"input_id": input_id}
        for field_name in (
            "transcript",
            "pixel_regions",
            "visual_evidence_selection",
            "factual_observations",
            "explicit_places",
            "explicit_times",
        ):
            value = getattr(patch, field_name)
            if value is None:
                continue
            if hasattr(value, "model_dump"):
                output[field_name] = value.model_dump(mode="json")
            else:
                output[field_name] = [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in value
                ]
        items.append(output)
    return {"schema_version": "candidate-patch-v1", "items": items}


def _initial_result(item: object) -> ItemResult:
    from fireviewer_contracts.contracts import BatchItem

    assert isinstance(item, BatchItem)
    metadata = item.metadata
    has_coordinates = metadata.latitude is not None
    origin = metadata.location_origin
    marker = None
    status = LocationStatus.NO_LOCATION
    if has_coordinates:
        if origin == LocationOrigin.METADATA:
            status = LocationStatus.CAPTURE_LOCATION_ONLY
        elif origin == LocationOrigin.USER_DECLARED:
            status = LocationStatus.USER_DECLARED_OBSERVATION_LOCATION
        elif origin == LocationOrigin.EXPLICIT_SOURCE_GEOMETRY:
            status = LocationStatus.EXPLICIT_SOURCE_GEOMETRY
        elif origin == LocationOrigin.HUMAN_CONFIRMED:
            status = LocationStatus.HUMAN_CONFIRMED_OBSERVATION_LOCATION
        assert origin is not None
        marker = GeographicMarkerCandidate(type="media_capture", geometry_origin=origin)
    visual_sources = [frame.frame_id for frame in item.frames]
    if not visual_sources and item.working_file_url is not None:
        visual_sources = [item.input_id]
    fallback_selected = set(_temporal_sample(visual_sources, limit=8))
    visual_selection = tuple(
        VisualEvidenceSelection(
            evidence_id=evidence_id,
            selected_for_grounding=evidence_id in fallback_selected,
            selection_reason=(
                "single_image"
                if len(visual_sources) == 1
                else "detector_fallback"
                if evidence_id in fallback_selected
                else "capacity_limit"
            ),
        )
        for evidence_id in visual_sources
    )
    return ItemResult(
        input_id=item.input_id,
        metadata_result=MetadataResult(
            capture_location_available=has_coordinates,
            capture_location_origin=origin,
        ),
        transcript=Transcript(),
        visual_evidence_selection=visual_selection,
        location_status=status,
        geographic_marker_candidate=marker,
    )


def _temporal_sample(evidence_ids: list[str], *, limit: int) -> tuple[str, ...]:
    if len(evidence_ids) <= limit:
        return tuple(evidence_ids)
    if limit == 1:
        return (evidence_ids[len(evidence_ids) // 2],)
    indexes = {round(position * (len(evidence_ids) - 1) / (limit - 1)) for position in range(limit)}
    return tuple(evidence_ids[index] for index in sorted(indexes))


def _as_patch(result: ItemResult) -> ItemPatch:
    return ItemPatch(
        transcript=result.transcript,
        pixel_regions=result.pixel_regions,
        visual_evidence_selection=result.visual_evidence_selection,
        factual_observations=result.factual_observations,
        explicit_places=result.explicit_places,
        explicit_times=result.explicit_times,
    )


def _merge(result: ItemResult, patch: ItemPatch) -> ItemResult:
    updates = {
        field: value
        for field, value in (
            ("transcript", patch.transcript),
            ("pixel_regions", patch.pixel_regions),
            ("visual_evidence_selection", patch.visual_evidence_selection),
            ("factual_observations", patch.factual_observations),
            ("explicit_places", patch.explicit_places),
            ("explicit_times", patch.explicit_times),
        )
        if value is not None
    }
    return result.model_copy(update=updates)


def _validated_candidates(
    batch: WorkerInput,
    results: Mapping[str, ItemResult],
    patches: Mapping[str, ItemPatch],
) -> dict[str, ItemResult]:
    candidate = dict(results)
    for input_id, patch in patches.items():
        if input_id not in candidate:
            raise OutputValidationError(f"adapter returned unknown input_id {input_id}")
        candidate[input_id] = _merge(candidate[input_id], patch)
    for item in batch.items:
        validate_item_result(item, candidate[item.input_id])
    return candidate


def _maximum_stage_output_items(
    role: PipelineModelRole,
    results: Mapping[str, ItemResult],
) -> int:
    counts: list[int] = []
    for result in results.values():
        if role == "asr":
            counts.append(len(result.transcript.segments))
        elif role == "fire_detection":
            counts.append(
                sum(region.task == "fire_detection" for region in result.pixel_regions)
                + len(result.visual_evidence_selection)
            )
        elif role == "visual_grounding":
            counts.append(
                sum(region.task in {"phrase_grounding", "ocr"} for region in result.pixel_regions)
            )
        else:
            counts.append(
                len(result.factual_observations)
                + len(result.explicit_places)
                + len(result.explicit_times)
            )
    return max(counts, default=0)


def _trace_gate(record: StageGateRecord) -> WorkerStageGateV2:
    return WorkerStageGateV2(
        phase=record.phase.value,
        decision=record.decision.value,
        reason_codes=record.reason_codes,
        available_capabilities=tuple(
            capability.value for capability in record.available_capabilities
        ),
        missing_capabilities=tuple(capability.value for capability in record.missing_capabilities),
        downstream_possible=record.downstream_possible,
    )


@dataclass(frozen=True, slots=True)
class SessionRunResult:
    output: WorkerOutput
    gate_records: tuple[StageGateRecord, ...]
    stage_traces: tuple[WorkerStageTraceV2, ...]
    candidate_runs: tuple[WorkerModelCandidateRunV2, ...]
    consensus_results: tuple[WorkerConsensusResultV2, ...]
    contract_digest: str


@dataclass(frozen=True, slots=True)
class _CandidateExecution:
    candidate: ModelCandidateSpec
    run: ModelRun
    candidate_run: WorkerModelCandidateRunV2
    results: Mapping[str, ItemResult] | None
    attempts: tuple[WorkerStageAttemptV2, ...]
    repaired: bool


@dataclass(frozen=True, slots=True)
class _AdjudicatorExecution:
    candidate_run: WorkerModelCandidateRunV2
    verdict: ConsensusJudgeVerdict | None


class SessionRunner:
    def __init__(
        self,
        *,
        registry: Mapping[ModelRole, ModelSpec | ModelGroupSpec],
        adapter_factory: AdapterFactory,
        memory: MemoryManager | None = None,
        boot_ms: int = 0,
        contracts: StageContractRegistry | None = None,
        gate_engine: StageGateEngine | None = None,
    ) -> None:
        normalized_registry: dict[ModelRole, ModelGroupSpec] = {}
        for role, value in registry.items():
            if isinstance(value, ModelGroupSpec):
                group = value
            else:
                group = ModelGroupSpec(
                    role=role,
                    candidates=(
                        ModelCandidateSpec(
                            candidate_id=f"{role}.primary",
                            spec=value,
                            rank=1,
                        ),
                    ),
                )
            if group.role != role:
                raise ValueError(f"registry key {role} does not match model group {group.role}")
            group.validate()
            normalized_registry[role] = group
        self.registry = normalized_registry
        self.adapter_factory = adapter_factory
        self.memory = memory or MemoryManager()
        self.boot_ms = boot_ms
        self.contracts = contracts or load_stage_contract_registry()
        self.gate_engine = gate_engine or StageGateEngine()
        for role in ROLE_ORDER:
            contract = self.contracts[StageRole(role)]
            if contract.model_binding != role:
                raise ValueError(f"stage contract {contract.contract_id} is not bound to {role}")

    def run(self, batch: WorkerInput) -> WorkerOutput:
        return self.run_with_trace(batch).output

    def run_with_trace(self, batch: WorkerInput) -> SessionRunResult:
        try:
            return self._run_with_trace(batch)
        finally:
            self.memory.finalize_job()

    def _execute_candidate(
        self,
        *,
        role: PipelineModelRole,
        candidate: ModelCandidateSpec,
        batch: WorkerInput,
        base_results: Mapping[str, ItemResult],
        max_repair_attempts: int,
    ) -> _CandidateExecution:
        spec = candidate.spec
        started_at = _now()
        adapter = self.adapter_factory.create(spec)
        load_started = perf_counter()
        load_ms = 0
        inference_ms = 0
        peak_vram: int | None = None
        error_code: str | None = None
        status: Literal["succeeded", "failed", "skipped"] = "failed"
        stage_attempts: list[WorkerStageAttemptV2] = []
        candidate_results: Mapping[str, ItemResult] | None = None
        output_payload: dict[str, object] | None = None
        repaired = False
        try:
            self.memory.reset_peak()
            adapter.load()
            synchronize_cuda()
            load_ms = round((perf_counter() - load_started) * 1_000)
            infer_started_at = _now()
            infer_started = perf_counter()
            try:
                patches = adapter.infer(
                    batch.items,
                    {input_id: _as_patch(result) for input_id, result in base_results.items()},
                )
                synchronize_cuda()
                initial_inference_ms = round((perf_counter() - infer_started) * 1_000)
                inference_ms = initial_inference_ms
                candidate_results = _validated_candidates(batch, base_results, patches)
            except (ModelOutputError, OutputValidationError) as first_validation_error:
                initial_inference_ms = round((perf_counter() - infer_started) * 1_000)
                inference_ms = initial_inference_ms
                stage_attempts.append(
                    WorkerStageAttemptV2(
                        attempt=1,
                        kind="initial",
                        status="failed",
                        started_at=infer_started_at,
                        finished_at=_now(),
                        inference_ms=initial_inference_ms,
                        peak_vram_bytes=self.memory.peak_vram_bytes(),
                        error_code="invalid_model_output",
                    )
                )
                if max_repair_attempts < 1:
                    raise
                correction_started_at = _now()
                correction_started = perf_counter()
                try:
                    patches = adapter.infer(
                        batch.items,
                        {input_id: _as_patch(result) for input_id, result in base_results.items()},
                        correction=True,
                    )
                    synchronize_cuda()
                    candidate_results = _validated_candidates(batch, base_results, patches)
                except (ModelOutputError, OutputValidationError) as correction_error:
                    correction_inference_ms = round((perf_counter() - correction_started) * 1_000)
                    inference_ms += correction_inference_ms
                    stage_attempts.append(
                        WorkerStageAttemptV2(
                            attempt=2,
                            kind="repair",
                            status="failed",
                            started_at=correction_started_at,
                            finished_at=_now(),
                            inference_ms=correction_inference_ms,
                            peak_vram_bytes=self.memory.peak_vram_bytes(),
                            error_code="invalid_model_output",
                        )
                    )
                    raise correction_error from first_validation_error
                except Exception:
                    correction_inference_ms = round((perf_counter() - correction_started) * 1_000)
                    inference_ms += correction_inference_ms
                    stage_attempts.append(
                        WorkerStageAttemptV2(
                            attempt=2,
                            kind="repair",
                            status="failed",
                            started_at=correction_started_at,
                            finished_at=_now(),
                            inference_ms=correction_inference_ms,
                            peak_vram_bytes=self.memory.peak_vram_bytes(),
                            error_code="model_runtime_error",
                        )
                    )
                    raise
                correction_inference_ms = round((perf_counter() - correction_started) * 1_000)
                inference_ms += correction_inference_ms
                repaired = True
                stage_attempts.append(
                    WorkerStageAttemptV2(
                        attempt=2,
                        kind="repair",
                        status="succeeded",
                        started_at=correction_started_at,
                        finished_at=_now(),
                        inference_ms=correction_inference_ms,
                        peak_vram_bytes=self.memory.peak_vram_bytes(),
                    )
                )
            except Exception:
                initial_inference_ms = round((perf_counter() - infer_started) * 1_000)
                inference_ms = initial_inference_ms
                stage_attempts.append(
                    WorkerStageAttemptV2(
                        attempt=1,
                        kind="initial",
                        status="failed",
                        started_at=infer_started_at,
                        finished_at=_now(),
                        inference_ms=initial_inference_ms,
                        peak_vram_bytes=self.memory.peak_vram_bytes(),
                        error_code="model_runtime_error",
                    )
                )
                raise
            else:
                stage_attempts.append(
                    WorkerStageAttemptV2(
                        attempt=1,
                        kind="initial",
                        status="succeeded",
                        started_at=infer_started_at,
                        finished_at=_now(),
                        inference_ms=initial_inference_ms,
                        peak_vram_bytes=self.memory.peak_vram_bytes(),
                    )
                )
            output_payload = _candidate_payload(patches)
            status = "succeeded"
        except (ModelOutputError, OutputValidationError) as exc:
            error_code = "invalid_model_output"
            print("FV_LAB_DIAGNOSTIC " + __import__("json").dumps(_lab_error_diagnostic(exc)), flush=True)
        except Exception as exc:
            error_code = "model_runtime_error"
            print("FV_LAB_DIAGNOSTIC " + __import__("json").dumps(_lab_error_diagnostic(exc)), flush=True)
        finally:
            peak_vram = self.memory.peak_vram_bytes()
            self.memory.release(adapter)

        finished_at = _now()
        run = ModelRun(
            model_role=role,
            model_id=spec.model_id,
            revision=spec.revision,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            load_ms=load_ms,
            inference_ms=inference_ms,
            peak_vram_bytes=peak_vram,
            error_code=error_code,
        )
        candidate_run = WorkerModelCandidateRunV2(
            candidate_id=candidate.candidate_id,
            candidate_rank=candidate.rank,
            stage_role=role,
            model_role=_model_role_v2(role),
            model_id=spec.model_id,
            revision=spec.revision,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            load_ms=load_ms,
            inference_ms=inference_ms,
            peak_vram_bytes=peak_vram,
            repaired=repaired,
            output_digest=_json_digest(output_payload) if output_payload is not None else None,
            output_payload=(
                cast(dict[str, JsonValue], output_payload) if output_payload is not None else None
            ),
            error_code=error_code,
        )
        return _CandidateExecution(
            candidate=candidate,
            run=run,
            candidate_run=candidate_run,
            results=candidate_results,
            attempts=tuple(stage_attempts),
            repaired=repaired,
        )

    def _execute_adjudicator(
        self,
        *,
        role: PipelineModelRole,
        adjudicator: ModelCandidateSpec,
        batch: WorkerInput,
        candidates: tuple[JudgeCandidate, ...],
        comparison_payload: Mapping[str, object],
    ) -> _AdjudicatorExecution:
        started_at = _now()
        status: Literal["succeeded", "failed", "skipped"] = "failed"
        error_code: str | None = None
        load_ms = 0
        inference_ms = 0
        peak_vram: int | None = None
        repaired = False
        verdict: ConsensusJudgeVerdict | None = None
        output_payload: dict[str, object] | None = None
        adapter = None
        try:
            if not isinstance(self.adapter_factory, ConsensusJudgeFactory):
                raise RuntimeError("adapter factory does not provide a consensus judge")
            adapter = self.adapter_factory.create_consensus_judge(adjudicator.spec)
            self.memory.reset_peak()
            load_started = perf_counter()
            adapter.load()
            synchronize_cuda()
            load_ms = round((perf_counter() - load_started) * 1_000)
            infer_started = perf_counter()
            try:
                verdict = adapter.adjudicate(
                    batch=batch,
                    stage_role=role,
                    candidates=candidates,
                    comparison_payload=comparison_payload,
                )
            except ModelOutputError:
                first_inference_ms = round((perf_counter() - infer_started) * 1_000)
                correction_started = perf_counter()
                verdict = adapter.adjudicate(
                    batch=batch,
                    stage_role=role,
                    candidates=candidates,
                    comparison_payload=comparison_payload,
                    correction=True,
                )
                inference_ms = first_inference_ms + round(
                    (perf_counter() - correction_started) * 1_000
                )
                repaired = True
            else:
                inference_ms = round((perf_counter() - infer_started) * 1_000)
            synchronize_cuda()
            output_payload = verdict.output_payload
            status = "succeeded"
        except ModelOutputError:
            error_code = "invalid_model_output"
        except Exception as exc:
            error_code = "model_runtime_error"
            print("FV_LAB_DIAGNOSTIC " + __import__("json").dumps(_lab_error_diagnostic(exc)), flush=True)
        finally:
            peak_vram = self.memory.peak_vram_bytes()
            if adapter is not None:
                self.memory.release(adapter)
        candidate_run = WorkerModelCandidateRunV2(
            candidate_id=adjudicator.candidate_id,
            candidate_rank=adjudicator.rank,
            stage_role=role,
            model_role="consensus_judge",
            model_id=adjudicator.spec.model_id,
            revision=adjudicator.spec.revision,
            status=status,
            started_at=started_at,
            finished_at=_now(),
            load_ms=load_ms,
            inference_ms=inference_ms,
            peak_vram_bytes=peak_vram,
            repaired=repaired,
            output_digest=_json_digest(output_payload) if output_payload is not None else None,
            output_payload=(
                cast(dict[str, JsonValue], output_payload) if output_payload is not None else None
            ),
            error_code=error_code,
        )
        return _AdjudicatorExecution(candidate_run=candidate_run, verdict=verdict)

    def _run_with_trace(self, batch: WorkerInput) -> SessionRunResult:
        results = {item.input_id: _initial_result(item) for item in batch.items}
        runs: list[ModelRun] = []
        errors: list[str] = []
        gate_records: list[StageGateRecord] = []
        stage_traces: list[WorkerStageTraceV2] = []
        candidate_runs: list[WorkerModelCandidateRunV2] = []
        consensus_results: list[WorkerConsensusResultV2] = []

        for sequence, role in enumerate(ROLE_ORDER, start=1):
            contract = self.contracts[StageRole(role)]
            before = derive_capabilities(batch, results)
            preflight = self.gate_engine.preflight(
                contract,
                before,
                input_items=len(batch.items),
            )
            gate_records.append(preflight)
            group = self.registry.get(role)
            primary_spec = group.candidates[0].spec if group is not None else None
            if preflight.decision != GateDecision.PROCEED:
                moment = _now()
                preflight_error_code = (
                    None
                    if preflight.decision == GateDecision.NOT_APPLICABLE
                    else f"gate_{preflight.reason_codes[0]}"
                )
                if preflight_error_code is not None:
                    errors.append(f"{role}:{preflight_error_code}")
                runs.append(
                    ModelRun(
                        model_role=role,
                        model_id=primary_spec.model_id if primary_spec else "unconfigured",
                        revision=primary_spec.revision if primary_spec else "unconfigured",
                        status="skipped",
                        started_at=moment,
                        finished_at=moment,
                        load_ms=0,
                        inference_ms=0,
                        error_code=preflight_error_code,
                    )
                )
                stage_traces.append(
                    WorkerStageTraceV2(
                        stage_role=contract.role.value,
                        contract_id=contract.contract_id,
                        sequence=sequence,
                        status="skipped",
                        retryable=preflight.decision == GateDecision.FAILED_RETRYABLE,
                        preflight=_trace_gate(preflight),
                    )
                )
                continue
            if group is None:
                missing_model_error_code = (
                    "checkpoint_not_configured"
                    if role == "fire_detection"
                    else "model_not_configured"
                )
                errors.append(f"{role}:{missing_model_error_code}")
                moment = _now()
                runs.append(
                    ModelRun(
                        model_role=role,
                        model_id="unconfigured",
                        revision="unconfigured",
                        status="skipped",
                        started_at=moment,
                        finished_at=moment,
                        load_ms=0,
                        inference_ms=0,
                        error_code=missing_model_error_code,
                    )
                )
                postflight = self.gate_engine.postflight(
                    contract,
                    before=before,
                    after=before,
                    status="skipped",
                    error_code=missing_model_error_code,
                    elapsed_seconds=0,
                    maximum_output_items=0,
                )
                gate_records.append(postflight)
                stage_traces.append(
                    WorkerStageTraceV2(
                        stage_role=contract.role.value,
                        contract_id=contract.contract_id,
                        sequence=sequence,
                        status="skipped",
                        retryable=False,
                        preflight=_trace_gate(preflight),
                        postflight=_trace_gate(postflight),
                    )
                )
                continue

            stage_started = perf_counter()
            executions: list[_CandidateExecution] = []
            for candidate in group.candidates:
                execution = self._execute_candidate(
                    role=role,
                    candidate=candidate,
                    batch=batch,
                    base_results=results,
                    max_repair_attempts=contract.max_repair_attempts,
                )
                executions.append(execution)
                candidate_runs.append(execution.candidate_run)
                if execution.run.status == "failed":
                    errors.append(f"{role}:{candidate.candidate_id}:{execution.run.error_code}")
                if (
                    group.strategy == ConsensusStrategy.CASCADE
                    and len(executions) == 1
                    and execution.results is not None
                    and group.minimum_successful == 1
                    and group.minimum_agreeing == 1
                    and not group.always_challenge
                    and not candidate_requires_challenge(role, execution.results)
                ):
                    break

            successful = tuple(
                SuccessfulCandidate(
                    candidate_id=execution.candidate.candidate_id,
                    results=execution.results,
                    repaired=execution.repaired,
                )
                for execution in executions
                if execution.run.status == "succeeded" and execution.results is not None
            )
            evaluation = evaluate_consensus(role, group, successful)
            adjudicator_candidate_id: str | None = None
            if evaluation.requires_adjudication:
                assert group.adjudicator is not None
                judge_candidates = tuple(
                    JudgeCandidate(
                        candidate_id=execution.candidate.candidate_id,
                        model_id=execution.candidate.spec.model_id,
                        revision=execution.candidate.spec.revision,
                        output_payload=execution.candidate_run.output_payload,
                    )
                    for execution in executions
                    if execution.run.status == "succeeded"
                    and execution.candidate_run.output_payload is not None
                )
                adjudication = self._execute_adjudicator(
                    role=role,
                    adjudicator=group.adjudicator,
                    batch=batch,
                    candidates=judge_candidates,
                    comparison_payload=evaluation.comparison_payload,
                )
                candidate_runs.append(adjudication.candidate_run)
                adjudicator_candidate_id = group.adjudicator.candidate_id
                if adjudication.verdict is not None:
                    evaluation = apply_adjudication(
                        evaluation,
                        adjudication.verdict,
                        candidate_ids=tuple(candidate.candidate_id for candidate in successful),
                        confidence_threshold=group.adjudication_confidence_threshold,
                        failure_decision=group.disagreement_decision,
                    )
                else:
                    errors.append(
                        f"consensus:{role}:adjudicator:{adjudication.candidate_run.error_code}"
                    )
                    evaluation = ConsensusEvaluation(
                        decision=group.disagreement_decision.value,
                        selected_candidate_id=None,
                        agreement_score=evaluation.agreement_score,
                        reason_codes=("adjudicator_failed",),
                        downstream_allowed=False,
                        comparison_payload={
                            **evaluation.comparison_payload,
                            "adjudication": {
                                "status": "failed",
                                "error_code": adjudication.candidate_run.error_code,
                            },
                        },
                    )
            comparison_payload = evaluation.comparison_payload
            consensus_results.append(
                WorkerConsensusResultV2(
                    consensus_id=f"consensus:{role}",
                    stage_role=role,
                    strategy=group.strategy.value,
                    decision=evaluation.decision,
                    candidate_ids=tuple(
                        execution.candidate.candidate_id for execution in executions
                    ),
                    selected_candidate_id=evaluation.selected_candidate_id,
                    adjudicator_candidate_id=adjudicator_candidate_id,
                    reason_codes=evaluation.reason_codes,
                    successful_candidates=len(successful),
                    required_successful=group.minimum_successful,
                    agreement_score=evaluation.agreement_score,
                    agreement_threshold=group.agreement_threshold,
                    downstream_allowed=evaluation.downstream_allowed,
                    comparison_digest=_json_digest(comparison_payload),
                    comparison_payload=cast(dict[str, JsonValue], comparison_payload),
                    evaluated_at=_now(),
                )
            )

            selected = next(
                (
                    execution
                    for execution in executions
                    if execution.candidate.candidate_id == evaluation.selected_candidate_id
                ),
                None,
            )
            if (
                evaluation.downstream_allowed
                and selected is not None
                and selected.results is not None
            ):
                results = dict(selected.results)
                run = selected.run
                status: Literal["succeeded", "failed", "skipped"] = "succeeded"
                error_code = None
                stage_attempts = selected.attempts
            else:
                representative = executions[0]
                consensus_error = f"consensus_{evaluation.reason_codes[0]}"
                if successful:
                    run = representative.run.model_copy(
                        update={"status": "failed", "error_code": consensus_error}
                    )
                    error_code = consensus_error
                    stage_attempts = ()
                else:
                    run = representative.run
                    error_code = representative.run.error_code
                    stage_attempts = representative.attempts
                status = "failed"
                errors.append(f"consensus:{role}:{evaluation.reason_codes[0]}")
            runs.append(run)
            after = derive_capabilities(batch, results)
            postflight = self.gate_engine.postflight(
                contract,
                before=before,
                after=after,
                status=status,
                error_code=error_code,
                elapsed_seconds=perf_counter() - stage_started,
                maximum_output_items=_maximum_stage_output_items(role, results),
            )
            gate_records.append(postflight)
            stage_traces.append(
                WorkerStageTraceV2(
                    stage_role=contract.role.value,
                    contract_id=contract.contract_id,
                    sequence=sequence,
                    status=status,
                    retryable=postflight.decision == GateDecision.FAILED_RETRYABLE,
                    preflight=_trace_gate(preflight),
                    postflight=_trace_gate(postflight),
                    attempts=tuple(stage_attempts),
                )
            )
            if postflight.decision in {
                GateDecision.HUMAN_REVIEW,
                GateDecision.FAILED_RETRYABLE,
                GateDecision.FAILED_TERMINAL,
            }:
                gate_error = f"gate:{role}:{postflight.reason_codes[0]}"
                if gate_error not in errors:
                    errors.append(gate_error)

        succeeded = sum(run.status == "succeeded" for run in runs)
        failed = sum(run.status == "failed" for run in runs)
        incomplete = any(run.status == "skipped" and run.error_code for run in runs) or any(
            record.decision
            in {
                GateDecision.HUMAN_REVIEW,
                GateDecision.FAILED_RETRYABLE,
                GateDecision.FAILED_TERMINAL,
            }
            for record in gate_records
        )
        if failed == 0 and not incomplete:
            overall_status: Literal["succeeded", "partial_failure", "failed"] = "succeeded"
        elif succeeded or incomplete:
            overall_status = "partial_failure"
        else:
            overall_status = "failed"
        output = WorkerOutput(
            batch_id=batch.batch_id,
            status=overall_status,
            retryable=any(
                record.decision == GateDecision.FAILED_RETRYABLE for record in gate_records
            ),
            model_runs=tuple(runs),
            items=tuple(results[item.input_id] for item in batch.items),
            validation_errors=tuple(errors),
            boot_ms=self.boot_ms,
        )
        return SessionRunResult(
            output=output,
            gate_records=tuple(gate_records),
            stage_traces=tuple(stage_traces),
            candidate_runs=tuple(candidate_runs),
            consensus_results=tuple(consensus_results),
            contract_digest=self.contracts.digest,
        )


def _lab_error_diagnostic(exc):
    """Keep diagnostics structural: exception messages can contain credentials/evidence."""
    import traceback
    return {"diagnostic": {"exception": type(exc).__name__,
            "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
            "frames": [{"file": f.filename.rsplit("/", 1)[-1], "line": f.lineno,
                        "function": f.name}
                       for f in traceback.extract_tb(exc.__traceback__)[-5:]]}}
