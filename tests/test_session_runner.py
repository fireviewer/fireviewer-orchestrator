from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from fireviewer_vision_runtime.adapters import ItemPatch, ModelOutputError
from fireviewer_vision_runtime.consensus import (
    ConsensusJudgeVerdict,
    JudgeCandidate,
    PipelineRole,
)
from fireviewer_contracts.contracts import (
    BatchItem,
    FactualObservation,
    PixelRegion,
    Transcript,
    TranscriptSegment,
    WorkerInput,
)
from fireviewer_contracts.model_registry import (
    ConsensusStrategy,
    ModelCandidateSpec,
    ModelGroupSpec,
    ModelSpec,
)
from fireviewer_orchestrator.session_runner import SessionRunner
from fireviewer_contracts.stage_contracts import StageRole
from fireviewer_orchestrator.stage_gates import GateDecision
from fireviewer_vision_runtime.transformers_adapters import RTDETRAdapter


@dataclass(slots=True)
class FakeAdapter:
    spec: ModelSpec
    calls: list[tuple[str, bool]]
    failing_role: str | None = None

    def load(self) -> None:
        self.calls.append((f"load:{self.spec.role}", False))

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        self.calls.append((f"infer:{self.spec.role}", correction))
        if self.spec.role == self.failing_role:
            raise RuntimeError("planned failure")
        item = items[0]
        if self.spec.role == "asr":
            return {
                item.input_id: ItemPatch(
                    transcript=Transcript(
                        language="fr",
                        segments=(
                            TranscriptSegment(
                                segment_id=f"{item.input_id}:audio:0001",
                                start_s=0,
                                end_s=1,
                                text="fumée visible",
                            ),
                        ),
                    )
                )
            }
        if self.spec.role == "fire_detection":
            return {
                item.input_id: ItemPatch(
                    pixel_regions=(
                        PixelRegion(
                            region_id="frame-1:det:0001",
                            evidence_id="frame-1",
                            label="smoke_visible",
                            bbox_normalized=(0.1, 0.1, 0.5, 0.5),
                            task="fire_detection",
                            model_score=0.8,
                        ),
                    )
                )
            }
        if self.spec.role == "multimodal_extraction":
            description = "Une fumée est directement visible."
            return {
                item.input_id: ItemPatch(
                    factual_observations=(
                        FactualObservation(
                            type="smoke_visible",
                            evidence_kind="frame",
                            evidence_id="frame-1",
                            region_id="frame-1:det:0001",
                            description=description,
                            certainty="directly_visible",
                        ),
                    )
                )
            }
        return {}

    def unload(self) -> None:
        self.calls.append((f"unload:{self.spec.role}", False))


@dataclass(slots=True)
class FakeFactory:
    failing_role: str | None = None
    calls: list[tuple[str, bool]] = field(default_factory=list)

    def create(self, spec: ModelSpec) -> FakeAdapter:
        return FakeAdapter(spec, self.calls, self.failing_role)


class FakeMemory:
    def __init__(self) -> None:
        self.finalizations = 0

    def reset_peak(self) -> None:
        return None

    def peak_vram_bytes(self) -> int:
        return 123

    def release(self, adapter: FakeAdapter) -> None:
        adapter.unload()

    def finalize_job(self) -> None:
        self.finalizations += 1


def _batch() -> WorkerInput:
    return WorkerInput.model_validate(
        {
            "batch_id": "BATCH-1",
            "batch_type": "user_media",
            "priority": "user_deadline",
            "items": [
                {
                    "input_id": "INPUT-1",
                    "media_type": "video",
                    "working_file_url": "https://media.internal/video.mp4",
                    "audio_url": "https://media.internal/audio.wav",
                    "frames": [
                        {
                            "frame_id": "frame-1",
                            "timestamp_s": 1,
                            "working_file_url": "https://media.internal/frame.jpg",
                        }
                    ],
                }
            ],
        }
    )


def _registry() -> dict[str, ModelSpec]:
    return {
        role: ModelSpec(role=role, model_id=f"org/{role}", revision=index * 40)
        for role, index in (
            ("asr", "a"),
            ("fire_detection", "b"),
            ("visual_grounding", "c"),
            ("multimodal_extraction", "d"),
        )
    }


ASR_PRIMARY_MODEL_ID = "org/asr"
ASR_CHALLENGER_MODEL_ID = "org/asr-challenger"


def _registry_with_asr_group(
    *,
    strategy: ConsensusStrategy,
    minimum_successful: int,
    minimum_agreeing: int,
    agreement_threshold: float = 0.95,
) -> dict[str, ModelSpec | ModelGroupSpec]:
    registry: dict[str, ModelSpec | ModelGroupSpec] = dict(_registry())
    registry["asr"] = ModelGroupSpec(
        role="asr",
        candidates=(
            ModelCandidateSpec(
                candidate_id="asr.primary",
                spec=registry["asr"],
                rank=1,
            ),
            ModelCandidateSpec(
                candidate_id="asr.challenger",
                spec=ModelSpec(
                    role="asr",
                    model_id=ASR_CHALLENGER_MODEL_ID,
                    revision="e" * 40,
                ),
                rank=2,
            ),
        ),
        strategy=strategy,
        minimum_successful=minimum_successful,
        minimum_agreeing=minimum_agreeing,
        agreement_threshold=agreement_threshold,
        adjudicator=(
            ModelCandidateSpec(
                candidate_id="asr.judge",
                spec=ModelSpec(
                    role="consensus_judge",
                    model_id="Qwen/Qwen3.5-27B",
                    revision="f" * 40,
                ),
                rank=8,
            )
            if strategy == ConsensusStrategy.QUORUM or minimum_agreeing > 1
            else None
        ),
    )
    return registry


@dataclass(slots=True)
class ConsensusAdapter(FakeAdapter):
    transcripts_by_model: Mapping[str, str | None] = field(default_factory=dict)

    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        if self.spec.role != "asr" or self.spec.model_id not in self.transcripts_by_model:
            return FakeAdapter.infer(self, items, accumulated, correction=correction)
        self.calls.append((f"infer:{self.spec.role}:{self.spec.model_id}", correction))
        text = self.transcripts_by_model[self.spec.model_id]
        if text is None:
            return {}
        item = items[0]
        return {
            item.input_id: ItemPatch(
                transcript=Transcript(
                    language="fr",
                    segments=(
                        TranscriptSegment(
                            segment_id=f"{item.input_id}:audio:0001",
                            start_s=0,
                            end_s=1,
                            text=text,
                        ),
                    ),
                )
            )
        }


@dataclass(slots=True)
class ConsensusFactory(FakeFactory):
    transcripts_by_model: dict[str, str | None] = field(default_factory=dict)
    created_models: list[str] = field(default_factory=list)
    judge_selected_candidate_id: str | None = "asr.primary"
    judge_confidence: float = 0.9
    judge_calls: list[str] = field(default_factory=list)

    def create(self, spec: ModelSpec) -> FakeAdapter:
        self.created_models.append(spec.model_id)
        return ConsensusAdapter(
            spec,
            self.calls,
            self.failing_role,
            self.transcripts_by_model,
        )

    def create_consensus_judge(self, spec: ModelSpec) -> FakeConsensusJudge:
        return FakeConsensusJudge(
            spec=spec,
            selected_candidate_id=self.judge_selected_candidate_id,
            confidence=self.judge_confidence,
            calls=self.judge_calls,
        )


@dataclass(slots=True)
class FakeConsensusJudge:
    spec: ModelSpec
    selected_candidate_id: str | None
    confidence: float
    calls: list[str]

    def load(self) -> None:
        self.calls.append("load")

    def adjudicate(
        self,
        *,
        batch: WorkerInput,
        stage_role: PipelineRole,
        candidates: Sequence[JudgeCandidate],
        comparison_payload: Mapping[str, object],
        correction: bool = False,
    ) -> ConsensusJudgeVerdict:
        del batch, comparison_payload
        self.calls.append(f"adjudicate:{stage_role}:{correction}")
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        selected = self.selected_candidate_id
        if selected not in candidate_ids:
            selected = None
        reason_codes = (
            ("source_evidence_supports_candidate",)
            if selected is not None
            else ("source_evidence_ambiguous",)
        )
        payload: dict[str, object] = {
            "selected_candidate_id": selected,
            "confidence": self.confidence,
            "reason_codes": list(reason_codes),
        }
        return ConsensusJudgeVerdict(
            selected_candidate_id=selected,
            confidence=self.confidence,
            reason_codes=reason_codes,
            output_payload=payload,
        )

    def unload(self) -> None:
        self.calls.append("unload")


def test_models_execute_and_release_in_the_required_order() -> None:
    factory = FakeFactory()
    memory = FakeMemory()
    output = SessionRunner(registry=_registry(), adapter_factory=factory, memory=memory).run(
        _batch()
    )
    assert output.status == "succeeded"
    assert [run.model_role for run in output.model_runs] == [
        "asr",
        "fire_detection",
        "visual_grounding",
        "multimodal_extraction",
    ]
    assert [name for name, _ in factory.calls if name.startswith(("load", "unload"))] == [
        "load:asr",
        "unload:asr",
        "load:fire_detection",
        "unload:fire_detection",
        "load:visual_grounding",
        "unload:visual_grounding",
        "load:multimodal_extraction",
        "unload:multimodal_extraction",
    ]
    assert memory.finalizations == 1


def test_quorum_keeps_only_the_agreed_candidate_output_for_downstream() -> None:
    factory = ConsensusFactory(
        transcripts_by_model={
            ASR_PRIMARY_MODEL_ID: "fumée visible",
            ASR_CHALLENGER_MODEL_ID: "Fumée visible.",
        }
    )
    execution = SessionRunner(
        registry=_registry_with_asr_group(
            strategy=ConsensusStrategy.QUORUM,
            minimum_successful=2,
            minimum_agreeing=2,
        ),
        adapter_factory=factory,
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert execution.output.status == "succeeded"
    assert execution.output.items[0].transcript.segments[0].text == "fumée visible"
    asr_consensus = next(
        result for result in execution.consensus_results if result.stage_role == "asr"
    )
    assert asr_consensus.decision == "pass"
    assert asr_consensus.selected_candidate_id == "asr.primary"
    assert asr_consensus.downstream_allowed is True
    assert asr_consensus.agreement_score == 1.0
    assert [run.candidate_id for run in execution.candidate_runs if run.stage_role == "asr"] == [
        "asr.primary",
        "asr.challenger",
    ]


def test_quorum_disagreement_is_resolved_by_the_large_qwen_judge() -> None:
    factory = ConsensusFactory(
        transcripts_by_model={
            ASR_PRIMARY_MODEL_ID: "fumée visible",
            ASR_CHALLENGER_MODEL_ID: "aucun signe de feu",
        },
        judge_selected_candidate_id="asr.challenger",
        judge_confidence=0.93,
    )
    execution = SessionRunner(
        registry=_registry_with_asr_group(
            strategy=ConsensusStrategy.QUORUM,
            minimum_successful=2,
            minimum_agreeing=2,
        ),
        adapter_factory=factory,
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert execution.output.status == "succeeded"
    assert execution.output.items[0].transcript.segments[0].text == "aucun signe de feu"
    asr_consensus = next(
        result for result in execution.consensus_results if result.stage_role == "asr"
    )
    assert asr_consensus.decision == "adjudicated"
    assert asr_consensus.selected_candidate_id == "asr.challenger"
    assert asr_consensus.adjudicator_candidate_id == "asr.judge"
    assert asr_consensus.downstream_allowed is True
    assert factory.judge_calls == ["load", "adjudicate:asr:False", "unload"]
    judge_run = next(run for run in execution.candidate_runs if run.candidate_id == "asr.judge")
    assert judge_run.model_role == "consensus_judge"
    assert judge_run.status == "succeeded"


def test_qwen_judge_low_confidence_abstains_without_releasing_a_candidate() -> None:
    factory = ConsensusFactory(
        transcripts_by_model={
            ASR_PRIMARY_MODEL_ID: "fumée visible",
            ASR_CHALLENGER_MODEL_ID: "aucun signe de feu",
        },
        judge_selected_candidate_id="asr.challenger",
        judge_confidence=0.2,
    )
    execution = SessionRunner(
        registry=_registry_with_asr_group(
            strategy=ConsensusStrategy.QUORUM,
            minimum_successful=2,
            minimum_agreeing=2,
        ),
        adapter_factory=factory,
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert execution.output.status == "partial_failure"
    assert execution.output.items[0].transcript.segments == ()
    asr_consensus = next(
        result for result in execution.consensus_results if result.stage_role == "asr"
    )
    assert asr_consensus.decision == "human_review"
    assert asr_consensus.selected_candidate_id is None
    assert asr_consensus.adjudicator_candidate_id == "asr.judge"
    assert asr_consensus.downstream_allowed is False
    assert asr_consensus.reason_codes[0] == "adjudicator_abstained"


def test_cascade_does_not_load_the_challenger_for_a_confident_primary() -> None:
    factory = ConsensusFactory(
        transcripts_by_model={
            ASR_PRIMARY_MODEL_ID: "fumée visible",
            ASR_CHALLENGER_MODEL_ID: "texte du challenger",
        }
    )
    execution = SessionRunner(
        registry=_registry_with_asr_group(
            strategy=ConsensusStrategy.CASCADE,
            minimum_successful=1,
            minimum_agreeing=1,
        ),
        adapter_factory=factory,
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert ASR_CHALLENGER_MODEL_ID not in factory.created_models
    asr_consensus = next(
        result for result in execution.consensus_results if result.stage_role == "asr"
    )
    assert asr_consensus.candidate_ids == ("asr.primary",)
    assert asr_consensus.reason_codes == ("cascade_primary_selected",)


def test_cascade_uses_the_challenger_when_primary_output_is_insufficient() -> None:
    factory = ConsensusFactory(
        transcripts_by_model={
            ASR_PRIMARY_MODEL_ID: None,
            ASR_CHALLENGER_MODEL_ID: "fumée visible",
        }
    )
    execution = SessionRunner(
        registry=_registry_with_asr_group(
            strategy=ConsensusStrategy.CASCADE,
            minimum_successful=1,
            minimum_agreeing=1,
        ),
        adapter_factory=factory,
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert execution.output.status == "succeeded"
    assert execution.output.items[0].transcript.segments[0].text == "fumée visible"
    assert ASR_CHALLENGER_MODEL_ID in factory.created_models
    asr_consensus = next(
        result for result in execution.consensus_results if result.stage_role == "asr"
    )
    assert asr_consensus.selected_candidate_id == "asr.challenger"
    assert asr_consensus.reason_codes == ("cascade_fallback_selected",)


def test_run_trace_records_preflight_and_postflight_for_every_executed_stage() -> None:
    execution = SessionRunner(
        registry=_registry(), adapter_factory=FakeFactory(), memory=FakeMemory()
    ).run_with_trace(_batch())

    assert len(execution.contract_digest) == 64
    assert [record.phase for record in execution.gate_records].count("preflight") == 4
    assert [record.phase for record in execution.gate_records].count("postflight") == 4
    assert (
        next(
            record
            for record in execution.gate_records
            if record.role == StageRole.MULTIMODAL_EXTRACTION and record.phase == "postflight"
        ).decision
        == GateDecision.PROCEED
    )


def test_a_model_failure_preserves_previous_stage_results() -> None:
    output = SessionRunner(
        registry=_registry(),
        adapter_factory=FakeFactory(failing_role="visual_grounding"),
        memory=FakeMemory(),
    ).run(_batch())
    assert output.status == "partial_failure"
    assert output.items[0].transcript.segments
    assert output.items[0].pixel_regions
    failed = next(run for run in output.model_runs if run.model_role == "visual_grounding")
    assert failed.status == "failed"
    assert failed.error_code == "model_runtime_error"


def test_downstream_model_is_not_started_when_upstream_left_no_usable_evidence() -> None:
    batch = WorkerInput.model_validate(
        {
            "batch_id": "BATCH-AUDIO",
            "batch_type": "user_media",
            "priority": "user_deadline",
            "items": [
                {
                    "input_id": "AUDIO-1",
                    "media_type": "audio",
                    "audio_url": "https://media.internal/audio.wav",
                }
            ],
        }
    )
    factory = FakeFactory(failing_role="asr")

    execution = SessionRunner(
        registry=_registry(), adapter_factory=factory, memory=FakeMemory()
    ).run_with_trace(batch)

    assert execution.output.status == "partial_failure"
    assert not any(name == "load:multimodal_extraction" for name, _ in factory.calls)
    multimodal_gate = next(
        record
        for record in execution.gate_records
        if record.role == StageRole.MULTIMODAL_EXTRACTION and record.phase == "preflight"
    )
    assert multimodal_gate.decision == GateDecision.HUMAN_REVIEW
    assert (
        next(
            run for run in execution.output.model_runs if run.model_role == "multimodal_extraction"
        ).error_code
        == "gate_required_capability_missing"
    )
    multimodal_trace = next(
        trace
        for trace in execution.stage_traces
        if trace.stage_role == StageRole.MULTIMODAL_EXTRACTION
    )
    assert multimodal_trace.status == "skipped"
    assert multimodal_trace.attempts == ()
    assert multimodal_trace.postflight is None


@dataclass(slots=True)
class EmptyMultimodalAdapter(FakeAdapter):
    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        if self.spec.role == "multimodal_extraction":
            self.calls.append((f"infer:{self.spec.role}", correction))
            return {}
        return FakeAdapter.infer(self, items, accumulated, correction=correction)


@dataclass(slots=True)
class EmptyMultimodalFactory(FakeFactory):
    def create(self, spec: ModelSpec) -> FakeAdapter:
        return EmptyMultimodalAdapter(spec, self.calls, self.failing_role)


def test_schema_valid_but_empty_multimodal_output_requires_human_review() -> None:
    execution = SessionRunner(
        registry=_registry(),
        adapter_factory=EmptyMultimodalFactory(),
        memory=FakeMemory(),
    ).run_with_trace(_batch())

    assert execution.output.status == "partial_failure"
    assert "gate:multimodal_extraction:minimum_output_missing" in (
        execution.output.validation_errors
    )
    assert (
        next(
            record
            for record in execution.gate_records
            if record.role == StageRole.MULTIMODAL_EXTRACTION and record.phase == "postflight"
        ).decision
        == GateDecision.HUMAN_REVIEW
    )


@dataclass(slots=True)
class CorrectionAdapter(FakeAdapter):
    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        if self.spec.role != "multimodal_extraction" or correction:
            return FakeAdapter.infer(self, items, accumulated, correction=correction)
        self.calls.append((f"infer:{self.spec.role}", correction))
        item = items[0]
        return {
            item.input_id: ItemPatch(
                factual_observations=(
                    FactualObservation(
                        type="smoke_visible",
                        evidence_kind="frame",
                        evidence_id="frame-1",
                        region_id="frame-1:det:0001",
                        description="La fumée pourrait atteindre la route.",
                        certainty="directly_visible",
                    ),
                )
            )
        }


@dataclass(slots=True)
class CorrectionFactory(FakeFactory):
    def create(self, spec: ModelSpec) -> FakeAdapter:
        return CorrectionAdapter(spec, self.calls, self.failing_role)


def test_qwen_gets_one_strict_correction_after_invalid_output() -> None:
    factory = CorrectionFactory()
    execution = SessionRunner(
        registry=_registry(), adapter_factory=factory, memory=FakeMemory()
    ).run_with_trace(_batch())
    qwen_calls = [
        correction for name, correction in factory.calls if name == "infer:multimodal_extraction"
    ]
    assert qwen_calls == [False, True]
    assert execution.output.status == "succeeded"
    assert execution.output.items[0].factual_observations[0].description == (
        "Une fumée est directement visible."
    )
    qwen_trace = next(
        trace
        for trace in execution.stage_traces
        if trace.stage_role == StageRole.MULTIMODAL_EXTRACTION
    )
    assert [(attempt.kind, attempt.status) for attempt in qwen_trace.attempts] == [
        ("initial", "failed"),
        ("repair", "succeeded"),
    ]
    assert qwen_trace.attempts[0].error_code == "invalid_model_output"
    assert all(
        len(trace.attempts) == 1
        for trace in execution.stage_traces
        if trace.stage_role != StageRole.MULTIMODAL_EXTRACTION and trace.status == "succeeded"
    )


@dataclass(slots=True)
class MalformedAdapter(FakeAdapter):
    def infer(
        self,
        items: Sequence[BatchItem],
        accumulated: Mapping[str, ItemPatch],
        *,
        correction: bool = False,
    ) -> Mapping[str, ItemPatch]:
        if self.spec.role == "multimodal_extraction" and not correction:
            self.calls.append((f"infer:{self.spec.role}", correction))
            raise ModelOutputError("invalid JSON")
        return FakeAdapter.infer(self, items, accumulated, correction=correction)


@dataclass(slots=True)
class MalformedFactory(FakeFactory):
    def create(self, spec: ModelSpec) -> FakeAdapter:
        return MalformedAdapter(spec, self.calls, self.failing_role)


def test_malformed_qwen_json_also_gets_one_correction() -> None:
    factory = MalformedFactory()
    output = SessionRunner(registry=_registry(), adapter_factory=factory, memory=FakeMemory()).run(
        _batch()
    )
    qwen_calls = [
        correction for name, correction in factory.calls if name == "infer:multimodal_extraction"
    ]
    assert qwen_calls == [False, True]
    assert output.status == "succeeded"


def test_ten_cycles_release_every_loaded_adapter() -> None:
    factory = FakeFactory()
    runner = SessionRunner(registry=_registry(), adapter_factory=factory, memory=FakeMemory())
    for _ in range(10):
        assert runner.run(_batch()).status == "succeeded"
    loads = sum(name.startswith("load:") for name, _ in factory.calls)
    unloads = sum(name.startswith("unload:") for name, _ in factory.calls)
    assert loads == unloads == 40


def test_rtdetr_prioritizes_targets_without_dropping_all_context() -> None:
    evidence_ids = [f"frame-{index:02d}" for index in range(12)]
    selected = RTDETRAdapter._select_sources(
        evidence_ids,
        {f"frame-{index:02d}": 1 - index / 100 for index in range(10)},
        limit=8,
    )

    assert len(selected) == 8
    assert set(evidence_ids[:6]).issubset(selected)
    assert len(selected - set(evidence_ids[:6])) == 2


def test_missing_detector_checkpoint_marks_visual_package_partial() -> None:
    registry = _registry()
    registry.pop("fire_detection")

    output = SessionRunner(
        registry=registry, adapter_factory=FakeFactory(), memory=FakeMemory()
    ).run(_batch())

    assert output.status == "partial_failure"
    assert output.items[0].visual_evidence_selection[0].selection_reason == "single_image"
    assert "fire_detection:checkpoint_not_configured" in output.validation_errors
