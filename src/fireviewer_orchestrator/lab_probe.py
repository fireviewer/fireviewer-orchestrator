"""Opt-in component validation route alongside the unchanged full job handler."""
import json
from time import perf_counter
from typing import Any, Literal
from pydantic import Field
from fireviewer_contracts.contracts import StrictModel, WorkerInput
from fireviewer_contracts.model_registry import CONSENSUS_JUDGE
from fireviewer_contracts.mvp_stack import load_mvp_stack, mvp_stack_digest
from fireviewer_vision_runtime.consensus import JudgeCandidate


class ProbeCandidate(StrictModel):
    candidate_id: str = Field(pattern=r'^[a-z0-9_]{1,64}$')
    output_payload: dict[str, Any]


class JudgeProbe(StrictModel):
    schema_version: Literal['lab-bonsai-integration-1']
    batch: WorkerInput
    stage_role: Literal['source_research', 'fire_detection', 'visual_grounding']
    candidates: tuple[ProbeCandidate, ...] = Field(min_length=2, max_length=3)
    comparison: dict[str, Any]


def run_probe(raw_input, factory):
    if len(json.dumps(raw_input).encode()) > 64000:
        raise ValueError('lab_probe_too_large')
    request = JudgeProbe.model_validate(raw_input)
    judge = factory.create_consensus_judge(CONSENSUS_JUDGE)
    started = perf_counter()
    try:
        judge.load()
        load_ms = round(1000*(perf_counter()-started), 2)
        verdict = judge.adjudicate(batch=request.batch, stage_role=request.stage_role,
            candidates=tuple(JudgeCandidate(c.candidate_id, 'controlled_integration_probe', 'lab-v1', c.output_payload)
                             for c in request.candidates), comparison_payload=request.comparison)
        return {'schema_version': 'lab-bonsai-integration-result-1', 'scope': 'component_integration_probe',
                'full_pipeline_result': False, 'accuracy_benchmark': False,
                'stack_id': load_mvp_stack().stack_id, 'stack_digest': mvp_stack_digest(),
                'load_ms': load_ms, 'verdict': verdict.output_payload}
    finally:
        judge.unload()
