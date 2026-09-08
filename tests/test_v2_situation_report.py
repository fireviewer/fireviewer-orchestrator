from __future__ import annotations

from fireviewer_contracts.resources import schema_path

import json
from pathlib import Path

from fireviewer_contracts.contracts import FactProposalV2, WorkerInputV2, WorkerItemResultV2
from fireviewer_orchestrator.v2_runner import _fact_category, _report, to_legacy_input

EXAMPLES = schema_path("agent-worker/v2/examples/valid-input.json").parent


def test_private_report_keeps_press_claim_attributed_and_pending_review() -> None:
    payload = json.loads((EXAMPLES / "valid-input.json").read_text(encoding="utf-8"))
    provenance = payload["items"][0]["provenance"]
    provenance.update(
        {
            "source_kind": "press",
            "source_confidence": "lead",
            "source_policy_domain": "press.example",
            "publication_policy": "private_analysis_only",
            "claim_types": ["reported_situational_claim", "media_asset"],
        }
    )
    batch = WorkerInputV2.model_validate(payload)
    fact = FactProposalV2(
        fact_id="FACT-PERSONNEL-1",
        input_id="INPUT-1",
        category="resources",
        fact_key="personnel_engaged",
        as_of="2026-07-09T19:30:00Z",
        evidence_kind="article_text",
        evidence_id="INPUT-1",
        certainty="explicitly_written",
        value_text="La presse rapporte 420 pompiers engagés.",
        summary="420 pompiers annoncés",
    )

    report = _report(
        batch,
        (WorkerItemResultV2(input_id="INPUT-1", fact_proposals=(fact,)),),
    )

    assert "420 pompiers annoncés" in report.body_markdown
    assert "rapporté, à recouper" in report.body_markdown
    assert "Photographe de test" in report.body_markdown
    assert any(section.key == "sources_and_freshness" for section in report.sections)


def test_situational_fact_types_route_to_expected_report_categories() -> None:
    assert _fact_category("aircraft_engaged") == "resources"
    assert _fact_category("evacuation_count") == "evacuation"
    assert _fact_category("air_quality_alert") == "weather"
    assert _fact_category("public_relief_and_donations") == "resources"
    assert _fact_category("access_restriction") == "access"
    assert _fact_category("burned_area") == "burned_area"


def test_public_declared_observation_reaches_the_sequential_model_context() -> None:
    payload = json.loads((EXAMPLES / "valid-input.json").read_text(encoding="utf-8"))
    payload["items"][0]["provenance"]["declared_observation"] = {
        "observed_at": "2026-07-10T09:00:00+02:00",
        "observation_type": "front de flammes",
        "direct_observation": True,
        "description": (
            "Un front de flammes est visible sur le versant au-dessus de la commune fictive."
        ),
        "location_mode": "place",
        "location_label": "Commune fictive, secteur de démonstration",
        "latitude": None,
        "longitude": None,
        "uncertainty_m": None,
        "media_captured_at": "2026-07-10T08:58:00+02:00",
        "media_direction": "vers le nord-est",
    }
    payload["items"][0]["camera"] = None

    legacy = to_legacy_input(WorkerInputV2.model_validate(payload))

    declared = legacy.items[0].source_context.declared_observation
    assert declared is not None
    assert declared.description.startswith("Un front de flammes")
    assert declared.location_label == "Commune fictive, secteur de démonstration"
    assert legacy.items[0].metadata.latitude is None
    assert legacy.items[0].metadata.longitude is None
