from __future__ import annotations

from decimal import Decimal

from fireviewer_contracts.mvp.contracts import EvidenceMedia
from fireviewer_orchestrator.mvp.managed_costs import (
    ManagedMvpScenario,
    calculate_managed_mvp_cost,
)
from fireviewer_vision_runtime.mvp.vision.bedrock_nova import (
    BedrockImage,
    BedrockNovaVisionProvider,
)


def _media() -> EvidenceMedia:
    return EvidenceMedia(
        media_id="MEDIA-1",
        source_id="SOURCE-1",
        media_group_id="GROUP-1",
        origin_id="ORIGIN-1",
        kind="photo",
        sha256="a" * 64,
    )


class _ImageLoader:
    def load(self, media: EvidenceMedia) -> BedrockImage:
        return BedrockImage(data=b"jpeg-fixture", format="jpeg")


class _BedrockClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 730, "outputTokens": 300},
        }


def test_bedrock_nova_provider_normalizes_managed_boxes_and_records_cost() -> None:
    client = _BedrockClient(
        '{"detections":['
        '{"class":"smoke","bbox":[100,200,700,800],"score":0.82},'
        '{"class":"fire","bbox":[300,600,500,900],"score":0.74}'
        "]}"
    )
    provider = BedrockNovaVisionProvider(image_loader=_ImageLoader(), client=client)

    result = provider.detect(_media())

    assert result.status == "fire_and_smoke"
    assert result.needs_human_review is True
    assert result.detections[0].bbox == (0.1, 0.2, 0.7, 0.8)
    assert result.provider_run.cost_usd == 0.000969
    assert client.calls[0]["modelId"] == "eu.amazon.nova-2-lite-v1:0"
    assert client.calls[0]["inferenceConfig"] == {"maxTokens": 1024, "temperature": 0}


def test_bedrock_nova_provider_abstains_on_invalid_model_output() -> None:
    provider = BedrockNovaVisionProvider(
        image_loader=_ImageLoader(),
        client=_BedrockClient("not valid JSON"),
    )

    result = provider.detect(_media())

    assert result.status == "uncertain"
    assert result.detections == ()
    assert result.needs_human_review is True


def test_managed_mvp_cost_is_bounded_for_initial_and_extended_corpus() -> None:
    initial = calculate_managed_mvp_cost(
        ManagedMvpScenario(name="initial-9-events", event_count=9)
    )
    extended = calculate_managed_mvp_cost(
        ManagedMvpScenario(name="extended-30-events", event_count=30)
    )

    assert initial["web_search_queries"] == 180
    assert initial["detector_model_id"] == "mfranzon/fire-smoke-yolov8"
    assert initial["detector_model_revision"] == (
        "f1c6426b069c1849cbf13b1ef5d2a260289286db"
    )
    assert initial["detector_vcpu_seconds"] == Decimal("4320")
    assert initial["detector_gib_seconds"] == Decimal("8640")
    assert initial["detector_within_azure_monthly_free_grant"] is True
    assert initial["detector_compute_usd"] == Decimal("0.000000")
    assert initial["modal_detector_gross_usd"] == Decimal("0.047477")
    assert initial["beam_detector_gross_usd"] == Decimal("0.045144")
    assert initial["nova_verification_media"] == 0
    assert initial["aws_total_usd"] == Decimal("1.453563")
    assert initial["azure_total_list_usd"] == Decimal("0.042523")
    assert extended["web_search_queries"] == 600
    assert extended["detector_vcpu_seconds"] == Decimal("14400")
    assert extended["detector_gib_seconds"] == Decimal("28800")
    assert extended["modal_detector_gross_usd"] == Decimal("0.158256")
    assert extended["modal_detector_net_after_monthly_credit_usd"] == Decimal("0.000000")
    assert extended["beam_detector_gross_usd"] == Decimal("0.150480")
    assert extended["beam_detector_net_after_monthly_credit_usd"] == Decimal("0.000000")
    assert extended["aws_total_usd"] == Decimal("4.845210")
    assert extended["azure_total_list_usd"] == Decimal("0.141745")


def test_nova_vision_is_an_explicit_optional_verification_cost() -> None:
    scenario = ManagedMvpScenario(
        name="one-event-with-verification",
        event_count=1,
        nova_verification_media_per_event=4,
    )

    result = calculate_managed_mvp_cost(scenario)

    assert result["nova_verification_media"] == 4
    assert result["nova_verification_usd"] == Decimal("0.003876")
