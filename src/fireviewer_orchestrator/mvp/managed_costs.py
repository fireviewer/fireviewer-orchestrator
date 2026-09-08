from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

MILLION = Decimal("1000000")
THOUSAND = Decimal("1000")
GB = Decimal(1_000_000_000)


@dataclass(frozen=True, slots=True)
class ManagedMvpRates:
    nova_input_per_million_tokens_usd: Decimal = Decimal("0.30")
    nova_output_per_million_tokens_usd: Decimal = Decimal("2.50")
    agentcore_web_search_per_thousand_usd: Decimal = Decimal("7.00")
    agentcore_gateway_per_thousand_calls_usd: Decimal = Decimal("0.005")
    azure_blob_hot_lrs_per_gb_month_usd: Decimal = Decimal("0.0192")
    azure_blob_write_per_ten_thousand_usd: Decimal = Decimal("0.059")
    azure_blob_read_per_ten_thousand_usd: Decimal = Decimal("0.0047")
    container_apps_free_vcpu_seconds: Decimal = Decimal("180000")
    container_apps_free_gib_seconds: Decimal = Decimal("360000")
    container_apps_free_requests: int = 2_000_000
    modal_cpu_per_physical_core_second_usd: Decimal = Decimal("0.0000131")
    modal_memory_per_gib_second_usd: Decimal = Decimal("0.00000222")
    modal_monthly_compute_credit_usd: Decimal = Decimal("30")
    beam_cpu_per_physical_core_second_usd: Decimal = Decimal("0.0000125")
    beam_memory_per_gib_second_usd: Decimal = Decimal("0.0000021")
    beam_monthly_compute_credit_usd: Decimal = Decimal("30")


@dataclass(frozen=True, slots=True)
class ManagedMvpScenario:
    name: str
    event_count: int
    media_per_event: int = 20
    search_queries_per_event: int = 20
    research_input_tokens_per_event: int = 20_000
    research_output_tokens_per_event: int = 5_000
    vision_prompt_tokens_per_media: int = 500
    vision_output_tokens_per_media: int = 300
    image_tokens_per_media: int = 230
    nova_verification_media_per_event: int = 0
    detector_vcpus: Decimal = Decimal("4")
    detector_memory_gib: Decimal = Decimal("8")
    detector_active_seconds_per_event: Decimal = Decimal("120")
    satellite_composites_per_event: int = 3
    reference_storage_bytes: int = 2_082_475_200
    reference_storage_events: int = 9

    def __post_init__(self) -> None:
        integer_values = (
            self.event_count,
            self.media_per_event,
            self.search_queries_per_event,
            self.research_input_tokens_per_event,
            self.research_output_tokens_per_event,
            self.vision_prompt_tokens_per_media,
            self.vision_output_tokens_per_media,
            self.image_tokens_per_media,
            self.satellite_composites_per_event,
            self.reference_storage_bytes,
            self.reference_storage_events,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("managed MVP scenario values must be positive")
        if not 0 <= self.nova_verification_media_per_event <= self.media_per_event:
            raise ValueError("Nova verification media must be between zero and media_per_event")
        detector_values = (
            self.detector_vcpus,
            self.detector_memory_gib,
            self.detector_active_seconds_per_event,
        )
        if any(value <= 0 for value in detector_values):
            raise ValueError("CPU detector resource assumptions must be positive")


def _token_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    rates: ManagedMvpRates,
) -> Decimal:
    return (
        Decimal(input_tokens) * rates.nova_input_per_million_tokens_usd / MILLION
        + Decimal(output_tokens) * rates.nova_output_per_million_tokens_usd / MILLION
    )


def calculate_managed_mvp_cost(
    scenario: ManagedMvpScenario,
    *,
    rates: ManagedMvpRates | None = None,
) -> dict[str, Decimal | bool | int | str]:
    selected_rates = rates or ManagedMvpRates()
    vision_media = scenario.event_count * scenario.media_per_event
    nova_verification_media = (
        scenario.event_count * scenario.nova_verification_media_per_event
    )
    satellite_media = scenario.event_count * scenario.satellite_composites_per_event
    searches = scenario.event_count * scenario.search_queries_per_event

    nova_verification = _token_cost(
        input_tokens=nova_verification_media
        * (scenario.image_tokens_per_media + scenario.vision_prompt_tokens_per_media),
        output_tokens=nova_verification_media * scenario.vision_output_tokens_per_media,
        rates=selected_rates,
    )
    detector_vcpu_seconds = (
        Decimal(scenario.event_count)
        * scenario.detector_active_seconds_per_event
        * scenario.detector_vcpus
    )
    detector_gib_seconds = (
        Decimal(scenario.event_count)
        * scenario.detector_active_seconds_per_event
        * scenario.detector_memory_gib
    )
    detector_requests = vision_media
    detector_physical_cores = scenario.detector_vcpus / Decimal(2)
    detector_within_free_grant = (
        detector_vcpu_seconds <= selected_rates.container_apps_free_vcpu_seconds
        and detector_gib_seconds <= selected_rates.container_apps_free_gib_seconds
        and detector_requests <= selected_rates.container_apps_free_requests
    )
    if not detector_within_free_grant:
        raise ValueError(
            "CPU detector scenario exceeds the Azure Container Apps monthly free grant; "
            "refresh the regional PAYG rates before estimating it"
        )
    detector_compute = Decimal(0)
    detector_active_seconds = (
        Decimal(scenario.event_count) * scenario.detector_active_seconds_per_event
    )
    modal_detector_gross = detector_active_seconds * (
        detector_physical_cores
        * selected_rates.modal_cpu_per_physical_core_second_usd
        + scenario.detector_memory_gib
        * selected_rates.modal_memory_per_gib_second_usd
    )
    beam_detector_gross = detector_active_seconds * (
        detector_physical_cores * selected_rates.beam_cpu_per_physical_core_second_usd
        + scenario.detector_memory_gib
        * selected_rates.beam_memory_per_gib_second_usd
    )
    satellite = _token_cost(
        input_tokens=satellite_media
        * (scenario.image_tokens_per_media + scenario.vision_prompt_tokens_per_media),
        output_tokens=satellite_media * scenario.vision_output_tokens_per_media,
        rates=selected_rates,
    )
    research = _token_cost(
        input_tokens=scenario.event_count * scenario.research_input_tokens_per_event,
        output_tokens=scenario.event_count * scenario.research_output_tokens_per_event,
        rates=selected_rates,
    )
    web_search = (
        Decimal(searches) * selected_rates.agentcore_web_search_per_thousand_usd / THOUSAND
    )
    gateway = (
        Decimal(searches) * selected_rates.agentcore_gateway_per_thousand_calls_usd / THOUSAND
    )

    storage_bytes = (
        Decimal(scenario.reference_storage_bytes)
        * Decimal(scenario.event_count)
        / Decimal(scenario.reference_storage_events)
    )
    storage_gb = storage_bytes / GB
    storage = storage_gb * selected_rates.azure_blob_hot_lrs_per_gb_month_usd
    write_requests = 2 * (vision_media + satellite_media)
    read_requests = vision_media + satellite_media
    azure_blob_requests = (
        Decimal(write_requests)
        * selected_rates.azure_blob_write_per_ten_thousand_usd
        / Decimal(10_000)
        + Decimal(read_requests)
        * selected_rates.azure_blob_read_per_ten_thousand_usd
        / Decimal(10_000)
    )

    aws_total = (
        nova_verification
        + satellite
        + research
        + web_search
        + gateway
    )
    azure_total = (
        detector_compute
        + storage
        + azure_blob_requests
    )
    money_quantum = Decimal("0.000001")
    return {
        "scenario": scenario.name,
        "events": scenario.event_count,
        "vision_media": vision_media,
        "detector_model_id": "mfranzon/fire-smoke-yolov8",
        "detector_model_revision": "f1c6426b069c1849cbf13b1ef5d2a260289286db",
        "detector_vcpu_seconds": detector_vcpu_seconds,
        "detector_gib_seconds": detector_gib_seconds,
        "detector_requests": detector_requests,
        "detector_within_azure_monthly_free_grant": detector_within_free_grant,
        "detector_compute_usd": detector_compute.quantize(money_quantum),
        "modal_detector_gross_usd": modal_detector_gross.quantize(money_quantum),
        "modal_detector_net_after_monthly_credit_usd": max(
            Decimal(0),
            modal_detector_gross - selected_rates.modal_monthly_compute_credit_usd,
        ).quantize(money_quantum),
        "beam_detector_gross_usd": beam_detector_gross.quantize(money_quantum),
        "beam_detector_net_after_monthly_credit_usd": max(
            Decimal(0),
            beam_detector_gross - selected_rates.beam_monthly_compute_credit_usd,
        ).quantize(money_quantum),
        "nova_verification_media": nova_verification_media,
        "satellite_composites": satellite_media,
        "web_search_queries": searches,
        "storage_gb_month": storage_gb.quantize(Decimal("0.0001")),
        "nova_verification_usd": nova_verification.quantize(money_quantum),
        "satellite_usd": satellite.quantize(money_quantum),
        "research_synthesis_usd": research.quantize(money_quantum),
        "web_search_usd": web_search.quantize(money_quantum),
        "agentcore_gateway_usd": gateway.quantize(money_quantum),
        "azure_blob_storage_list_usd": storage.quantize(money_quantum),
        "azure_blob_requests_list_usd": azure_blob_requests.quantize(money_quantum),
        "aws_total_usd": aws_total.quantize(money_quantum),
        "azure_total_list_usd": azure_total.quantize(money_quantum),
    }


__all__ = [
    "ManagedMvpRates",
    "ManagedMvpScenario",
    "calculate_managed_mvp_cost",
]
