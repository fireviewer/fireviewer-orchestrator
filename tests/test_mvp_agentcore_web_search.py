from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_orchestrator.mvp.orchestration import merge_event_evidence
from fireviewer_contracts.mvp.providers import ResearchRequest
from fireviewer_evidence_ingestion.mvp.research import AgentCoreWebSearchProvider


class _FakeMcpClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call_tool(self, *, name: str, arguments: dict[str, object]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.response


NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _response(*results: object, is_error: bool = False) -> dict[str, Any]:
    return {
        "isError": is_error,
        "content": [
            {
                "type": "text",
                "text": json.dumps({"id": "fixture", "results": list(results)}),
            }
        ],
    }


def test_agentcore_search_maps_citations_and_date_filters_to_event_evidence() -> None:
    client = _FakeMcpClient(
        _response(
            {
                "text": "Le feu a parcouru 120 hectares.",
                "publishedDate": "2026-07-12",
                "url": "HTTPS://Example.com:443/incendie?id=7#fragment",
                "title": "Incendie de Test",
            },
            {
                "text": "Une route a été fermée.",
                "publishedDate": "2026-07-12T10:30:00Z",
                "url": "https://example.com/route",
                "title": "Circulation",
            },
        )
    )
    provider = AgentCoreWebSearchProvider(client=client, clock=lambda: NOW)
    request = ResearchRequest.model_validate(
        {
            "event_id": "EVENT-WEB-1",
            "query": "incendie test juillet 2026",
            "time_window": {
                "from_at": "2026-07-01T00:00:00+02:00",
                "to_at": "2026-07-31T23:59:59+02:00",
            },
        }
    )

    evidence = provider.search(request)

    assert client.calls == [
        (
            "WebSearch",
            {
                "query": "incendie test juillet 2026",
                "maxResults": 20,
                "filters": {
                    "publishedDateFilter": {
                        "from": "2026-06-30T22:00:00Z",
                        "to": "2026-07-31T21:59:59Z",
                    }
                },
            },
        )
    ]
    assert len(evidence.sources) == 2
    assert len(evidence.claims) == 2
    assert evidence.media == ()
    assert str(evidence.sources[0].source_url) == "https://example.com/incendie?id=7"
    assert evidence.sources[0].origin_id == evidence.sources[1].origin_id
    assert evidence.sources[0].publisher == "example.com"
    assert evidence.sources[0].published_at == datetime(2026, 7, 12, tzinfo=UTC)
    assert evidence.claims[0].text.startswith("Incendie de Test\n")
    assert evidence.uncertainties[0].code == "web_search_snippets_unverified"
    assert evidence.needs_human_review is True


def test_agentcore_search_skips_existing_url_and_merges_additively() -> None:
    existing = EventEvidenceV1.model_validate(
        {
            "event_id": "EVENT-WEB-2",
            "sources": [
                {
                    "source_id": "SOURCE-EXISTING",
                    "origin_id": "ORIGIN-EXISTING",
                    "source_url": "https://example.org/already-there",
                    "publisher": "example.org",
                    "retrieved_at": "2026-08-20T10:00:00Z",
                    "source_type": "press",
                    "independence_weight": 0.8,
                }
            ],
        }
    )
    client = _FakeMcpClient(
        _response(
            {
                "text": "Duplicate",
                "url": "https://EXAMPLE.org:443/already-there#top",
            },
            {
                "text": "Nouvelle source",
                "url": "https://service-public.fr/fire/42",
            },
        )
    )
    provider = AgentCoreWebSearchProvider(client=client, clock=lambda: NOW)

    delta = provider.search(
        ResearchRequest(
            event_id=existing.event_id,
            query="incendie événement 2",
            time_window=existing.time_window,
            existing_evidence=existing,
        )
    )
    merged = merge_event_evidence(existing, delta)

    assert len(delta.sources) == 1
    assert len(merged.sources) == 2
    assert len(merged.claims) == 1


def test_agentcore_search_returns_explicit_abstention_on_mcp_error() -> None:
    provider = AgentCoreWebSearchProvider(
        client=_FakeMcpClient({"isError": True, "content": []}),
        clock=lambda: NOW,
    )

    evidence = provider.search(ResearchRequest(event_id="EVENT-WEB-3", query="fire"))

    assert evidence.sources == ()
    assert evidence.claims == ()
    assert [item.code for item in evidence.uncertainties] == ["web_search_provider_error"]
    assert evidence.needs_human_review is True


def test_agentcore_search_rejects_overlong_query_before_call() -> None:
    client = _FakeMcpClient(_response())
    provider = AgentCoreWebSearchProvider(client=client, clock=lambda: NOW)

    with pytest.raises(ValueError, match="at most 200"):
        provider.search(ResearchRequest(event_id="EVENT-WEB-4", query="x" * 201))

    assert client.calls == []
