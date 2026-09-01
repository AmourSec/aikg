from __future__ import annotations

import json

from runtime.ascend_kg import AscendKgProvider
from runtime.contracts import (
    ActivationConsent,
    CandidateId,
    Candidates,
    LoadRequest,
    NetworkConsent,
    ResponseToken,
    SearchQuery,
    SearchRequest,
    TransportResponse,
    prepare_search,
)
from tests.fakes import FakeSleeper, FakeTransport


def make_provider(
    responses: tuple[TransportResponse, ...] = (),
    api_key: str | None = "secret-key",
) -> tuple[AscendKgProvider, FakeTransport, FakeSleeper]:
    transport = FakeTransport(responses)
    sleeper = FakeSleeper()
    return (
        AscendKgProvider(
            transport=transport,
            sleeper=sleeper,
            api_key=api_key,
        ),
        transport,
        sleeper,
    )


def search_request(
    consent: NetworkConsent = NetworkConsent.GRANTED,
    query: SearchQuery = SearchQuery("find an NPU skill"),
) -> SearchRequest:
    return SearchRequest(
        prepared=prepare_search(query),
        network_consent=consent,
    )


def load_request(
    candidate_id: str = "remote-1",
    *,
    network: NetworkConsent = NetworkConsent.GRANTED,
    activation: ActivationConsent = ActivationConsent.GRANTED,
    member_ids: tuple[str, ...] = ("remote-1",),
) -> LoadRequest:
    return LoadRequest(
        response_token=ResponseToken(
            candidate_ids=tuple(CandidateId(value) for value in member_ids),
        ),
        candidate_id=CandidateId(candidate_id),
        network_consent=network,
        activation_consent=activation,
    )


def json_response(payload: bytes | list[dict[str, str | int | float]] | dict) -> TransportResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return TransportResponse(status_code=200, headers=(), body=body)


def status_response(status_code: int, body: bytes = b"") -> TransportResponse:
    return TransportResponse(status_code=status_code, headers=(), body=body)


def candidate_search_response(*candidate_ids: str) -> TransportResponse:
    return json_response(
        {
            "results": [
                {
                    "id": candidate_id,
                    "source_repo": "org/skills",
                    "source_file": f"skills/{candidate_id}/SKILL.md",
                }
                for candidate_id in candidate_ids
            ],
        },
    )


def issued_load_request(
    provider: AscendKgProvider,
    candidate_id: str = "remote-1",
) -> LoadRequest:
    result = provider.search(search_request())
    assert isinstance(result, Candidates)
    return LoadRequest(
        response_token=result.response_token,
        candidate_id=CandidateId(candidate_id),
        network_consent=NetworkConsent.GRANTED,
        activation_consent=ActivationConsent.GRANTED,
    )
