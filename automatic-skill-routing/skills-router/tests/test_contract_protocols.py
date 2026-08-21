from __future__ import annotations

import unittest

from runtime.contracts import (
    ActivationConsent,
    ByteLimit,
    CandidateId,
    Candidates,
    DisplayName,
    Endpoint,
    HeaderName,
    HeaderValue,
    HttpMethod,
    LoadRequest,
    NetworkConsent,
    Provider,
    ProviderId,
    ProviderScore,
    RemoteCandidate,
    RemoteSkillContent,
    ResponseToken,
    SearchQuery,
    SearchRequest,
    Seconds,
    Sleeper,
    SourceFile,
    SourceRepo,
    Transport,
    TransportRequest,
    TransportResponse,
    TransportTimeout,
    TransportUnavailable,
    UntrustedText,
    UnavailableReason,
    prepare_search,
)
from tests.fakes import FakeProvider, FakeSleeper, FakeTransport
class ProviderFakeTests(unittest.TestCase):
    def test_fake_provider_scripts_search_and_load_skill_results(self) -> None:
        # Given
        token = ResponseToken()
        search_request = SearchRequest(
            prepared=prepare_search(SearchQuery("find an NPU skill")),
            network_consent=NetworkConsent.GRANTED,
        )
        load_request = LoadRequest(
            response_token=token,
            candidate_id=CandidateId("remote-1"),
            network_consent=NetworkConsent.GRANTED,
            activation_consent=ActivationConsent.GRANTED,
        )
        search_result = Candidates(
            response_token=token,
            candidates=(
                RemoteCandidate(
                    candidate_id=CandidateId("remote-1"),
                    provider_id=ProviderId("ascend-kg"),
                    display_name=DisplayName("remote skill"),
                    source_repo=SourceRepo("org/skills"),
                    source_file=SourceFile("skills/remote/SKILL.md"),
                    score=ProviderScore("provider-rank:alpha"),
                ),
            ),
        )
        load_result = RemoteSkillContent(
            response_token=token,
            candidate_id=CandidateId("remote-1"),
            content=UntrustedText("remote skill body"),
        )
        fake_provider = FakeProvider((search_result,), (load_result,))
        provider: Provider = fake_provider

        # When
        actual = (
            provider.search(search_request),
            provider.load_skill(load_request),
        )

        # Then
        self.assertEqual(actual, (search_result, load_result))
        self.assertEqual(fake_provider.search_requests, (search_request,))
        self.assertEqual(fake_provider.load_requests, (load_request,))


class TransportFakeTests(unittest.TestCase):
    def test_transport_request_keeps_all_http_controls_typed_and_immutable(self) -> None:
        # Given
        headers = (
            (HeaderName("accept"), HeaderValue("application/json")),
            (HeaderName("authorization"), HeaderValue("Bearer redacted")),
        )
        request = TransportRequest(
            method=HttpMethod.POST,
            endpoint=Endpoint("https://provider.invalid/search"),
            headers=headers,
            body=b"request",
            timeout_seconds=Seconds(3.0),
            max_response_bytes=ByteLimit(4096),
        )

        # When
        controls = (
            request.method,
            request.endpoint,
            request.headers,
            request.body,
            request.timeout_seconds,
            request.max_response_bytes,
        )

        # Then
        self.assertEqual(
            controls,
            (
                HttpMethod.POST,
                Endpoint("https://provider.invalid/search"),
                headers,
                b"request",
                Seconds(3.0),
                ByteLimit(4096),
            ),
        )
        self.assertIsInstance(request.headers, tuple)

    def test_fake_transport_returns_response_without_network(self) -> None:
        # Given
        request = TransportRequest(
            method=HttpMethod.GET,
            endpoint=Endpoint("https://provider.invalid/health"),
            headers=(),
            body=b"",
            timeout_seconds=Seconds(1.0),
            max_response_bytes=ByteLimit(1024),
        )
        response = TransportResponse(status_code=200, headers=(), body=b"ok")
        fake_transport = FakeTransport((response,))
        transport: Transport = fake_transport

        # When
        actual = transport.send(request)

        # Then
        self.assertIs(actual, response)
        self.assertEqual(fake_transport.requests, (request,))

    def test_fake_transport_returns_typed_timeout_instead_of_throwing(self) -> None:
        # Given
        request = TransportRequest(
            method=HttpMethod.POST,
            endpoint=Endpoint("https://provider.invalid/load"),
            headers=(),
            body=b"request",
            timeout_seconds=Seconds(2.0),
            max_response_bytes=ByteLimit(2048),
        )
        timeout = TransportTimeout(timeout_seconds=Seconds(2.0))
        fake_transport = FakeTransport((timeout,))
        transport: Transport = fake_transport

        # When
        actual = transport.send(request)

        # Then
        self.assertIs(actual, timeout)

    def test_fake_transport_returns_typed_connection_failure(self) -> None:
        # Given
        request = TransportRequest(
            method=HttpMethod.POST,
            endpoint=Endpoint("https://provider.invalid/search"),
            headers=(),
            body=b"request",
            timeout_seconds=Seconds(2.0),
            max_response_bytes=ByteLimit(2048),
        )
        unavailable = TransportUnavailable(reason=UnavailableReason.SERVICE)
        fake_transport = FakeTransport((unavailable,))
        transport: Transport = fake_transport

        # When
        actual = transport.send(request)

        # Then
        self.assertIs(actual, unavailable)


class SleeperFakeTests(unittest.TestCase):
    def test_fake_sleeper_records_delays_without_waiting(self) -> None:
        # Given
        fake_sleeper = FakeSleeper()
        sleeper: Sleeper = fake_sleeper
        delay = Seconds(2.0)

        # When
        sleeper.sleep(delay)

        # Then
        self.assertEqual(fake_sleeper.delays, (delay,))


if __name__ == "__main__":
    unittest.main()
