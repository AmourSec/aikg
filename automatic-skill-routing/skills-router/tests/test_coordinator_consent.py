from __future__ import annotations

import unittest
from pathlib import Path
from typing import assert_never

from runtime.contracts import (
    Ambiguous,
    CandidateId,
    Candidates,
    DisplayName,
    InvalidResponse,
    InvalidResponseReason,
    LocalCandidate,
    LocalScore,
    NetworkConsent,
    NoMatch,
    ProviderId,
    RemoteCandidate,
    ResponseToken,
    SearchQuery,
    SkillName,
    SourceFile,
    SourceRepo,
    Unavailable,
    UnavailableReason,
    prepare_search,
)
from runtime.coordinator import (
    CandidatesReady,
    Coordinator,
    DegradedReady,
    LocalOnlyReady,
    NetworkDecision,
    NetworkConsentRequired,
    SelectionAmbiguous,
    SelectionNoMatch,
    StartRequest,
)
from tests.fakes import FakeProvider


def _local_candidate() -> LocalCandidate:
    return LocalCandidate(
        name=SkillName("local-skill"),
        description="local fallback",
        path=Path("skills/local-skill/SKILL.md"),
        score=LocalScore(0.8),
    )


def _remote_candidate() -> RemoteCandidate:
    return RemoteCandidate(
        candidate_id=CandidateId("remote-1"),
        provider_id=ProviderId("provider"),
        display_name=DisplayName("remote skill"),
        source_repo=SourceRepo("org/repo"),
        source_file=SourceFile("skills/remote/SKILL.md"),
    )


class CoordinatorConsentTests(unittest.TestCase):
    def test_start_returns_local_only_without_provider_call(self) -> None:
        # Given
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)
        request = StartRequest(SearchQuery("local task"), local_only=True)

        # When
        state = coordinator.start(local_candidates, request)

        # Then
        self.assertEqual(state, LocalOnlyReady(local_candidates))
        self.assertEqual(provider.search_requests, ())
        self.assertEqual(provider.load_requests, ())

    def test_start_requests_network_consent_without_provider_call(self) -> None:
        # Given
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)
        query = SearchQuery("disclosed task")

        # When
        state = coordinator.start(
            local_candidates,
            StartRequest(query, local_only=False),
        )

        # Then
        self.assertEqual(
            state,
            NetworkConsentRequired(local_candidates, prepare_search(query)),
        )
        self.assertEqual(provider.search_requests, ())

    def test_not_requested_network_consent_keeps_waiting_without_search(self) -> None:
        # Given
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        query = SearchQuery("find a skill")
        state = NetworkConsentRequired((_local_candidate(),), prepare_search(query))

        # When
        actual = coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.NOT_REQUESTED),
        )

        # Then
        self.assertIs(actual, state)
        self.assertEqual(provider.search_requests, ())

    def test_refused_network_consent_returns_local_fallback_without_search(self) -> None:
        # Given
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)
        state = NetworkConsentRequired(
            local_candidates,
            prepare_search(SearchQuery("find a skill")),
        )

        # When
        actual = coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.REFUSED),
        )

        # Then
        self.assertEqual(
            actual,
            LocalOnlyReady(local_candidates, NetworkConsent.REFUSED),
        )
        self.assertEqual(provider.search_requests, ())

    def test_granted_network_consent_searches_exactly_once(self) -> None:
        # Given
        token = ResponseToken()
        result = Candidates(token, (_remote_candidate(),))
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)
        query = SearchQuery("find a skill")
        state = NetworkConsentRequired(local_candidates, prepare_search(query))

        # When
        actual = coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(actual, CandidatesReady(local_candidates, result))
        self.assertEqual(len(provider.search_requests), 1)
        self.assertEqual(provider.search_requests[0].prepared.query, query)
        self.assertEqual(provider.load_requests, ())
        self.assertFalse(hasattr(actual, "query"))

    def test_grant_uses_exact_query_disclosed_at_start(self) -> None:
        # Given
        result = NoMatch()
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        disclosed_query = SearchQuery("disclosed task A")
        match coordinator.start(
            (_local_candidate(),),
            StartRequest(disclosed_query, local_only=False),
        ):
            case NetworkConsentRequired() as state:
                pass
            case LocalOnlyReady():
                self.fail("network-enabled start must require consent")
            case unreachable:
                assert_never(unreachable)

        # When
        actual = coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertIsInstance(actual, SelectionNoMatch)
        self.assertEqual(provider.search_requests[0].prepared.query, disclosed_query)

    def test_no_match_preserves_local_lane_as_structured_degradation(self) -> None:
        # Given
        result = NoMatch()
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)

        # When
        actual = coordinator.resolve_network(
            NetworkConsentRequired(
                local_candidates,
                prepare_search(SearchQuery("find a skill")),
            ),
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(actual, SelectionNoMatch(local_candidates, result))

    def test_ambiguous_preserves_local_lane_and_validated_response(self) -> None:
        # Given
        token = ResponseToken()
        result = Ambiguous(token, (_remote_candidate(),))
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)

        # When
        actual = coordinator.resolve_network(
            NetworkConsentRequired(
                local_candidates,
                prepare_search(SearchQuery("find a skill")),
            ),
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(actual, SelectionAmbiguous(local_candidates, result))

    def test_unavailable_preserves_local_lane_as_structured_degradation(self) -> None:
        # Given
        result = Unavailable(UnavailableReason.SERVICE)
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)

        # When
        actual = coordinator.resolve_network(
            NetworkConsentRequired(
                local_candidates,
                prepare_search(SearchQuery("find a skill")),
            ),
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(actual, DegradedReady(local_candidates, (result,)))

    def test_invalid_response_preserves_local_lane_as_structured_degradation(self) -> None:
        # Given
        result = InvalidResponse(InvalidResponseReason.INVALID_SCHEMA)
        provider = FakeProvider((result,), ())
        coordinator = Coordinator(provider)
        local_candidates = (_local_candidate(),)

        # When
        actual = coordinator.resolve_network(
            NetworkConsentRequired(
                local_candidates,
                prepare_search(SearchQuery("find a skill")),
            ),
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(actual, DegradedReady(local_candidates, (result,)))


if __name__ == "__main__":
    unittest.main()
