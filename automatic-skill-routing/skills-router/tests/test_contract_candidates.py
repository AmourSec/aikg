from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from typing import assert_never

from runtime.contracts import (
    ActivationConsent,
    Ambiguous,
    CandidateId,
    Candidates,
    DisplayName,
    InvalidResponse,
    InvalidResponseReason,
    LoadRequest,
    NetworkConsent,
    NoMatch,
    ProviderId,
    ProviderResult,
    ProviderScore,
    RemoteCandidate,
    ResponseToken,
    SourceFile,
    SourceRepo,
    Unavailable,
    UnavailableReason,
)


def _remote_candidate(
    candidate_id: str = "remote-1",
    score: ProviderScore | None = ProviderScore("provider-rank:alpha"),
) -> RemoteCandidate:
    return RemoteCandidate(
        candidate_id=CandidateId(candidate_id),
        provider_id=ProviderId("ascend-kg"),
        display_name=DisplayName("remote skill"),
        source_repo=SourceRepo("org/skills"),
        source_file=SourceFile("skills/remote/SKILL.md"),
        score=score,
    )


def _provider_result_name(result: ProviderResult) -> str:
    match result:
        case Candidates():
            return "candidates"
        case NoMatch():
            return "no_match"
        case Ambiguous():
            return "ambiguous"
        case Unavailable():
            return "unavailable"
        case InvalidResponse():
            return "invalid_response"
        case unreachable:
            assert_never(unreachable)


class RemoteCandidateTests(unittest.TestCase):
    def test_remote_candidate_preserves_provenance_without_activation_content(self) -> None:
        # Given
        candidate = _remote_candidate()

        # When
        provenance = (
            candidate.candidate_id,
            candidate.provider_id,
            candidate.display_name,
            candidate.source_repo,
            candidate.source_file,
        )

        # Then
        self.assertEqual(
            provenance,
            (
                CandidateId("remote-1"),
                ProviderId("ascend-kg"),
                DisplayName("remote skill"),
                SourceRepo("org/skills"),
                SourceFile("skills/remote/SKILL.md"),
            ),
        )
        self.assertFalse(hasattr(candidate, "body"))
        self.assertFalse(hasattr(candidate, "snippet"))

    def test_provider_score_is_optional_opaque_text(self) -> None:
        # Given
        scored = _remote_candidate(score=ProviderScore("opaque:high"))
        unscored = _remote_candidate(score=None)

        # When
        scores = (scored.score, unscored.score)

        # Then
        self.assertEqual(scores, (ProviderScore("opaque:high"), None))

    def test_remote_candidate_is_frozen_and_slotted(self) -> None:
        # Given
        candidate = _remote_candidate()

        # When / Then
        with self.assertRaises(FrozenInstanceError):
            setattr(candidate, "display_name", DisplayName("changed"))
        with self.assertRaises(FrozenInstanceError):
            setattr(candidate, "content", "forbidden")


class ConsentAndProviderResultTests(unittest.TestCase):
    def test_network_and_activation_consent_expose_all_states(self) -> None:
        # Given
        network_states = tuple(str(state) for state in NetworkConsent)
        activation_states = tuple(str(state) for state in ActivationConsent)

        # When
        states = (network_states, activation_states)

        # Then
        expected = ("not_requested", "granted", "refused")
        self.assertEqual(states, (expected, expected))

    def test_unavailable_carries_each_explicit_reason(self) -> None:
        # Given
        results = tuple(Unavailable(reason=reason) for reason in UnavailableReason)

        # When
        reasons = tuple(result.reason for result in results)

        # Then
        self.assertEqual(reasons, tuple(UnavailableReason))

    def test_invalid_response_carries_each_explicit_reason(self) -> None:
        # Given
        results = tuple(
            InvalidResponse(reason=reason) for reason in InvalidResponseReason
        )

        # When
        reasons = tuple(result.reason for result in results)

        # Then
        self.assertEqual(reasons, tuple(InvalidResponseReason))

    def test_ambiguous_carries_validated_candidates_and_response_identity(self) -> None:
        # Given
        token = ResponseToken()
        candidates = (_remote_candidate(), _remote_candidate("remote-2"))

        # When
        result = Ambiguous(response_token=token, candidates=candidates)

        # Then
        self.assertIs(result.response_token, token)
        self.assertEqual(result.candidates, candidates)

    def test_provider_results_remain_distinct_exhaustive_variants(self) -> None:
        # Given
        token = ResponseToken()
        candidate = _remote_candidate()
        results: tuple[ProviderResult, ...] = (
            Candidates(response_token=token, candidates=(candidate,)),
            NoMatch(),
            Ambiguous(response_token=token, candidates=(candidate,)),
            Unavailable(reason=UnavailableReason.SERVICE),
            InvalidResponse(reason=InvalidResponseReason.INVALID_SCHEMA),
        )

        # When
        names = tuple(_provider_result_name(result) for result in results)

        # Then
        self.assertEqual(
            names,
            (
                "candidates",
                "no_match",
                "ambiguous",
                "unavailable",
                "invalid_response",
            ),
        )


class ResponseMembershipTests(unittest.TestCase):
    def test_response_tokens_use_in_memory_identity(self) -> None:
        # Given
        first = ResponseToken()
        second = ResponseToken()

        # When
        tokens = {first, second}

        # Then
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        self.assertEqual(len(tokens), 2)

    def test_membership_requires_matching_token_and_candidate_id(self) -> None:
        # Given
        token = ResponseToken()
        candidate = _remote_candidate()
        result = Candidates(response_token=token, candidates=(candidate,))
        matching = LoadRequest(
            response_token=token,
            candidate_id=candidate.candidate_id,
            network_consent=NetworkConsent.GRANTED,
            activation_consent=ActivationConsent.GRANTED,
        )
        foreign_response = LoadRequest(
            response_token=ResponseToken(),
            candidate_id=candidate.candidate_id,
            network_consent=NetworkConsent.GRANTED,
            activation_consent=ActivationConsent.GRANTED,
        )
        unknown_candidate = LoadRequest(
            response_token=token,
            candidate_id=CandidateId("remote-2"),
            network_consent=NetworkConsent.GRANTED,
            activation_consent=ActivationConsent.GRANTED,
        )

        # When
        membership = (
            result.contains(matching),
            result.contains(foreign_response),
            result.contains(unknown_candidate),
        )

        # Then
        self.assertEqual(membership, (True, False, False))


if __name__ == "__main__":
    unittest.main()
