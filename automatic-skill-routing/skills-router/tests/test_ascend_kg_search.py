from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from runtime.ascend_kg import AscendKgProvider
from runtime.contracts import (
    ByteLimit,
    CandidateId,
    Candidates,
    ContentTrust,
    DisplayName,
    Endpoint,
    HttpMethod,
    InvalidResponse,
    InvalidResponseReason,
    NetworkConsent,
    NoMatch,
    ProviderId,
    ProviderScore,
    Seconds,
    SearchQuery,
    SourceFile,
    SourceRepo,
    Unavailable,
    UnavailableReason,
)
from tests.ascend_kg_support import json_response, make_provider, search_request


class AscendKgSearchRequestTests(unittest.TestCase):
    def test_search_posts_authenticated_bounded_request(self) -> None:
        # Given
        provider, transport, _ = make_provider((json_response({"results": []}),))

        # When
        result = provider.search(search_request())

        # Then
        self.assertIsInstance(result, NoMatch)
        sent = transport.requests[0]
        self.assertEqual(sent.method, HttpMethod.POST)
        self.assertEqual(sent.endpoint, Endpoint("https://ascend.wiki/search"))
        self.assertEqual(dict(sent.headers)["X-API-Key"], "secret-key")
        self.assertEqual(
            json.loads(sent.body),
            {"query": "find an NPU skill", "top_k": 10, "with_neighbors": False},
        )
        self.assertEqual(sent.timeout_seconds, Seconds(10.0))
        self.assertEqual(sent.max_response_bytes, ByteLimit(1_048_576))

    def test_search_sends_canonical_utf8_body(self) -> None:
        # Given
        provider, transport, _ = make_provider((json_response({"results": []}),))
        request = search_request(query=SearchQuery("查找 NPU skill"))

        # When
        provider.search(request)

        # Then
        self.assertIs(transport.requests[0].body, request.prepared.body)
        self.assertEqual(
            transport.requests[0].body,
            '{"query":"查找 NPU skill","top_k":10,"with_neighbors":false}'.encode(),
        )

    def test_search_without_key_is_unavailable_without_transport(self) -> None:
        # Given
        provider, transport, _ = make_provider(api_key=None)

        # When
        result = provider.search(search_request())

        # Then
        self.assertEqual(result, Unavailable(UnavailableReason.NO_API_KEY))
        self.assertEqual(transport.requests, ())

    def test_search_treats_blank_key_as_missing(self) -> None:
        # Given
        provider, transport, _ = make_provider(api_key="   ")

        # When
        result = provider.search(search_request())

        # Then
        self.assertEqual(result, Unavailable(UnavailableReason.NO_API_KEY))
        self.assertEqual(transport.requests, ())

    def test_provider_reads_api_key_from_environment(self) -> None:
        # Given
        _, transport, sleeper = make_provider((json_response({"results": []}),))

        # When
        with patch.dict(os.environ, {"ASCEND_KG_API_KEY": " env-key "}):
            provider = AscendKgProvider.from_environment(transport, sleeper)
            provider.search(search_request())

        # Then
        self.assertEqual(dict(transport.requests[0].headers)["X-API-Key"], "env-key")

    def test_search_rejects_non_header_safe_key_without_transport(self) -> None:
        for api_key in ("bad\nkey", "密钥"):
            with self.subTest(api_key=api_key):
                # Given
                provider, transport, _ = make_provider(api_key=api_key)

                # When
                result = provider.search(search_request())

                # Then
                self.assertEqual(
                    result,
                    Unavailable(UnavailableReason.CONFIGURATION),
                )
                self.assertEqual(transport.requests, ())

    def test_search_requires_granted_network_consent_without_transport(self) -> None:
        for consent in (NetworkConsent.NOT_REQUESTED, NetworkConsent.REFUSED):
            with self.subTest(consent=consent):
                # Given
                provider, transport, _ = make_provider()

                # When
                result = provider.search(search_request(consent))

                # Then
                self.assertEqual(
                    result,
                    InvalidResponse(InvalidResponseReason.CONSENT_REQUIRED),
                )
                self.assertEqual(transport.requests, ())


class AscendKgSearchParsingTests(unittest.TestCase):
    def test_search_accepts_results_and_preserves_order_and_scores(self) -> None:
        # Given
        payload = {
            "results": [
                {
                    "id": "first",
                    "source_repo": "org/skills",
                    "source_file": "skills/vector/SKILL.md",
                    "score": 0.75,
                },
                {
                    "id": "second",
                    "source_repo": "org/other",
                    "source_file": "guides/README.md",
                    "score": "provider:high",
                },
            ],
        }
        provider, _, _ = make_provider((json_response(payload),))

        # When
        result = provider.search(search_request())

        # Then
        self.assertIsInstance(result, Candidates)
        assert isinstance(result, Candidates)
        self.assertEqual(
            result.response_token.candidate_ids,
            (CandidateId("first"), CandidateId("second")),
        )
        self.assertEqual(
            tuple(candidate.display_name for candidate in result.candidates),
            (DisplayName("vector"), DisplayName("README.md")),
        )
        self.assertEqual(
            tuple(candidate.score for candidate in result.candidates),
            (ProviderScore("0.75"), ProviderScore("provider:high")),
        )
        self.assertEqual(
            tuple(candidate.provider_id for candidate in result.candidates),
            (ProviderId("ascend-kg"), ProviderId("ascend-kg")),
        )
        self.assertEqual(result.candidates[0].source_repo, SourceRepo("org/skills"))
        self.assertEqual(
            result.candidates[0].source_file,
            SourceFile("skills/vector/SKILL.md"),
        )
        self.assertEqual(
            tuple((candidate.trust, candidate.policy_authority) for candidate in result.candidates),
            ((ContentTrust.UNTRUSTED_EXTERNAL, False),) * 2,
        )

    def test_search_accepts_data_and_legacy_node_id(self) -> None:
        # Given
        provider, _, _ = make_provider(
            (
                json_response(
                    {
                        "data": [
                            {
                                "node_id": "legacy-id",
                                "source_repo": "org/skills",
                                "source_file": "legacy/SKILL.md",
                            },
                        ],
                    },
                ),
            ),
        )

        # When
        result = provider.search(search_request())

        # Then
        self.assertIsInstance(result, Candidates)
        assert isinstance(result, Candidates)
        self.assertEqual(result.candidates[0].candidate_id, CandidateId("legacy-id"))
        self.assertIsNone(result.candidates[0].score)

    def test_search_accepts_equal_primary_and_legacy_ids(self) -> None:
        # Given
        provider, _, _ = make_provider(
            (
                json_response(
                    {
                        "results": [
                            {
                                "id": "same-id",
                                "node_id": "same-id",
                                "source_repo": "org/skills",
                                "source_file": "same/SKILL.md",
                            },
                        ],
                    },
                ),
            ),
        )

        # When
        result = provider.search(search_request())

        # Then
        self.assertIsInstance(result, Candidates)
        assert isinstance(result, Candidates)
        self.assertEqual(result.candidates[0].candidate_id, CandidateId("same-id"))

    def test_search_maps_empty_list_to_no_match(self) -> None:
        # Given
        provider, _, _ = make_provider((json_response({"data": []}),))

        # When
        result = provider.search(search_request())

        # Then
        self.assertEqual(result, NoMatch())


if __name__ == "__main__":
    unittest.main()
