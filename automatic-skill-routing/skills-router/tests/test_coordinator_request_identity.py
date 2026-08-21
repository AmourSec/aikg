from __future__ import annotations

import unittest

from runtime.contracts import NetworkConsent, NoMatch, SearchQuery
from runtime.coordinator import (
    Coordinator,
    NetworkConsentRequired,
    NetworkDecision,
    SelectionNoMatch,
    StartRequest,
)
from tests.fakes import FakeProvider


class CoordinatorRequestIdentityTests(unittest.TestCase):
    def test_start_freezes_canonical_utf8_body_before_network_consent(self) -> None:
        # Given
        provider = FakeProvider((), ())
        coordinator = Coordinator(provider)
        query = SearchQuery("检索昇腾算子 skill")

        # When
        state = coordinator.start((), StartRequest(query, local_only=False))

        # Then
        self.assertIsInstance(state, NetworkConsentRequired)
        assert isinstance(state, NetworkConsentRequired)
        self.assertEqual(state.prepared.query, query)
        self.assertEqual(
            state.prepared.body,
            b'{"query":"\xe6\xa3\x80\xe7\xb4\xa2\xe6\x98\x87\xe8\x85\xbe\xe7\xae\x97\xe5\xad\x90 skill","top_k":10,"with_neighbors":false}',
        )
        self.assertEqual(provider.search_requests, ())

    def test_granted_consent_sends_exact_prepared_object_and_body(self) -> None:
        # Given
        provider = FakeProvider((NoMatch(),), ())
        coordinator = Coordinator(provider)
        state = coordinator.start(
            (),
            StartRequest(SearchQuery("检索昇腾算子 skill"), local_only=False),
        )
        assert isinstance(state, NetworkConsentRequired)

        # When
        actual = coordinator.resolve_network(
            state,
            NetworkDecision(NetworkConsent.GRANTED),
        )

        # Then
        self.assertEqual(len(provider.search_requests), 1)
        request = provider.search_requests[0]
        self.assertIs(request.prepared, state.prepared)
        self.assertIs(request.prepared.body, state.prepared.body)
        self.assertIs(request.network_consent, NetworkConsent.GRANTED)
        self.assertIsInstance(actual, SelectionNoMatch)


if __name__ == "__main__":
    unittest.main()
